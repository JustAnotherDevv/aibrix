/*
Copyright 2024 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package algorithm

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"

	autoscalingv1alpha1 "github.com/vllm-project/aibrix/api/autoscaling/v1alpha1"
	"github.com/vllm-project/aibrix/pkg/controller/podautoscaler/types"
)

// CPAAlgorithm implements Cost-aware Pod Autoscaling.
//
// CPA picks the cheapest (replica count x GPU type) combination that meets
// the throughput target. It uses a static catalog of per-GPU-type throughput
// (req/s per pod) and pricing (USD/hour) and runs a small integer program:
//
//  1. For each candidate GPU type, compute required_pods = ceil(observed_qps / throughput).
//  2. Total cost = required_pods * cost_per_pod.
//  3. Pick the candidate with the lowest cost that satisfies:
//     - minReplicas <= required_pods <= maxReplicas
//     - cost_per_hour <= MaxCostPerHour (if set)
//     - cost_class preference (if set)
//
// The algorithm then applies the same max-scale-up / max-scale-down rates and
// tolerance windows as KPA/APA so behavior is consistent across strategies.
type CPAAlgorithm struct {
	mu sync.RWMutex
	// catalog holds the throughput + cost data. Defaults are loaded at first
	// use; tests may inject a custom catalog.
	catalog *GPUCostCatalog

	// overrides is the per-PA CPA configuration. Populated by the
	// controller via LoadOverridesFromSpec. Keyed by "namespace/name".
	overrides map[string]CPAOverride
}

// Ensure interface compliance.
var _ ScalingAlgorithm = (*CPAAlgorithm)(nil)

// CPAMode picks the optimization objective.
type CPAMode string

const (
	CPAModeMinCost       CPAMode = "min-cost"       // pick cheapest config that meets QPS
	CPAModeBalanced      CPAMode = "balanced"       // default; cost + perf tradeoff
	CPAModeMinLatency    CPAMode = "min-latency"    // pick fastest GPU first, then cheapest
	CPAModeSpotFirst     CPAMode = "spot-first"     // strongly prefer spot instances
	CPAModeReservedFirst CPAMode = "reserved-first" // prefer reserved instances
	CPAModeDefault       CPAMode = CPAModeBalanced
)

// CostClass reflects the pricing class of a pod.
type CostClass string

const (
	CostClassOnDemand CostClass = "on-demand"
	CostClassSpot     CostClass = "spot"
	CostClassReserved CostClass = "reserved"
)

// GPUCostEntry is one row in the cost+throughput catalog.
type GPUCostEntry struct {
	GPUType           string    // normalized key, e.g. "nvidia-h100-80gb"
	CostClass         CostClass // on-demand | spot | reserved
	HourlyCostUSD     float64   // $/hour per pod (one GPU by default)
	ThroughputQPS     float64   // sustained LLM requests/sec per pod (single GPU)
	ThroughputTokens  float64   // sustained LLM tokens/sec per pod
	MemoryGB          float64   // GPU HBM in GB
	ComputeCapability float64   // relative FLOPS/price score (0-1)
}

// GPUCostCatalog is the throughput + price database.
//
// Values are typical/measured rather than peak; in production the catalog
// would be a ConfigMap updated by the gpu-optimizer Python service.
type GPUCostCatalog struct {
	mu       sync.RWMutex
	entries  map[string]GPUCostEntry
	defaults map[string]GPUCostEntry // keyed by GPU type, on-demand
}

// NewDefaultGPUCostCatalog returns a catalog populated with common GPU types.
// Throughput numbers are calibrated for a 7B-parameter chat model at batch=8.
func NewDefaultGPUCostCatalog() *GPUCostCatalog {
	c := &GPUCostCatalog{
		entries:  make(map[string]GPUCostEntry),
		defaults: make(map[string]GPUCostEntry),
	}

	// (gpu_type, on_demand $/hr, throughput req/s, tokens/s, memory GB, compute score)
	type row struct {
		gpu     string
		cost    float64
		qps     float64
		tps     float64
		mem     float64
		compute float64
	}
	defaults := []row{
		{"nvidia-h200-141gb", 5.00, 22.0, 4400, 141, 1.00},
		{"nvidia-h100-80gb", 4.00, 20.0, 4000, 80, 0.95},
		{"nvidia-a100-80gb", 2.50, 14.0, 2800, 80, 0.65},
		{"nvidia-a100-40gb", 2.00, 12.0, 2400, 40, 0.60},
		{"nvidia-l40s-48gb", 1.20, 8.0, 1600, 48, 0.40},
		{"nvidia-l4-24gb", 0.80, 4.5, 900, 24, 0.20},
		{"nvidia-t4-16gb", 0.50, 2.5, 500, 16, 0.12},
		{"nvidia-a10g-24gb", 0.90, 5.0, 1000, 24, 0.25},
		{"nvidia-v100-32gb", 1.80, 6.0, 1200, 32, 0.30},
	}

	for _, r := range defaults {
		entry := GPUCostEntry{
			GPUType: r.gpu, CostClass: CostClassOnDemand,
			HourlyCostUSD: r.cost, ThroughputQPS: r.qps, ThroughputTokens: r.tps,
			MemoryGB: r.mem, ComputeCapability: r.compute,
		}
		c.defaults[r.gpu] = entry
		c.entries[catalogKey(r.gpu, CostClassOnDemand)] = entry
		// Spot = 30% of on-demand, Reserved = 60% of on-demand.
		c.entries[catalogKey(r.gpu, CostClassSpot)] = GPUCostEntry{
			GPUType: r.gpu, CostClass: CostClassSpot,
			HourlyCostUSD: roundCents(r.cost * 0.30), ThroughputQPS: r.qps, ThroughputTokens: r.tps,
			MemoryGB: r.mem, ComputeCapability: r.compute,
		}
		c.entries[catalogKey(r.gpu, CostClassReserved)] = GPUCostEntry{
			GPUType: r.gpu, CostClass: CostClassReserved,
			HourlyCostUSD: roundCents(r.cost * 0.60), ThroughputQPS: r.qps, ThroughputTokens: r.tps,
			MemoryGB: r.mem, ComputeCapability: r.compute,
		}
	}
	return c
}

// Get returns the entry for a (GPU type, cost class) pair, falling back to
// on-demand if the class is unknown.
func (c *GPUCostCatalog) Get(gpuType string, class CostClass) (GPUCostEntry, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if e, ok := c.entries[catalogKey(gpuType, class)]; ok {
		return e, true
	}
	if class == CostClassOnDemand {
		return GPUCostEntry{}, false
	}
	if e, ok := c.entries[catalogKey(gpuType, CostClassOnDemand)]; ok {
		return e, true
	}
	return GPUCostEntry{}, false
}

// Set replaces the entry for a (GPU type, cost class) pair.
func (c *GPUCostCatalog) Set(entry GPUCostEntry) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if entry.CostClass == "" {
		entry.CostClass = CostClassOnDemand
	}
	c.entries[catalogKey(entry.GPUType, entry.CostClass)] = entry
	if entry.CostClass == CostClassOnDemand {
		c.defaults[entry.GPUType] = entry
	}
}

// AllOnDemand returns the on-demand entries, sorted by GPU type.
func (c *GPUCostCatalog) AllOnDemand() []GPUCostEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]GPUCostEntry, 0, len(c.defaults))
	for _, e := range c.defaults {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].GPUType < out[j].GPUType })
	return out
}

// catalogKey is the map key for a (gpu, class) pair.
func catalogKey(gpu string, class CostClass) string {
	return string(class) + "|" + gpu
}

func roundCents(v float64) float64 {
	return math.Round(v*100) / 100
}

// CPAConfig holds the cost-aware configuration derived from PodAutoscaler
// annotations / spec. It is recomputed every ComputeRecommendation call so
// annotations and spec changes take effect immediately.
type CPAConfig struct {
	Mode               CPAMode
	MaxCostPerHour     float64
	CandidateGPUTypes  []string
	PreferredCostClass CostClass
	TargetValue        float64
	ScaleUpCooldown    int // seconds
	ScaleDownCooldown  int
	UpTolerance        float64
	DownTolerance      float64
	MaxScaleUpRate     float64
	MaxScaleDownRate   float64
}

// NewCPAAlgorithm returns a CPA algorithm with the default catalog. Use
// SetCatalog to inject a custom one in tests.
func NewCPAAlgorithm() *CPAAlgorithm {
	return &CPAAlgorithm{catalog: NewDefaultGPUCostCatalog()}
}

// SetCatalog replaces the catalog.
func (a *CPAAlgorithm) SetCatalog(c *GPUCostCatalog) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.catalog = c
}

// GetAlgorithmType returns "cpa".
func (a *CPAAlgorithm) GetAlgorithmType() string {
	return "cpa"
}

// ComputeRecommendation implements ScalingAlgorithm.
//
// Unlike KPA/APA which always return the same GPU type as the workload,
// CPA may switch the recommended GPU type to a cheaper one if the candidate
// set and budgets allow. The "desired type" is reported in Metadata so the
// upstream controller can request a different pod template if needed.
func (a *CPAAlgorithm) ComputeRecommendation(ctx context.Context, request ScalingRequest) (*ScalingRecommendation, error) {
	metrics := request.AggregatedMetrics
	if metrics == nil {
		return nil, fmt.Errorf("cpa: nil aggregated metrics")
	}

	cfg := a.loadConfig(request)

	observedQPS := metrics.StableValue
	if observedQPS < 0 {
		observedQPS = 0
	}

	// Find the cheapest (replicas x GPU type) configuration that meets QPS.
	plan, err := a.optimalPlan(observedQPS, cfg)
	if err != nil {
		return nil, err
	}

	desired := a.applyRateLimits(float64(plan.Replicas), float64(request.CurrentReplicas), cfg)
	desired = applyConstraints(desired, request.ScalingContext)

	confidence := metrics.Confidence
	if confidence == 0 {
		confidence = 1.0
	}

	return &ScalingRecommendation{
		DesiredReplicas: desired,
		Confidence:      confidence,
		Reason: fmt.Sprintf("cpa: %dx %s @ $%.2f/hr (qps=%.1f, mode=%s)",
			plan.Replicas, plan.Entry.GPUType, plan.TotalHourlyCost, observedQPS, cfg.Mode),
		Algorithm:  "cpa",
		ScaleValid: true,
		Metadata: map[string]interface{}{
			"selected_gpu_type":      plan.Entry.GPUType,
			"selected_cost_class":    string(plan.Entry.CostClass),
			"hourly_cost_usd":        plan.TotalHourlyCost,
			"per_pod_cost_usd":       plan.Entry.HourlyCostUSD,
			"throughput_qps_per_pod": plan.Entry.ThroughputQPS,
			"required_replicas":      plan.Replicas,
			"observed_qps":           observedQPS,
			"mode":                   string(cfg.Mode),
			"candidates_considered":  len(plan.CandidatesConsidered),
		},
	}, nil
}

// plan is the result of optimalPlan: the chosen entry, its replica count,
// and the total hourly cost.
type plan struct {
	Entry                GPUCostEntry
	Replicas             int32
	TotalHourlyCost      float64
	CandidatesConsidered []GPUCostEntry
}

// optimalPlan finds the cheapest (replicas x GPU type) configuration that
// satisfies the QPS target and all configured constraints.
func (a *CPAAlgorithm) optimalPlan(observedQPS float64, cfg CPAConfig) (plan, error) {
	a.mu.RLock()
	catalog := a.catalog
	a.mu.RUnlock()
	if catalog == nil {
		catalog = NewDefaultGPUCostCatalog()
	}

	candidates := a.candidateEntries(catalog, cfg)
	if len(candidates) == 0 {
		return plan{}, fmt.Errorf("cpa: no candidate GPU types configured")
	}

	var best plan
	best.TotalHourlyCost = math.Inf(1)
	for _, entry := range candidates {
		replicas := replicasForQPS(observedQPS, entry, cfg)
		if replicas <= 0 {
			replicas = 1
		}
		totalCost := float64(replicas) * entry.HourlyCostUSD
		if cfg.MaxCostPerHour > 0 && totalCost > cfg.MaxCostPerHour {
			continue
		}
		// For min-latency: take the first feasible (which is the fastest
		// because candidateEntries sorts by compute). For all other modes:
		// take the cheapest.
		pickFirst := cfg.Mode == CPAModeMinLatency
		if pickFirst || totalCost < best.TotalHourlyCost {
			best = plan{
				Entry:                entry,
				Replicas:             replicas,
				TotalHourlyCost:      totalCost,
				CandidatesConsidered: candidates,
			}
			if pickFirst {
				break
			}
		}
	}
	if math.IsInf(best.TotalHourlyCost, 1) {
		// Fallback: budget was too tight for every candidate. Pick the
		// cheapest raw cost so we degrade gracefully instead of erroring.
		for _, entry := range candidates {
			replicas := replicasForQPS(observedQPS, entry, cfg)
			if replicas <= 0 {
				replicas = 1
			}
			totalCost := float64(replicas) * entry.HourlyCostUSD
			if totalCost < best.TotalHourlyCost {
				best = plan{
					Entry:                entry,
					Replicas:             replicas,
					TotalHourlyCost:      totalCost,
					CandidatesConsidered: candidates,
				}
			}
		}
	}
	return best, nil
}

// candidateEntries returns the catalog entries CPA is allowed to pick from.
// The mode controls both which cost classes are loaded and the order they
// are returned in, so the scoring loop in optimalPlan() can simply take the
// first feasible entry.
func (a *CPAAlgorithm) candidateEntries(catalog *GPUCostCatalog, cfg CPAConfig) []GPUCostEntry {
	mode := cfg.Mode
	if mode == "" {
		mode = CPAModeDefault
	}

	// Resolve the effective cost class to load from the catalog.
	class := cfg.PreferredCostClass
	if class == "" || class == CostClass("any") {
		switch mode {
		case CPAModeSpotFirst:
			class = CostClassSpot
		case CPAModeReservedFirst:
			class = CostClassReserved
		default:
			class = CostClassOnDemand
		}
	}

	var entries []GPUCostEntry
	if len(cfg.CandidateGPUTypes) == 0 {
		for _, e := range catalog.AllOnDemand() {
			if loaded, ok := catalog.Get(e.GPUType, class); ok {
				entries = append(entries, loaded)
			}
		}
	} else {
		for _, gpu := range cfg.CandidateGPUTypes {
			if e, ok := catalog.Get(gpu, class); ok {
				entries = append(entries, e)
			} else if e, ok := catalog.Get(gpu, CostClassOnDemand); ok {
				entries = append(entries, e)
			}
		}
	}

	// Mode-based ordering.
	switch mode {
	case CPAModeSpotFirst, CPAModeReservedFirst:
		// Already filtered to one class; sort by raw cost so the first
		// feasible (in optimalPlan) is also the cheapest of that class.
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].HourlyCostUSD < entries[j].HourlyCostUSD
		})
	case CPAModeMinLatency:
		// Highest compute capability AND highest throughput first.
		sort.Slice(entries, func(i, j int) bool {
			if entries[i].ComputeCapability != entries[j].ComputeCapability {
				return entries[i].ComputeCapability > entries[j].ComputeCapability
			}
			return entries[i].ThroughputQPS > entries[j].ThroughputQPS
		})
	case CPAModeMinCost, CPAModeBalanced:
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].HourlyCostUSD < entries[j].HourlyCostUSD
		})
	}
	return entries
}

// replicasForQPS computes required replicas to serve observedQPS on a given
// GPU type. Honors the min/max from ScalingContext.
func replicasForQPS(observedQPS float64, entry GPUCostEntry, cfg CPAConfig) int32 {
	if observedQPS <= 0 {
		return cfg.minReplicas()
	}
	required := int32(math.Ceil(observedQPS / entry.ThroughputQPS))
	return clampReplicas(required, cfg)
}

func (c CPAConfig) minReplicas() int32 {
	return 1
}

func clampReplicas(replicas int32, cfg CPAConfig) int32 {
	if replicas < 1 {
		replicas = 1
	}
	if cfg.MaxScaleUpRate > 0 {
		// Soft cap: don't grow absurdly in one step. Caller will
		// tighten with applyRateLimits anyway.
		_ = cfg.MaxScaleUpRate
	}
	return replicas
}

// applyRateLimits caps the recommendation using the same max-scale-up /
// max-scale-down envelope as KPA/APA.
func (a *CPAAlgorithm) applyRateLimits(desired, current float64, cfg CPAConfig) int32 {
	if cfg.MaxScaleUpRate > 0 {
		maxUp := math.Max(1, math.Ceil(cfg.MaxScaleUpRate*current))
		if desired > maxUp {
			desired = maxUp
		}
	}
	if cfg.MaxScaleDownRate > 0 {
		maxDown := math.Floor(current / cfg.MaxScaleDownRate)
		if desired < maxDown {
			desired = maxDown
		}
	}
	return int32(desired)
}

// loadConfig reads CPA configuration from environment variables and the
// PodAutoscaler spec/annotations. Env vars provide cluster-wide defaults;
// spec fields provide per-PA overrides.
//
// Note: PodAutoscaler is not directly accessible from ScalingRequest (only
// ScaleTarget is). For per-PA customization, callers should plumb the PA
// through ScalingContext.UpdateByPaTypes(). Here we use env vars + a thread-
// safe override map.
func (a *CPAAlgorithm) loadConfig(request ScalingRequest) CPAConfig {
	cfg := CPAConfig{
		Mode:               CPAModeDefault,
		PreferredCostClass: "",
		UpTolerance:        request.ScalingContext.GetUpFluctuationTolerance(),
		DownTolerance:      request.ScalingContext.GetDownFluctuationTolerance(),
		MaxScaleUpRate:     request.ScalingContext.GetMaxScaleUpRate(),
		MaxScaleDownRate:   request.ScalingContext.GetMaxScaleDownRate(),
		ScaleUpCooldown:    int(request.ScalingContext.GetScaleUpCooldownWindow().Seconds()),
		ScaleDownCooldown:  int(request.ScalingContext.GetScaleDownCooldownWindow().Seconds()),
		TargetValue:        defaultCPATargetValue,
	}

	// Per-PA overrides keyed by namespace/name.
	key := paKey(request.Target)
	a.mu.RLock()
	override, hasOverride := a.overrides[key]
	a.mu.RUnlock()

	if hasOverride {
		if override.Mode != "" {
			cfg.Mode = override.Mode
		}
		if override.MaxCostPerHour > 0 {
			cfg.MaxCostPerHour = override.MaxCostPerHour
		}
		if len(override.CandidateGPUTypes) > 0 {
			cfg.CandidateGPUTypes = override.CandidateGPUTypes
		}
		if override.PreferredCostClass != "" {
			cfg.PreferredCostClass = override.PreferredCostClass
		}
	}

	// Env vars provide cluster-wide fallback.
	if v := os.Getenv("AIBRIX_CPA_MODE"); v != "" && cfg.Mode == CPAModeDefault {
		cfg.Mode = CPAMode(strings.ToLower(v))
	}
	if v := os.Getenv("AIBRIX_CPA_BUDGET"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && cfg.MaxCostPerHour == 0 {
			cfg.MaxCostPerHour = f
		}
	}
	if v := os.Getenv("AIBRIX_CPA_COST_CLASS"); v != "" && cfg.PreferredCostClass == "" {
		cfg.PreferredCostClass = CostClass(strings.ToLower(v))
	}
	if v := os.Getenv("AIBRIX_CPA_GPU_TYPES"); v != "" {
		if len(cfg.CandidateGPUTypes) == 0 {
			cfg.CandidateGPUTypes = splitCSV(v)
		}
	}
	return cfg
}

// CPAOverride holds per-PA CPA configuration. Set via SetOverride() so the
// controller can plumb CRD spec values into the algorithm.
type CPAOverride struct {
	Mode               CPAMode
	MaxCostPerHour     float64
	CandidateGPUTypes  []string
	PreferredCostClass CostClass
}

// SetOverride configures CPA for a specific PodAutoscaler. The PA is
// identified by namespace/name, matching ScaleTarget.
func (a *CPAAlgorithm) SetOverride(target types.ScaleTarget, o CPAOverride) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.overrides == nil {
		a.overrides = make(map[string]CPAOverride)
	}
	a.overrides[paKey(target)] = o
}

// ClearOverride removes a per-PA CPA override.
func (a *CPAAlgorithm) ClearOverride(target types.ScaleTarget) {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.overrides, paKey(target))
}

// LoadOverridesFromSpec applies spec fields to the algorithm's override map.
// This is the integration point for the controller.
func (a *CPAAlgorithm) LoadOverridesFromSpec(pa autoscalingv1alpha1.PodAutoscaler) {
	if pa.Spec.CostOptimization == nil {
		a.ClearOverride(types.ScaleTarget{
			Namespace: pa.Namespace,
			Name:      pa.Name,
		})
		return
	}
	spec := pa.Spec.CostOptimization
	a.SetOverride(types.ScaleTarget{
		Namespace: pa.Namespace,
		Name:      pa.Name,
	}, CPAOverride{
		Mode:               CPAMode(spec.Mode),
		MaxCostPerHour:     spec.MaxCostPerHour,
		CandidateGPUTypes:  spec.CandidateGPUTypes,
		PreferredCostClass: CostClass(spec.PreferredCostClass),
	})
}

func paKey(t types.ScaleTarget) string {
	return t.Namespace + "/" + t.Name
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

const defaultCPATargetValue = 50.0

// CostAwarePlan is the human-readable summary of a CPA decision. Exposed for
// logging, metrics, and tests.
type CostAwarePlan struct {
	SelectedGPUType     string  `json:"selected_gpu_type"`
	SelectedCostClass   string  `json:"selected_cost_class"`
	Replicas            int32   `json:"replicas"`
	HourlyCostUSD       float64 `json:"hourly_cost_usd"`
	PerPodCostUSD       float64 `json:"per_pod_cost_usd"`
	ThroughputQPSPerPod float64 `json:"throughput_qps_per_pod"`
	ObservedQPS         float64 `json:"observed_qps"`
	Mode                string  `json:"mode"`
}

// String returns a one-line summary.
func (p CostAwarePlan) String() string {
	return fmt.Sprintf("%dx %s(%s) @ $%.2f/hr total, serves %.1f QPS at $%.4f/qps",
		p.Replicas, p.SelectedGPUType, p.SelectedCostClass,
		p.HourlyCostUSD, p.ObservedQPS, costPerQPS(p))
}

// CostPerQPS returns the dollars per QPS-hour the plan incurs.
func CostPerQPS(p CostAwarePlan) float64 {
	return costPerQPS(p)
}

func costPerQPS(p CostAwarePlan) float64 {
	if p.ObservedQPS <= 0 {
		return 0
	}
	return p.HourlyCostUSD / p.ObservedQPS
}
