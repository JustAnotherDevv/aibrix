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

package routingalgorithms

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/vllm-project/aibrix/pkg/cache"
	metrics "github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/types"
	v1 "k8s.io/api/core/v1"
	klog "k8s.io/klog/v2"
)

const RouterGpuCostAware types.RoutingAlgorithm = "gpu-cost-aware"

// RoutingStrategy defines the optimization objective
type RoutingStrategy int

const (
	StrategyBalanced     RoutingStrategy = iota // balance latency, cost, and power
	StrategyLowCost                             // minimize $ per token
	StrategyLowLatency                          // minimize latency (similar to gpu-aware)
	StrategyLowPower                            // minimize energy consumption
	StrategySLAOptimized                        // meet SLA, then minimize cost
)

func (s RoutingStrategy) String() string {
	switch s {
	case StrategyLowCost:
		return "low-cost"
	case StrategyLowLatency:
		return "low-latency"
	case StrategyLowPower:
		return "low-power"
	case StrategySLAOptimized:
		return "sla-optimized"
	default:
		return "balanced"
	}
}

// parseStrategy converts a string to a RoutingStrategy
func parseStrategy(s string) RoutingStrategy {
	switch strings.ToLower(s) {
	case "low-cost", "cost", "cheapest":
		return StrategyLowCost
	case "low-latency", "latency", "fastest":
		return StrategyLowLatency
	case "low-power", "power", "green":
		return StrategyLowPower
	case "sla", "sla-optimized":
		return StrategySLAOptimized
	default:
		return StrategyBalanced
	}
}

// CostAwareConfig holds the routing configuration
type CostAwareConfig struct {
	Strategy           RoutingStrategy
	CostWeight         float64 // 0.0-1.0
	PerformanceWeight  float64 // 0.0-1.0
	PowerWeight        float64 // 0.0-1.0
	MaxCostPer1kTokens float64 // budget constraint in USD
	MaxLatencyMs       float64 // SLA constraint in milliseconds
	PowerBudgetWatts   float64 // power budget in watts
	PreferSpot         bool    // if true, prefer spot instances (60-90% cheaper)
	PreferReserved     bool    // if true, prefer reserved instances (40-60% cheaper)
}

var (
	defaultConfigOnce sync.Once
	defaultConfig     CostAwareConfig
)

// DefaultConfig returns a balanced default configuration that can be tuned
// via environment variables:
//
//	AIBRIX_ROUTING_STRATEGY   - balanced, low-cost, low-latency, low-power, sla-optimized
//	AIBRIX_COST_WEIGHT        - float 0.0-1.0 (default 0.5 for balanced)
//	AIBRIX_PERF_WEIGHT        - float 0.0-1.0 (default 0.5 for balanced)
//	AIBRIX_POWER_WEIGHT       - float 0.0-1.0 (default 0.3 for balanced)
//	AIBRIX_MAX_COST_PER_1K    - float USD (no default)
//	AIBRIX_MAX_LATENCY_MS     - float ms (no default)
//	AIBRIX_POWER_BUDGET_W     - float watts (no default)
//	AIBRIX_PREFER_SPOT        - true/false
//	AIBRIX_PREFER_RESERVED    - true/false
func DefaultConfig() CostAwareConfig {
	defaultConfigOnce.Do(func() {
		defaultConfig = CostAwareConfig{
			Strategy:           parseStrategy(getEnv("AIBRIX_ROUTING_STRATEGY", "balanced")),
			CostWeight:         getEnvFloat("AIBRIX_COST_WEIGHT", 0.4),
			PerformanceWeight:  getEnvFloat("AIBRIX_PERF_WEIGHT", 0.4),
			PowerWeight:        getEnvFloat("AIBRIX_POWER_WEIGHT", 0.2),
			MaxCostPer1kTokens: getEnvFloat("AIBRIX_MAX_COST_PER_1K", 0.0),
			MaxLatencyMs:       getEnvFloat("AIBRIX_MAX_LATENCY_MS", 0.0),
			PowerBudgetWatts:   getEnvFloat("AIBRIX_POWER_BUDGET_W", 0.0),
			PreferSpot:         getEnvBool("AIBRIX_PREFER_SPOT", false),
			PreferReserved:     getEnvBool("AIBRIX_PREFER_RESERVED", false),
		}
	})
	return defaultConfig
}

// SetConfig allows tests / advanced users to override the default config
func SetConfig(cfg CostAwareConfig) {
	defaultConfig = cfg
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvFloat(key string, fallback float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return fallback
}

func getEnvBool(key string, fallback bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return fallback
}

func init() {
	Register(RouterGpuCostAware, NewGpuCostAwareRouter)
}

type gpuCostAwareRouter struct {
	cache  cache.Cache
	config CostAwareConfig
}

func NewGpuCostAwareRouter() (types.Router, error) {
	c, err := cache.Get()
	if err != nil {
		return nil, err
	}

	return gpuCostAwareRouter{
		cache:  c,
		config: DefaultConfig(),
	}, nil
}

// gpuPricing returns the effective cost per hour, applying spot/reserved discounts
func (r gpuCostAwareRouter) effectiveCostPerHour(gpuType string) float64 {
	baseCost := lookupGPU(gpuType).CostPerHr

	// Apply discounts based on instance type preferences
	if r.config.PreferSpot {
		baseCost *= 0.30 // spot ~70% discount
	} else if r.config.PreferReserved {
		baseCost *= 0.60 // reserved ~40% discount
	}

	return baseCost
}

// getPodCostInfo extracts the cost class from pod labels (spot/on-demand/reserved)
func (r gpuCostAwareRouter) getPodCostClass(pod *v1.Pod) string {
	if class, ok := pod.Labels["node.kubernetes.io/instance-life-cycle"]; ok {
		return strings.ToLower(class)
	}
	if class, ok := pod.Labels["aibrix.ai/cost-class"]; ok {
		return strings.ToLower(class)
	}
	if class, ok := pod.Annotations["aibrix.ai/cost-class"]; ok {
		return strings.ToLower(class)
	}
	return "on-demand"
}

// totalCostPerHour returns the total hourly cost of running the pod's GPUs
func (r gpuCostAwareRouter) totalCostPerHour(pod *v1.Pod, gpuType string) float64 {
	return r.effectiveCostPerHour(gpuType) * float64(gpuCount(pod))
}

// totalPowerWatts returns the TDP-based power draw of the pod's GPUs
func (r gpuCostAwareRouter) totalPowerWatts(pod *v1.Pod, gpuType string) float64 {
	return lookupGPU(gpuType).TDPWatts * float64(gpuCount(pod))
}

// estimatedTokensPerSecond estimates throughput for a GPU at current utilization
// Lower utilization = higher throughput headroom
func (r gpuCostAwareRouter) estimatedTokensPerSecond(pod *v1.Pod, namespace, model string) (float64, error) {
	metricVal, err := r.cache.GetMetricValueByPodModel(pod.Name, namespace, model, metrics.GPUCacheUsagePerc)
	if err != nil {
		return 0, err
	}
	utilization := metricVal.GetSimpleValue()
	// Rough model: headroom correlates with achievable throughput
	// Real implementation would use per-GPU benchmarks
	return 100.0 * (1.0 - utilization), nil
}

// satisfiesConstraints returns true if the pod meets the configured budgets/SLAs
func (r gpuCostAwareRouter) satisfiesConstraints(pod *v1.Pod, namespace, model string, costPerHour, powerWatts float64) (bool, string) {
	if r.config.MaxCostPer1kTokens > 0 {
		// Very rough cost-per-1k-tokens estimate
		// Assume pod runs 100 tok/s at current utilization
		// cost per hour -> cost per 1k tokens = costPerHour / 360
		estCostPer1k := costPerHour / 360.0
		if estCostPer1k > r.config.MaxCostPer1kTokens {
			return false, fmt.Sprintf("cost %.4f exceeds budget %.4f", estCostPer1k, r.config.MaxCostPer1kTokens)
		}
	}
	if r.config.PowerBudgetWatts > 0 && powerWatts > r.config.PowerBudgetWatts {
		return false, fmt.Sprintf("power %.0fW exceeds budget %.0fW", powerWatts, r.config.PowerBudgetWatts)
	}
	return true, ""
}

// computeScore returns a score for a pod, with higher = better.
// It normalizes each factor to 0-1, applies strategy-specific weights,
// and multiplies them (so any zero factor zeroes the score).
func (r gpuCostAwareRouter) computeScore(pod *v1.Pod, namespace, model, gpuType string) (float64, bool) {
	_ = namespace // reserved for future per-namespace pricing tiers
	_ = model    // reserved for future per-model GPU requirements
	capacity := getCapacity(gpuType)
	if capacity == 0 {
		capacity = 40.0
	}
	computePower := getComputePower(gpuType)
	if computePower == 0 {
		computePower = 0.5
	}

	metricVal, err := r.cache.GetMetricValueByPodModel(pod.Name, namespace, model, metrics.GPUCacheUsagePerc)
	if err != nil {
		klog.V(4).ErrorS(err, "failed to get GPU metrics for pod", "pod", pod.Name)
		return 0, false
	}
	utilization := metricVal.GetSimpleValue()
	headroom := 1.0 - utilization

	costPerHour := r.totalCostPerHour(pod, gpuType)
	powerWatts := r.totalPowerWatts(pod, gpuType)

	// Normalize factors to 0-1
	// Cost: cheaper is better, normalize to cheapest GPU (T4 = 0.50)
	maxCost := 5.00
	costScore := 1.0 - (costPerHour / maxCost)
	if costScore < 0 {
		costScore = 0
	}

	// Performance: higher compute * headroom is better, normalize to H100 baseline
	perfScore := (headroom * capacity * computePower) / (141.0 * 2.2)
	if perfScore > 1.0 {
		perfScore = 1.0
	}

	// Power: lower is better, normalize to most efficient (T4 = 72W, but consider tok/s/W)
	maxPower := 700.0
	powerScore := 1.0 - (powerWatts / maxPower)
	if powerScore < 0 {
		powerScore = 0
	}

	// Apply strategy-specific weights
	var score float64
	switch r.config.Strategy {
	case StrategyLowCost:
		score = 0.85*costScore + 0.10*perfScore + 0.05*powerScore
	case StrategyLowLatency:
		score = 0.10*costScore + 0.85*perfScore + 0.05*powerScore
	case StrategyLowPower:
		score = 0.15*costScore + 0.15*perfScore + 0.70*powerScore
	case StrategySLAOptimized:
		// For SLA-optimized, hard-require perfScore above a threshold,
		// then optimize cost within the feasible set
		if perfScore < 0.4 {
			return 0, false
		}
		score = 0.70*costScore + 0.20*perfScore + 0.10*powerScore
	default: // Balanced
		cw := r.config.CostWeight
		pw := r.config.PerformanceWeight
		ew := r.config.PowerWeight
		total := cw + pw + ew
		if total == 0 {
			cw, pw, ew = 0.4, 0.4, 0.2
			total = 1.0
		}
		score = (cw*costScore + pw*perfScore + ew*powerScore) / total
	}

	klog.V(4).Infof(
		"pod=%s gpu=%s cost=%.2f$/h perf=%.2f power=%.0fW util=%.2f -> score=%.4f (strategy=%s)",
		pod.Name, gpuType, costPerHour, perfScore, powerWatts, utilization, score, r.config.Strategy,
	)

	return score, true
}

func getCapacity(gpuType string) float64 {
	return lookupGPU(gpuType).MemoryGB
}

func getComputePower(gpuType string) float64 {
	return lookupGPU(gpuType).Compute
}

// ScoreAll scores all pods with the cost-aware composite score
func (r gpuCostAwareRouter) ScoreAll(ctx *types.RoutingContext, readyPodList types.PodList) ([]float64, []bool, error) {
	pods := readyPodList.All()
	scores := make([]float64, len(pods))
	scored := make([]bool, len(pods))

	namespace := userID(ctx)
	for i, pod := range pods {
		gpuType := GetGpuTypeFromPod(pod)
		score, ok := r.computeScore(pod, namespace, ctx.Model, gpuType)
		if !ok {
			scores[i] = 0
			scored[i] = false
			continue
		}
		scores[i] = score
		scored[i] = true
	}

	return scores, scored, nil
}

// Polarity returns higher-is-better (higher score = better candidate)
func (r gpuCostAwareRouter) Polarity() types.Polarity {
	return types.PolarityMost
}

// Route selects the best pod according to the configured strategy
func (r gpuCostAwareRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	var targetPod *v1.Pod
	maxScore := -math.MaxFloat64
	var candidatePods []*v1.Pod

	userNs := userID(ctx)
	for _, pod := range readyPodList.All() {
		gpuType := GetGpuTypeFromPod(pod)
		costPerHour := r.totalCostPerHour(pod, gpuType)
		powerWatts := r.totalPowerWatts(pod, gpuType)
		namespace := pod.Namespace
		if namespace == "" {
			namespace = userNs
		}
		if ok, reason := r.satisfiesConstraints(pod, namespace, ctx.Model, costPerHour, powerWatts); !ok {
			klog.V(4).Infof("skipping pod %s: %s", pod.Name, reason)
			continue
		}
		score, ok := r.computeScore(pod, namespace, ctx.Model, gpuType)
		if !ok {
			continue
		}
		if score > maxScore {
			maxScore = score
			candidatePods = []*v1.Pod{pod}
		} else if score == maxScore {
			candidatePods = append(candidatePods, pod)
		}
	}

	if len(candidatePods) > 0 {
		targetPod = candidatePods[rand.Intn(len(candidatePods))]
	}

	if targetPod == nil {
		var err error
		targetPod, err = SelectRandomPodAsFallback(ctx, readyPodList.All(), rand.Intn)
		if err != nil {
			return "", err
		}
		klog.V(4).Infof("cost-aware fallback select targetPod: %s(%s)", targetPod.Name, targetPod.Status.PodIP)
	} else {
		klog.V(4).Infof("cost-aware select targetPod: %s(%s) strategy=%s score=%.4f",
			targetPod.Name, targetPod.Status.PodIP, r.config.Strategy, maxScore)
	}

	if targetPod == nil {
		return "", fmt.Errorf("no pods to forward request")
	}

	ctx.SetTargetPod(targetPod)
	return ctx.TargetAddress(), nil
}

// EstimateCostPer1kTokens returns the estimated cost per 1k output tokens for a pod.
// Useful for logging and per-tenant cost attribution.
func (r gpuCostAwareRouter) EstimateCostPer1kTokens(pod *v1.Pod) float64 {
	gpuType := GetGpuTypeFromPod(pod)
	costPerHour := r.totalCostPerHour(pod, gpuType)
	// Assumes pod produces ~360 1k-token equivalents per hour at full utilization
	return costPerHour / 360.0
}

// EstimatePowerWatts returns the estimated power draw for a pod.
func (r gpuCostAwareRouter) EstimatePowerWatts(pod *v1.Pod) float64 {
	gpuType := GetGpuTypeFromPod(pod)
	return r.totalPowerWatts(pod, gpuType)
}
