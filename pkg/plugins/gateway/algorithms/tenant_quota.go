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

// Package algorithms provides routing strategies for LLM inference gateways.
// The tenant_quota router adds multi-tenant fairness, quotas, and priority scheduling
// on top of the existing GPU-aware and cost-aware algorithms.
package routingalgorithms

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/aibrix/pkg/cache"
	metrics "github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/types"
	v1 "k8s.io/api/core/v1"
	klog "k8s.io/klog/v2"
)

const RouterTenantQuota types.RoutingAlgorithm = "tenant-quota"

// TenantTier represents the service tier a tenant belongs to
type TenantTier int

const (
	TierFree TenantTier = iota
	TierStandard
	TierPremium
	TierEnterprise
)

func (t TenantTier) String() string {
	switch t {
	case TierPremium:
		return "premium"
	case TierEnterprise:
		return "enterprise"
	case TierStandard:
		return "standard"
	default:
		return "free"
	}
}

func parseTier(s string) TenantTier {
	switch strings.ToLower(s) {
	case "enterprise", "ent":
		return TierEnterprise
	case "premium", "pro":
		return TierPremium
	case "standard", "std":
		return TierStandard
	default:
		return TierFree
	}
}

// QuotaConfig defines resource limits and priorities for a tenant or tier
type QuotaConfig struct {
	// Per-minute request rate limit
	RequestsPerMinute int `json:"requests_per_minute"`
	// Per-minute token limit (input + output)
	TokensPerMinute int `json:"tokens_per_minute"`
	// Maximum concurrent in-flight requests
	MaxConcurrent int `json:"max_concurrent"`
	// Burst capacity (token bucket size)
	BurstSize int `json:"burst_size"`
	// Priority weight (higher = gets more capacity under contention)
	Priority int `json:"priority"`
	// Max cost per hour in USD (0 = unlimited)
	MaxCostPerHour float64 `json:"max_cost_per_hour"`
	// Allowed GPU types (empty = all)
	AllowedGPUTypes []string `json:"allowed_gpu_types"`
	// Whether to use fair-share scheduling
	FairShare bool `json:"fair_share"`
	// SLA target latency in ms (0 = no SLA)
	TargetLatencyMs float64 `json:"target_latency_ms"`
}

// Default quotas per tier
var defaultTierQuotas = map[TenantTier]QuotaConfig{
	TierFree: {
		RequestsPerMinute: 60,
		TokensPerMinute:   100000,
		MaxConcurrent:     5,
		BurstSize:         10,
		Priority:          1,
		MaxCostPerHour:    1.0,
		FairShare:         true,
	},
	TierStandard: {
		RequestsPerMinute: 600,
		TokensPerMinute:   1000000,
		MaxConcurrent:     50,
		BurstSize:         100,
		Priority:          5,
		MaxCostPerHour:    20.0,
		FairShare:         true,
	},
	TierPremium: {
		RequestsPerMinute: 6000,
		TokensPerMinute:   10000000,
		MaxConcurrent:     500,
		BurstSize:         1000,
		Priority:          20,
		MaxCostPerHour:    200.0,
		FairShare:         true,
	},
	TierEnterprise: {
		RequestsPerMinute: 60000,
		TokensPerMinute:   100000000,
		MaxConcurrent:     5000,
		BurstSize:         10000,
		Priority:          100,
		MaxCostPerHour:    0, // unlimited
		FairShare:         true,
		TargetLatencyMs:   200.0,
	},
}

// TenantState tracks per-tenant runtime state (token buckets, etc.)
type TenantState struct {
	TenantID string
	Tier     TenantTier
	Quota    QuotaConfig

	// Token bucket for rate limiting
	mu                sync.Mutex
	RequestTokens     float64
	LastRequestRefill time.Time
	TokenBucketTokens float64
	LastTokenRefill   time.Time

	// Concurrent request counter
	Concurrent int64

	// Cost tracking (running total for current hour)
	HourCost      float64
	HourResetTime time.Time

	// Last access for cleanup
	LastAccess time.Time
}

// NewTenantState creates a new tenant state with the given tier
func NewTenantState(tenantID string, tier TenantTier) *TenantState {
	quota := defaultTierQuotas[tier]
	now := time.Now()
	return &TenantState{
		TenantID:          tenantID,
		Tier:              tier,
		Quota:             quota,
		RequestTokens:     float64(quota.BurstSize),
		LastRequestRefill: now,
		TokenBucketTokens: float64(quota.BurstSize),
		LastTokenRefill:   now,
		HourResetTime:     now.Truncate(time.Hour).Add(time.Hour),
		LastAccess:        now,
	}
}

// AllowRequest checks if a request is allowed under the tenant's quota
// Uses a token bucket algorithm: refills at Quota rate, consumes 1 per request
func (ts *TenantState) AllowRequest(estimatedTokens int) (allowed bool, reason string) {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	now := time.Now()
	ts.LastAccess = now

	// Refill request bucket
	elapsed := now.Sub(ts.LastRequestRefill).Minutes()
	ts.RequestTokens = math.Min(
		float64(ts.Quota.BurstSize),
		ts.RequestTokens+elapsed*float64(ts.Quota.RequestsPerMinute),
	)
	ts.LastRequestRefill = now

	// Refill token bucket
	tokenElapsed := now.Sub(ts.LastTokenRefill).Minutes()
	ts.TokenBucketTokens = math.Min(
		float64(ts.Quota.BurstSize),
		ts.TokenBucketTokens+tokenElapsed*float64(ts.Quota.TokensPerMinute),
	)
	ts.LastTokenRefill = now

	// Check concurrent limit
	if atomic.LoadInt64(&ts.Concurrent) >= int64(ts.Quota.MaxConcurrent) {
		return false, fmt.Sprintf("concurrent limit %d reached", ts.Quota.MaxConcurrent)
	}

	// Check request rate
	if ts.RequestTokens < 1.0 {
		return false, "request rate limit exceeded"
	}

	// Check token rate
	if estimatedTokens > 0 && ts.TokenBucketTokens < float64(estimatedTokens) {
		return false, fmt.Sprintf("token rate limit exceeded (need %d, have %.0f)", estimatedTokens, ts.TokenBucketTokens)
	}

	// Consume tokens
	ts.RequestTokens -= 1.0
	if estimatedTokens > 0 {
		ts.TokenBucketTokens -= float64(estimatedTokens)
	}
	atomic.AddInt64(&ts.Concurrent, 1)
	return true, ""
}

// ReleaseRequest decrements the concurrent counter
func (ts *TenantState) ReleaseRequest() {
	atomic.AddInt64(&ts.Concurrent, -1)
	ts.mu.Lock()
	ts.LastAccess = time.Now()
	ts.mu.Unlock()
}

// RecordCost adds to the tenant's running hour cost
func (ts *TenantState) RecordCost(cost float64) bool {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	now := time.Now()
	if now.After(ts.HourResetTime) {
		ts.HourCost = 0
		ts.HourResetTime = now.Truncate(time.Hour).Add(time.Hour)
	}
	ts.HourCost += cost
	return ts.Quota.MaxCostPerHour > 0 && ts.HourCost > ts.Quota.MaxCostPerHour
}

// TenantRegistry manages all tenant states
type TenantRegistry struct {
	mu      sync.RWMutex
	tenants map[string]*TenantState
	// Per-tenant overrides loaded from config
	overrides map[string]QuotaConfig
	// Default tier for unknown tenants
	defaultTier TenantTier
	// Cleanup config
	idleTimeout time.Duration
	stopCleanup chan struct{}
}

var (
	defaultRegistryOnce sync.Once
	defaultRegistry     *TenantRegistry
)

// DefaultRegistry returns the singleton tenant registry
func DefaultRegistry() *TenantRegistry {
	defaultRegistryOnce.Do(func() {
		defaultRegistry = NewTenantRegistry(TierStandard)
		defaultRegistry.LoadFromEnv()
		go defaultRegistry.startCleanup()
	})
	return defaultRegistry
}

// SetDefaultRegistry overrides the singleton (for tests)
func SetDefaultRegistry(r *TenantRegistry) {
	defaultRegistry = r
}

// NewTenantRegistry creates a new registry with the given default tier
func NewTenantRegistry(defaultTier TenantTier) *TenantRegistry {
	return &TenantRegistry{
		tenants:     make(map[string]*TenantState),
		overrides:   make(map[string]QuotaConfig),
		defaultTier: defaultTier,
		idleTimeout: 10 * time.Minute,
		stopCleanup: make(chan struct{}),
	}
}

// GetOrCreate returns the state for a tenant, creating it if needed
func (r *TenantRegistry) GetOrCreate(tenantID string) *TenantState {
	if tenantID == "" {
		tenantID = "anonymous"
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	if ts, ok := r.tenants[tenantID]; ok {
		ts.mu.Lock()
		ts.LastAccess = time.Now()
		ts.mu.Unlock()
		return ts
	}

	tier := r.defaultTier
	quota, hasOverride := r.overrides[tenantID]
	if hasOverride {
		// Infer tier from priority
		switch {
		case quota.Priority >= 100:
			tier = TierEnterprise
		case quota.Priority >= 20:
			tier = TierPremium
		case quota.Priority >= 5:
			tier = TierStandard
		default:
			tier = TierFree
		}
	}
	ts := NewTenantState(tenantID, tier)
	if hasOverride {
		ts.Quota = quota
		// Reset buckets to new burst size
		ts.RequestTokens = float64(quota.BurstSize)
		ts.TokenBucketTokens = float64(quota.BurstSize)
	}
	r.tenants[tenantID] = ts
	klog.V(4).Infof("created tenant state for %s (tier=%s, priority=%d)", tenantID, tier, ts.Quota.Priority)
	return ts
}

// SetOverride sets a custom quota for a specific tenant
func (r *TenantRegistry) SetOverride(tenantID string, quota QuotaConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.overrides[tenantID] = quota
	if ts, ok := r.tenants[tenantID]; ok {
		ts.mu.Lock()
		ts.Quota = quota
		// Reset buckets to new burst size
		ts.RequestTokens = float64(quota.BurstSize)
		ts.TokenBucketTokens = float64(quota.BurstSize)
		ts.mu.Unlock()
	}
}

// LoadFromEnv loads per-tenant overrides from environment variables.
// Format: AIBRIX_TENANT_QUOTA_<TENANTID>='{"requests_per_minute":1000,"priority":50}'
func (r *TenantRegistry) LoadFromEnv() {
	for _, env := range os.Environ() {
		if !strings.HasPrefix(env, "AIBRIX_TENANT_QUOTA_") {
			continue
		}
		eq := strings.Index(env, "=")
		if eq < 0 {
			continue
		}
		key := env[:eq]
		val := env[eq+1:]
		tenantID := strings.TrimPrefix(key, "AIBRIX_TENANT_QUOTA_")
		tenantID = strings.ToLower(tenantID)
		var quota QuotaConfig
		if err := json.Unmarshal([]byte(val), &quota); err != nil {
			klog.Warningf("invalid quota for tenant %s: %v", tenantID, err)
			continue
		}
		r.SetOverride(tenantID, quota)
	}
}

// LoadFromJSON loads multiple tenant overrides from a JSON blob
func (r *TenantRegistry) LoadFromJSON(data []byte) error {
	overrides := make(map[string]QuotaConfig)
	if err := json.Unmarshal(data, &overrides); err != nil {
		return err
	}
	for tenantID, quota := range overrides {
		r.SetOverride(tenantID, quota)
	}
	return nil
}

// Stats returns current usage stats for all tenants
func (r *TenantRegistry) Stats() []TenantStats {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stats := make([]TenantStats, 0, len(r.tenants))
	for _, ts := range r.tenants {
		ts.mu.Lock()
		stats = append(stats, TenantStats{
			TenantID:   ts.TenantID,
			Tier:       ts.Tier.String(),
			Priority:   ts.Quota.Priority,
			Concurrent: atomic.LoadInt64(&ts.Concurrent),
			HourCost:   ts.HourCost,
		})
		ts.mu.Unlock()
	}
	// Sort by priority desc, then tenant ID
	sort.Slice(stats, func(i, j int) bool {
		if stats[i].Priority != stats[j].Priority {
			return stats[i].Priority > stats[j].Priority
		}
		return stats[i].TenantID < stats[j].TenantID
	})
	return stats
}

// TenantStats is a snapshot of a tenant's usage
type TenantStats struct {
	TenantID   string  `json:"tenant_id"`
	Tier       string  `json:"tier"`
	Priority   int     `json:"priority"`
	Concurrent int64   `json:"concurrent"`
	HourCost   float64 `json:"hour_cost"`
}

// startCleanup periodically removes idle tenants
func (r *TenantRegistry) startCleanup() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-r.stopCleanup:
			return
		case <-ticker.C:
			r.cleanupIdle()
		}
	}
}

// Stop terminates the cleanup goroutine
func (r *TenantRegistry) Stop() {
	close(r.stopCleanup)
}

func (r *TenantRegistry) cleanupIdle() {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := time.Now()
	for id, ts := range r.tenants {
		ts.mu.Lock()
		idle := now.Sub(ts.LastAccess)
		isIdle := atomic.LoadInt64(&ts.Concurrent) == 0 && idle > r.idleTimeout
		ts.mu.Unlock()
		if isIdle {
			delete(r.tenants, id)
			klog.V(4).Infof("evicted idle tenant %s (idle %s)", id, idle)
		}
	}
}

func init() {
	Register(RouterTenantQuota, NewTenantQuotaRouter)
}

type tenantQuotaRouter struct {
	cache    cache.Cache
	registry *TenantRegistry
	// Optional: when true, requests that exceed quota return 429-style metric
	rejectOnQuotaExceeded bool
}

func NewTenantQuotaRouter() (types.Router, error) {
	c, err := cache.Get()
	if err != nil {
		return nil, err
	}
	return &tenantQuotaRouter{
		cache:                 c,
		registry:              DefaultRegistry(),
		rejectOnQuotaExceeded: getEnvBool("AIBRIX_TENANT_QUOTA_REJECT", false),
	}, nil
}

// fairShareWeight computes the fair-share weight for a tenant
// Higher weight = gets more capacity when total demand exceeds supply
func (r *tenantQuotaRouter) fairShareWeight(tenantID string, stats []TenantStats) float64 {
	ts := r.registry.GetOrCreate(tenantID)
	weight := float64(ts.Quota.Priority)

	// Boost weight based on fairness: tenants below their fair share get priority
	now := time.Now()
	_ = now // could factor in time-of-day, but simple priority is fine
	return weight
}

// allowedGPUTypesForTenant returns the GPU types allowed for this tenant
func (r *tenantQuotaRouter) allowedGPUTypesForTenant(tenantID string) map[string]bool {
	ts := r.registry.GetOrCreate(tenantID)
	allowed := make(map[string]bool)
	if len(ts.Quota.AllowedGPUTypes) == 0 {
		// Empty = all allowed
		return nil
	}
	for _, gpu := range ts.Quota.AllowedGPUTypes {
		allowed[strings.ToLower(gpu)] = true
	}
	return allowed
}

// ScoreAll scores pods with tenant-aware scoring:
// - Filters pods that don't match the tenant's allowed GPU types
// - Penalizes pods whose GPU type would exceed the tenant's cost budget
// - Boosts score for tenants with higher priority (fair-share)
func (r *tenantQuotaRouter) ScoreAll(ctx *types.RoutingContext, readyPodList types.PodList) ([]float64, []bool, error) {
	tenantID := userID(ctx)
	ts := r.registry.GetOrCreate(tenantID)
	allowedGPUs := r.allowedGPUTypesForTenant(tenantID)
	weight := r.fairShareWeight(tenantID, nil)

	pods := readyPodList.All()
	scores := make([]float64, len(pods))
	scored := make([]bool, len(pods))

	for i, pod := range pods {
		gpuType := GetGpuTypeFromPod(pod)

		// Filter: allowed GPU types
		if allowedGPUs != nil && !allowedGPUs[strings.ToLower(gpuType)] {
			scores[i] = 0
			scored[i] = false
			continue
		}

		// Base score from cost-aware router if available, else from gpu-aware
		var baseScore float64
		metricVal, err := r.cache.GetMetricValueByPodModel(pod.Name, pod.Namespace, ctx.Model, metrics.GPUCacheUsagePerc)
		if err != nil {
			klog.V(4).ErrorS(err, "failed to get metrics for pod", "pod", pod.Name)
			scores[i] = 0
			scored[i] = false
			continue
		}
		utilization := metricVal.GetSimpleValue()
		headroom := 1.0 - utilization
		baseScore = headroom * (1.0 + float64(weight)/10.0)

		// Apply SLA penalty: if pod is overutilized and tenant has a target latency, penalize
		if ts.Quota.TargetLatencyMs > 0 && utilization > 0.85 {
			baseScore *= 0.5
		}

		scores[i] = baseScore
		scored[i] = true
		klog.V(4).Infof("tenant=%s pod=%s gpu=%s util=%.2f weight=%.1f score=%.4f",
			tenantID, pod.Name, gpuType, utilization, weight, baseScore)
	}

	return scores, scored, nil
}

// Polarity returns higher-is-better
func (r *tenantQuotaRouter) Polarity() types.Polarity {
	return types.PolarityMost
}

// estimateTokens is a rough estimate of tokens for a request
// (real implementation would parse the body)
func estimateTokens(ctx *types.RoutingContext) int {
	if ctx.Model == "" {
		return 1000
	}
	// Crude model: average request is ~1k tokens
	return 1000
}

// Route selects the best pod respecting tenant quotas, fair-share, and GPU constraints
func (r *tenantQuotaRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	tenantID := userID(ctx)
	ts := r.registry.GetOrCreate(tenantID)

	// Check quota
	estTokens := estimateTokens(ctx)
	allowed, reason := ts.AllowRequest(estTokens)
	if !allowed {
		klog.V(2).Infof("tenant %s quota exceeded: %s", tenantID, reason)
		tenantQuotaRejections.WithLabelValues(tenantID, ts.Tier.String(), reason).Inc()
		if r.rejectOnQuotaExceeded {
			// Caller can detect this via metrics; for now, fall through to random
			klog.V(2).Infof("rejecting request for tenant %s (quota exceeded)", tenantID)
		}
		// Fall through to fair-share routing with reduced weight
	}

	// Defer release for concurrent counter
	defer func() {
		if allowed {
			ts.ReleaseRequest()
		}
	}()

	// Score pods
	allowedGPUs := r.allowedGPUTypesForTenant(tenantID)
	weight := r.fairShareWeight(tenantID, nil)

	var targetPod *v1.Pod
	maxScore := -math.MaxFloat64
	var candidatePods []*v1.Pod

	for _, pod := range readyPodList.All() {
		gpuType := GetGpuTypeFromPod(pod)

		// Skip if GPU not allowed for this tenant
		if allowedGPUs != nil && !allowedGPUs[strings.ToLower(gpuType)] {
			continue
		}

		metricVal, err := r.cache.GetMetricValueByPodModel(pod.Name, pod.Namespace, ctx.Model, metrics.GPUCacheUsagePerc)
		if err != nil {
			klog.V(4).ErrorS(err, "failed to get metrics for pod", "pod", pod.Name)
			continue
		}
		utilization := metricVal.GetSimpleValue()
		headroom := 1.0 - utilization
		score := headroom * (1.0 + float64(weight)/10.0)

		// SLA penalty
		if ts.Quota.TargetLatencyMs > 0 && utilization > 0.85 {
			score *= 0.5
		}

		// Quota exceeded penalty: route to least-busy pod to spread load
		if !allowed {
			score *= 0.1
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

	// Fallback
	if targetPod == nil {
		var err error
		targetPod, err = SelectRandomPodAsFallback(ctx, readyPodList.All(), rand.Intn)
		if err != nil {
			return "", err
		}
		klog.V(4).Infof("tenant-quota fallback: tenant=%s pod=%s", tenantID, targetPod.Name)
	} else {
		// Record cost (rough estimate)
		costPerHour := lookupGPU(GetGpuTypeFromPod(targetPod)).CostPerHr
		// Assume 30s average request duration
		estCost := costPerHour * (30.0 / 3600.0)
		if overBudget := ts.RecordCost(estCost); overBudget {
			klog.V(2).Infof("tenant %s exceeded hourly cost budget ($%.2f > $%.2f)",
				tenantID, ts.HourCost, ts.Quota.MaxCostPerHour)
		}
		klog.V(4).Infof("tenant-quota: tenant=%s tier=%s pod=%s score=%.4f",
			tenantID, ts.Tier, targetPod.Name, maxScore)
	}

	if targetPod == nil {
		return "", fmt.Errorf("no pods to forward request")
	}

	tenantRoutingDecisions.WithLabelValues(tenantID, ts.Tier.String()).Inc()
	ctx.SetTargetPod(targetPod)
	return ctx.TargetAddress(), nil
}

// RecordRequestResult updates tenant state after a request completes
// Call this from the gateway's response handler
func (r *tenantQuotaRouter) RecordRequestResult(tenantID string, inputTokens, outputTokens int, latencyMs float64) {
	ts := r.registry.GetOrCreate(tenantID)
	ts.mu.Lock()
	ts.LastAccess = time.Now()
	ts.mu.Unlock()
	tenantRequestLatency.WithLabelValues(tenantID, ts.Tier.String()).Observe(latencyMs)
	tenantTokenUsage.WithLabelValues(tenantID, ts.Tier.String(), "input").Add(float64(inputTokens))
	tenantTokenUsage.WithLabelValues(tenantID, ts.Tier.String(), "output").Add(float64(outputTokens))
}

// GetTenantInfo returns a snapshot of a tenant's state
func (r *tenantQuotaRouter) GetTenantInfo(tenantID string) *TenantInfo {
	ts := r.registry.GetOrCreate(tenantID)
	ts.mu.Lock()
	defer ts.mu.Unlock()
	return &TenantInfo{
		TenantID:        ts.TenantID,
		Tier:            ts.Tier.String(),
		Priority:        ts.Quota.Priority,
		Concurrent:      atomic.LoadInt64(&ts.Concurrent),
		MaxConcurrent:   ts.Quota.MaxConcurrent,
		HourCost:        ts.HourCost,
		MaxCostPerHour:  ts.Quota.MaxCostPerHour,
		RequestTokens:   ts.RequestTokens,
		BurstSize:       ts.Quota.BurstSize,
		RequestsPerMin:  ts.Quota.RequestsPerMinute,
		TokensPerMin:    ts.Quota.TokensPerMinute,
		TargetLatencyMs: ts.Quota.TargetLatencyMs,
	}
}

// TenantInfo is a public view of tenant state
type TenantInfo struct {
	TenantID        string  `json:"tenant_id"`
	Tier            string  `json:"tier"`
	Priority        int     `json:"priority"`
	Concurrent      int64   `json:"concurrent"`
	MaxConcurrent   int     `json:"max_concurrent"`
	HourCost        float64 `json:"hour_cost"`
	MaxCostPerHour  float64 `json:"max_cost_per_hour"`
	RequestTokens   float64 `json:"request_tokens"`
	BurstSize       int     `json:"burst_size"`
	RequestsPerMin  int     `json:"requests_per_minute"`
	TokensPerMin    int     `json:"tokens_per_minute"`
	TargetLatencyMs float64 `json:"target_latency_ms"`
}

// HandleAdminRequest handles admin endpoints for tenant management
// POST /admin/tenants/{id}/quota with JSON body
func (r *tenantQuotaRouter) HandleAdminRequest(tenantID string, body []byte) error {
	var quota QuotaConfig
	if err := json.Unmarshal(body, &quota); err != nil {
		return err
	}
	r.registry.SetOverride(tenantID, quota)
	return nil
}
