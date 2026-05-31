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
	"encoding/json"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/aibrix/pkg/types"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestParseTier(t *testing.T) {
	tests := []struct {
		input    string
		expected TenantTier
	}{
		{"enterprise", TierEnterprise},
		{"ENTERPRISE", TierEnterprise},
		{"ent", TierEnterprise},
		{"premium", TierPremium},
		{"pro", TierPremium},
		{"standard", TierStandard},
		{"std", TierStandard},
		{"free", TierFree},
		{"", TierFree},
		{"unknown", TierFree},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			if got := parseTier(tt.input); got != tt.expected {
				t.Errorf("parseTier(%q) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}

func TestTierString(t *testing.T) {
	tests := []struct {
		tier     TenantTier
		expected string
	}{
		{TierEnterprise, "enterprise"},
		{TierPremium, "premium"},
		{TierStandard, "standard"},
		{TierFree, "free"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if got := tt.tier.String(); got != tt.expected {
				t.Errorf("tier.String() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestNewTenantState(t *testing.T) {
	ts := NewTenantState("acme-corp", TierPremium)
	if ts.TenantID != "acme-corp" {
		t.Errorf("TenantID = %q, want %q", ts.TenantID, "acme-corp")
	}
	if ts.Tier != TierPremium {
		t.Errorf("Tier = %v, want %v", ts.Tier, TierPremium)
	}
	expectedQuota := defaultTierQuotas[TierPremium]
	if ts.Quota.Priority != expectedQuota.Priority {
		t.Errorf("Priority = %d, want %d", ts.Quota.Priority, expectedQuota.Priority)
	}
}

func TestTenantState_AllowRequest_BurstAllowed(t *testing.T) {
	ts := NewTenantState("test", TierStandard)

	// Should be able to consume burst size quickly
	allowed := 0
	for i := 0; i < ts.Quota.BurstSize+5; i++ {
		ok, _ := ts.AllowRequest(1)
		if ok {
			allowed++
		}
		ts.ReleaseRequest()
	}

	if allowed != ts.Quota.BurstSize {
		t.Errorf("allowed = %d, want %d (burst size)", allowed, ts.Quota.BurstSize)
	}
}

func TestTenantState_AllowRequest_ConcurrentLimit(t *testing.T) {
	ts := NewTenantState("test", TierFree)

	// Free tier has max 5 concurrent
	for i := 0; i < ts.Quota.MaxConcurrent; i++ {
		ok, _ := ts.AllowRequest(1)
		if !ok {
			t.Errorf("request %d should be allowed", i)
		}
	}

	// 6th should be rejected
	ok, reason := ts.AllowRequest(1)
	if ok {
		t.Error("6th request should be rejected (concurrent limit)")
	}
	if reason == "" {
		t.Error("rejection reason should be provided")
	}
}

func TestTenantState_AllowRequest_TokenRate(t *testing.T) {
	ts := NewTenantState("test", TierStandard)

	// Consume almost all tokens
	hugeRequest := ts.Quota.TokensPerMinute / 2
	ok, _ := ts.AllowRequest(hugeRequest)
	if !ok {
		t.Skip("burst should allow first request")
	}
	ts.ReleaseRequest()

	ok, _ = ts.AllowRequest(hugeRequest)
	if !ok {
		t.Skip("burst should allow second request")
	}
	ts.ReleaseRequest()

	// Third huge request should be rejected
	ok, reason := ts.AllowRequest(hugeRequest)
	if ok {
		t.Error("third huge request should be rejected (token rate)")
	}
	if reason == "" {
		t.Error("rejection reason should mention token rate")
	}
}

func TestTenantState_RecordCost(t *testing.T) {
	ts := NewTenantState("test", TierStandard)
	initialBudget := ts.Quota.MaxCostPerHour

	// Add cost up to budget
	ok := ts.RecordCost(initialBudget - 0.01)
	if ok {
		t.Error("should not be over budget yet")
	}

	// Exceed
	ok = ts.RecordCost(0.02)
	if !ok {
		t.Error("should be over budget now")
	}
}

func TestTenantState_RecordCost_Unlimited(t *testing.T) {
	ts := NewTenantState("test", TierEnterprise)
	if ts.Quota.MaxCostPerHour != 0 {
		t.Skip("enterprise should be unlimited")
	}

	// Should never be over budget
	for i := 0; i < 100; i++ {
		ok := ts.RecordCost(1000.0)
		if ok {
			t.Error("enterprise should have unlimited budget")
		}
	}
}

func TestTenantRegistry_GetOrCreate(t *testing.T) {
	r := NewTenantRegistry(TierStandard)
	defer r.Stop()

	ts1 := r.GetOrCreate("tenant-a")
	ts2 := r.GetOrCreate("tenant-a")
	ts3 := r.GetOrCreate("tenant-b")

	if ts1 != ts2 {
		t.Error("GetOrCreate should return same instance for same ID")
	}
	if ts1 == ts3 {
		t.Error("GetOrCreate should return different instances for different IDs")
	}
}

func TestTenantRegistry_EmptyTenantID(t *testing.T) {
	r := NewTenantRegistry(TierStandard)
	defer r.Stop()

	ts := r.GetOrCreate("")
	if ts.TenantID != "anonymous" {
		t.Errorf("empty ID should map to 'anonymous', got %q", ts.TenantID)
	}
}

func TestTenantRegistry_SetOverride(t *testing.T) {
	r := NewTenantRegistry(TierStandard)
	defer r.Stop()

	// Get default
	ts1 := r.GetOrCreate("custom")
	if ts1.Quota.Priority != defaultTierQuotas[TierStandard].Priority {
		t.Errorf("default priority wrong")
	}

	// Override
	r.SetOverride("custom", QuotaConfig{
		Priority:          99,
		RequestsPerMinute: 9999,
		BurstSize:         500,
	})

	ts2 := r.GetOrCreate("custom")
	if ts2.Quota.Priority != 99 {
		t.Errorf("override priority = %d, want 99", ts2.Quota.Priority)
	}
	if ts2.Quota.RequestsPerMinute != 9999 {
		t.Errorf("override rpm = %d, want 9999", ts2.Quota.RequestsPerMinute)
	}
}

func TestTenantRegistry_LoadFromJSON(t *testing.T) {
	r := NewTenantRegistry(TierStandard)
	defer r.Stop()

	jsonData := []byte(`{
		"premium-customer": {
			"requests_per_minute": 10000,
			"priority": 50,
			"max_concurrent": 200,
			"allowed_gpu_types": ["nvidia-h100-80gb", "nvidia-a100-80gb"]
		},
		"free-tier": {
			"requests_per_minute": 10,
			"priority": 0,
			"max_concurrent": 1
		}
	}`)

	err := r.LoadFromJSON(jsonData)
	if err != nil {
		t.Fatalf("LoadFromJSON failed: %v", err)
	}

	ts := r.GetOrCreate("premium-customer")
	if ts.Quota.Priority != 50 {
		t.Errorf("priority = %d, want 50", ts.Quota.Priority)
	}
	if len(ts.Quota.AllowedGPUTypes) != 2 {
		t.Errorf("allowed GPU types = %d, want 2", len(ts.Quota.AllowedGPUTypes))
	}

	ts2 := r.GetOrCreate("free-tier")
	if ts2.Quota.Priority != 0 {
		t.Errorf("free-tier priority = %d, want 0", ts2.Quota.Priority)
	}
}

func TestTenantRegistry_LoadFromJSON_Invalid(t *testing.T) {
	r := NewTenantRegistry(TierStandard)
	defer r.Stop()

	err := r.LoadFromJSON([]byte(`not valid json`))
	if err == nil {
		t.Error("expected error on invalid JSON")
	}
}

func TestTenantRegistry_Stats(t *testing.T) {
	r := NewTenantRegistry(TierStandard)
	defer r.Stop()

	r.GetOrCreate("alice")
	r.SetOverride("bob", QuotaConfig{Priority: 99})
	r.GetOrCreate("bob")
	r.SetOverride("charlie", QuotaConfig{Priority: 10})
	r.GetOrCreate("charlie")

	stats := r.Stats()
	if len(stats) != 3 {
		t.Errorf("expected 3 stats, got %d", len(stats))
	}

	// First should be highest priority (bob = 99)
	if stats[0].TenantID != "bob" {
		t.Errorf("highest priority tenant should be first, got %q", stats[0].TenantID)
	}
}

func TestTenantQuotaRouter_AllowedGPUTypes(t *testing.T) {
	r := &tenantQuotaRouter{registry: NewTenantRegistry(TierStandard)}
	defer r.registry.Stop()

	r.registry.SetOverride("restricted-tenant", QuotaConfig{
		AllowedGPUTypes: []string{"nvidia-a100-80gb", "nvidia-h100-80gb"},
	})

	allowed := r.allowedGPUTypesForTenant("restricted-tenant")
	if !allowed["nvidia-a100-80gb"] {
		t.Error("A100 should be allowed for restricted-tenant")
	}
	if !allowed["nvidia-h100-80gb"] {
		t.Error("H100 should be allowed for restricted-tenant")
	}
	if allowed["nvidia-t4-16gb"] {
		t.Error("T4 should NOT be allowed for restricted-tenant")
	}

	// Default tier has no restrictions
	defaultAllowed := r.allowedGPUTypesForTenant("unrestricted")
	if defaultAllowed != nil {
		t.Error("unrestricted tenant should have nil (all allowed) map")
	}
}

func TestTenantQuotaRouter_FairShareWeight(t *testing.T) {
	r := &tenantQuotaRouter{registry: NewTenantRegistry(TierStandard)}
	defer r.registry.Stop()

	// Standard tier weight
	weight := r.fairShareWeight("std-tenant", nil)
	if weight != 5 {
		t.Errorf("standard tier weight = %f, want 5", weight)
	}

	// Premium tier weight
	r.registry.SetOverride("premium-tenant", QuotaConfig{Priority: 20})
	weight = r.fairShareWeight("premium-tenant", nil)
	if weight != 20 {
		t.Errorf("premium tier weight = %f, want 20", weight)
	}

	// Free tier weight
	r.registry.SetOverride("free-tenant", QuotaConfig{Priority: 1})
	weight = r.fairShareWeight("free-tenant", nil)
	if weight != 1 {
		t.Errorf("free tier weight = %f, want 1", weight)
	}
}

func TestTenantState_AllowRequest_Refill(t *testing.T) {
	ts := NewTenantState("test", TierStandard)

	// Drain the bucket
	for i := 0; i < ts.Quota.BurstSize+1; i++ {
		ts.AllowRequest(1)
		ts.ReleaseRequest()
	}

	// Manually move time back to simulate refill
	ts.mu.Lock()
	ts.LastRequestRefill = time.Now().Add(-time.Minute)
	ts.LastTokenRefill = time.Now().Add(-time.Minute)
	ts.RequestTokens = 0
	ts.TokenBucketTokens = 0
	ts.mu.Unlock()

	ok, _ := ts.AllowRequest(1)
	if !ok {
		t.Error("should allow after time-based refill")
	}
	ts.ReleaseRequest()
}

func TestTenantState_ReleaseRequest(t *testing.T) {
	ts := NewTenantState("test", TierStandard)
	ts.AllowRequest(1)
	ts.AllowRequest(1)
	if atomic.LoadInt64(&ts.Concurrent) != 2 {
		t.Errorf("concurrent = %d, want 2", atomic.LoadInt64(&ts.Concurrent))
	}
	ts.ReleaseRequest()
	if atomic.LoadInt64(&ts.Concurrent) != 1 {
		t.Errorf("concurrent after release = %d, want 1", atomic.LoadInt64(&ts.Concurrent))
	}
}

func TestEstimateTokens(t *testing.T) {
	// Default model: 1000 tokens
	if tokens := estimateTokens(&types.RoutingContext{}); tokens != 1000 {
		t.Errorf("default tokens = %d, want 1000", tokens)
	}

	// With model name
	ctx := &types.RoutingContext{Model: "llama-3.1-70b"}
	if tokens := estimateTokens(ctx); tokens != 1000 {
		t.Errorf("model-specific tokens = %d, want 1000", tokens)
	}
}

func TestDefaultTierQuotas(t *testing.T) {
	// Verify all tiers have increasing privileges
	tiers := []TenantTier{TierFree, TierStandard, TierPremium, TierEnterprise}
	for i := 0; i < len(tiers)-1; i++ {
		lower := defaultTierQuotas[tiers[i]]
		higher := defaultTierQuotas[tiers[i+1]]

		if higher.Priority <= lower.Priority {
			t.Errorf("tier %s priority %d should be less than %s priority %d",
				tiers[i], lower.Priority, tiers[i+1], higher.Priority)
		}
		if higher.RequestsPerMinute <= lower.RequestsPerMinute {
			t.Errorf("tier %s RPM %d should be less than %s RPM %d",
				tiers[i], lower.RequestsPerMinute, tiers[i+1], higher.RequestsPerMinute)
		}
		if higher.MaxConcurrent <= lower.MaxConcurrent {
			t.Errorf("tier %s concurrent %d should be less than %s concurrent %d",
				tiers[i], lower.MaxConcurrent, tiers[i+1], higher.MaxConcurrent)
		}
	}
}

func TestQuotaConfig_AllFields(t *testing.T) {
	original := QuotaConfig{
		RequestsPerMinute: 1000,
		TokensPerMinute:   5000000,
		MaxConcurrent:     100,
		BurstSize:         200,
		Priority:          50,
		MaxCostPerHour:    100.5,
		AllowedGPUTypes:   []string{"nvidia-h100-80gb", "nvidia-a100-80gb"},
		FairShare:         true,
		TargetLatencyMs:   250.0,
	}

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var decoded QuotaConfig
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if decoded.Priority != original.Priority {
		t.Errorf("priority = %d, want %d", decoded.Priority, original.Priority)
	}
	if len(decoded.AllowedGPUTypes) != 2 {
		t.Errorf("allowed GPU types count = %d, want 2", len(decoded.AllowedGPUTypes))
	}
}

func TestNewPod_Helper(t *testing.T) {
	pod := newGPUPod("test-pod", "nvidia-h100-80gb", 1)
	if pod.Name != "test-pod" {
		t.Errorf("name = %q, want test-pod", pod.Name)
	}
	if pod.Labels["gpu-type"] != "nvidia-h100-80gb" {
		t.Errorf("gpu-type = %q, want nvidia-h100-80gb", pod.Labels["gpu-type"])
	}
	if !hasGPUResource(pod, 1) {
		t.Error("pod should have 1 GPU")
	}
}

// Helper to construct a pod for tests
func newGPUPod(name, gpuType string, gpuCount int) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"gpu-type": gpuType},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "worker",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							"nvidia.com/gpu": resource.MustParse(strconv.Itoa(gpuCount)),
						},
					},
				},
			},
		},
	}
}

func hasGPUResource(pod *v1.Pod, count int) bool {
	for _, c := range pod.Spec.Containers {
		if qty, ok := c.Resources.Limits["nvidia.com/gpu"]; ok {
			return int(qty.Value()) == count
		}
	}
	return false
}
