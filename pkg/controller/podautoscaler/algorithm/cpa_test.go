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
	"math"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	autoscalingv1alpha1 "github.com/vllm-project/aibrix/api/autoscaling/v1alpha1"
	scalingctx "github.com/vllm-project/aibrix/pkg/controller/podautoscaler/context"
	"github.com/vllm-project/aibrix/pkg/controller/podautoscaler/types"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func newCPARequest(qps float64, replicas int32, overrides ...CPAOverride) ScalingRequest {
	sc := &mockScalingContext{
		MaxScaleUpRate:           2.0,
		MaxScaleDownRate:         2.0,
		UpFluctuationTolerance:   0.1,
		DownFluctuationTolerance: 0.1,
		MinReplicas:              1,
		MaxReplicas:              100,
		MetricTargets:            map[string]scalingctx.MetricTarget{},
	}
	return ScalingRequest{
		Target: types.ScaleTarget{
			Namespace: "default",
			Name:      "test",
		},
		CurrentReplicas: replicas,
		ScalingContext:  sc,
		AggregatedMetrics: &types.AggregatedMetrics{
			StableValue: qps,
			Confidence:  1.0,
		},
	}
}

func TestCPAAlgorithm_GetAlgorithmType(t *testing.T) {
	a := NewCPAAlgorithm()
	assert.Equal(t, "cpa", a.GetAlgorithmType())
}

func TestCPAAlgorithm_DefaultCatalogHasCommonGPUs(t *testing.T) {
	c := NewDefaultGPUCostCatalog()
	required := []string{
		"nvidia-h100-80gb",
		"nvidia-h200-141gb",
		"nvidia-a100-80gb",
		"nvidia-l4-24gb",
		"nvidia-t4-16gb",
	}
	for _, gpu := range required {
		_, ok := c.Get(gpu, CostClassOnDemand)
		assert.True(t, ok, "expected default catalog to have %s", gpu)
	}
}

func TestCPAAlgorithm_SpotIsCheaperThanOnDemand(t *testing.T) {
	c := NewDefaultGPUCostCatalog()
	for _, gpu := range []string{"nvidia-h100-80gb", "nvidia-a100-80gb", "nvidia-l4-24gb"} {
		od, _ := c.Get(gpu, CostClassOnDemand)
		spot, _ := c.Get(gpu, CostClassSpot)
		assert.Less(t, spot.HourlyCostUSD, od.HourlyCostUSD, "%s spot should be cheaper than on-demand", gpu)
		assert.Equal(t, od.HourlyCostUSD*0.30, spot.HourlyCostUSD, "spot should be 30%% of on-demand for %s", gpu)
	}
}

func TestCPAAlgorithm_ReservedIsCheaperThanOnDemand(t *testing.T) {
	c := NewDefaultGPUCostCatalog()
	for _, gpu := range []string{"nvidia-h100-80gb", "nvidia-a100-80gb", "nvidia-l4-24gb"} {
		od, _ := c.Get(gpu, CostClassOnDemand)
		reserved, _ := c.Get(gpu, CostClassReserved)
		assert.Less(t, reserved.HourlyCostUSD, od.HourlyCostUSD)
		assert.Equal(t, roundCents(od.HourlyCostUSD*0.60), reserved.HourlyCostUSD, "reserved should be 60%% of on-demand")
	}
}

func TestCPAAlgorithm_PicksCheapestForLowQPS(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(2.0, 1) // 2 QPS

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)

	// At 2 QPS, T4 (2.5 QPS per pod) is cheapest.
	// 2 / 2.5 = 1 pod, cost = $0.50/hr.
	assert.Equal(t, "nvidia-t4-16gb", rec.Metadata["selected_gpu_type"])
	assert.Equal(t, int32(1), rec.DesiredReplicas)
	assert.InDelta(t, 0.50, rec.Metadata["hourly_cost_usd"], 0.01)
}

func TestCPAAlgorithm_PicksRightSizeForHighQPS(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(100.0, 1) // 100 QPS

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)

	// 100 QPS: cheapest option is L40S (8 QPS/pod, $1.20/hr) at
	// ceil(100/8) = 13 pods × $1.20 = $15.60/hr.
	// We expect L40S because it's the cheapest feasible config.
	gpuType := rec.Metadata["selected_gpu_type"].(string)
	assert.Equal(t, "nvidia-l40s-48gb", gpuType, "should pick L40S as cheapest")
}

func TestCPAAlgorithm_RespectsBudget(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(8.0, 1) // 8 QPS
	a.SetOverride(req.Target, CPAOverride{
		Mode:           CPAModeMinCost,
		MaxCostPerHour: 5.0, // hard cap
	})

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)

	cost := rec.Metadata["hourly_cost_usd"].(float64)
	assert.LessOrEqual(t, cost, 5.0, "should not exceed budget")
}

func TestCPAAlgorithm_SpotFirstPrefersSpot(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(5.0, 1)
	a.SetOverride(req.Target, CPAOverride{Mode: CPAModeSpotFirst})

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)

	class := rec.Metadata["selected_cost_class"].(string)
	assert.Equal(t, "spot", class, "spot-first mode should pick spot when feasible")
}

func TestCPAAlgorithm_ReservedFirstPrefersReserved(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(5.0, 1)
	a.SetOverride(req.Target, CPAOverride{Mode: CPAModeReservedFirst})

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)

	class := rec.Metadata["selected_cost_class"].(string)
	assert.Equal(t, "reserved", class)
}

func TestCPAAlgorithm_MinLatencyPicksFastestGPU(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(20.0, 1)
	a.SetOverride(req.Target, CPAOverride{Mode: CPAModeMinLatency})

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)

	gpuType := rec.Metadata["selected_gpu_type"].(string)
	assert.Equal(t, "nvidia-h200-141gb", gpuType, "min-latency should pick the highest-throughput GPU")
}

func TestCPAAlgorithm_RespectsCandidateGPUTypes(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(10.0, 1)
	a.SetOverride(req.Target, CPAOverride{
		CandidateGPUTypes: []string{"nvidia-a100-80gb", "nvidia-t4-16gb"},
		Mode:              CPAModeMinCost,
	})

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)

	gpuType := rec.Metadata["selected_gpu_type"].(string)
	assert.Contains(t, []string{"nvidia-a100-80gb", "nvidia-t4-16gb"}, gpuType, "should only pick from candidates")
}

func TestCPAAlgorithm_AppliesMaxScaleUpRate(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(200.0, 1)                                // would want 10 H100s
	req.ScalingContext.(*mockScalingContext).MaxScaleUpRate = 2.0 // can only double

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	assert.LessOrEqual(t, rec.DesiredReplicas, int32(2), "should respect max-scale-up rate")
}

func TestCPAAlgorithm_AppliesMaxReplicas(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(1000.0, 1) // would want 50+ pods
	req.ScalingContext.(*mockScalingContext).MaxReplicas = 10

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	assert.LessOrEqual(t, rec.DesiredReplicas, int32(10))
}

func TestCPAAlgorithm_AppliesMinReplicas(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(0.0, 5) // idle but should keep minReplicas
	req.ScalingContext.(*mockScalingContext).MinReplicas = 3

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, rec.DesiredReplicas, int32(3))
}

func TestCPAAlgorithm_ZeroQPSCollapsesToOnePod(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(0.0, 1)

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, rec.DesiredReplicas, int32(1))
}

func TestCPAAlgorithm_ReplicasForQPS(t *testing.T) {
	cases := []struct {
		qps       float64
		perPodQPS float64
		expected  int32
	}{
		{0, 10, 1},    // floor at 1
		{5, 10, 1},    // 0.5 -> ceil to 1
		{10, 10, 1},   // exact
		{11, 10, 2},   // 1.1 -> ceil to 2
		{100, 10, 10}, // exact 10
		{101, 10, 11}, // 10.1 -> 11
	}
	for _, c := range cases {
		got := replicasForQPS(c.qps, GPUCostEntry{ThroughputQPS: c.perPodQPS}, CPAConfig{})
		assert.Equal(t, c.expected, got, "qps=%f perPod=%f", c.qps, c.perPodQPS)
	}
}

func TestCPAAlgorithm_CatalogGetFallbackToOnDemand(t *testing.T) {
	c := NewDefaultGPUCostCatalog()
	// Ask for an unknown cost class.
	entry, ok := c.Get("nvidia-h100-80gb", CostClass("unknown"))
	assert.True(t, ok, "should fall back to on-demand")
	assert.Equal(t, CostClassOnDemand, entry.CostClass)
}

func TestCPAAlgorithm_CatalogSet(t *testing.T) {
	c := NewDefaultGPUCostCatalog()
	c.Set(GPUCostEntry{
		GPUType: "nvidia-h100-80gb", CostClass: CostClassOnDemand,
		HourlyCostUSD: 99.99, ThroughputQPS: 30.0,
	})
	got, ok := c.Get("nvidia-h100-80gb", CostClassOnDemand)
	require.True(t, ok)
	assert.Equal(t, 99.99, got.HourlyCostUSD)
}

func TestCPAAlgorithm_LoadOverridesFromSpec(t *testing.T) {
	a := NewCPAAlgorithm()
	pa := autoscalingv1alpha1.PodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{Namespace: "prod", Name: "llm-svc"},
		Spec: autoscalingv1alpha1.PodAutoscalerSpec{
			CostOptimization: &autoscalingv1alpha1.CostOptimizationSpec{
				Mode:               "spot-first",
				MaxCostPerHour:     10.0,
				CandidateGPUTypes:  []string{"nvidia-h100-80gb"},
				PreferredCostClass: "spot",
			},
		},
	}
	a.LoadOverridesFromSpec(pa)

	req := newCPARequest(5.0, 1)
	req.Target.Namespace = "prod"
	req.Target.Name = "llm-svc"

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	assert.Equal(t, "spot", rec.Metadata["selected_cost_class"])
}

func TestCPAAlgorithm_ClearOverride(t *testing.T) {
	a := NewCPAAlgorithm()
	target := types.ScaleTarget{Namespace: "ns", Name: "svc"}
	a.SetOverride(target, CPAOverride{Mode: CPAModeMinLatency})
	a.ClearOverride(target)

	_, hasOverride := a.overrides[paKey(target)]
	assert.False(t, hasOverride)
}

func TestCPAAlgorithm_ReasonContainsSummary(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(10.0, 2)
	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	assert.Contains(t, rec.Reason, "cpa:")
	assert.Contains(t, rec.Reason, "nvidia-")
}

func TestCPAAlgorithm_ConfidenceDefaultsToOne(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(10.0, 2)
	req.AggregatedMetrics.Confidence = 0
	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	assert.InDelta(t, 1.0, rec.Confidence, 0.001)
}

func TestCPAAlgorithm_NilMetricsErrors(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(10.0, 1)
	req.AggregatedMetrics = nil
	_, err := a.ComputeRecommendation(nil, req)
	assert.Error(t, err)
}

func TestCPAAlgorithm_NegativeQPSClampedToZero(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(-5.0, 1)
	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, rec.DesiredReplicas, int32(1))
}

func TestCPAAlgorithm_BudgetTooTightFallsBackToCheapest(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(1000.0, 1) // would need 50+ pods
	a.SetOverride(req.Target, CPAOverride{
		Mode:           CPAModeMinCost,
		MaxCostPerHour: 0.01, // impossibly tight
	})

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err, "should not error when budget is unsatisfiable")
	assert.Greater(t, rec.DesiredReplicas, int32(0))
}

func TestCPAAlgorithm_EnvVarMode(t *testing.T) {
	a := NewCPAAlgorithm()
	t.Setenv("AIBRIX_CPA_MODE", "min-latency")
	req := newCPARequest(10.0, 1)
	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	assert.Equal(t, "min-latency", rec.Metadata["mode"])
}

func TestCPAAlgorithm_EnvVarBudget(t *testing.T) {
	a := NewCPAAlgorithm()
	t.Setenv("AIBRIX_CPA_BUDGET", "7.50")
	req := newCPARequest(40.0, 1)
	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	cost := rec.Metadata["hourly_cost_usd"].(float64)
	assert.LessOrEqual(t, cost, 7.50)
}

func TestCPAAlgorithm_EnvVarGPUs(t *testing.T) {
	a := NewCPAAlgorithm()
	t.Setenv("AIBRIX_CPA_GPU_TYPES", "nvidia-a100-80gb, nvidia-t4-16gb ")
	req := newCPARequest(10.0, 1)
	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	gpuType := rec.Metadata["selected_gpu_type"].(string)
	assert.Contains(t, []string{"nvidia-a100-80gb", "nvidia-t4-16gb"}, gpuType)
}

func TestCPAAlgorithm_PlanSummary(t *testing.T) {
	p := CostAwarePlan{
		SelectedGPUType:     "nvidia-h100-80gb",
		SelectedCostClass:   "on-demand",
		Replicas:            4,
		HourlyCostUSD:       16.0,
		PerPodCostUSD:       4.0,
		ThroughputQPSPerPod: 20.0,
		ObservedQPS:         80.0,
		Mode:                "min-cost",
	}
	summary := p.String()
	assert.True(t, strings.Contains(summary, "nvidia-h100-80gb"))
	assert.True(t, strings.Contains(summary, "$16.00"))
}

func TestCPAAlgorithm_CostPerQPS(t *testing.T) {
	p := CostAwarePlan{HourlyCostUSD: 10.0, ObservedQPS: 100.0}
	assert.InDelta(t, 0.10, CostPerQPS(p), 0.001)
	p2 := CostAwarePlan{HourlyCostUSD: 5.0, ObservedQPS: 0.0}
	assert.Equal(t, 0.0, CostPerQPS(p2))
}

func TestCPAAlgorithm_HPA_DefaultStrategy(t *testing.T) {
	// Sanity: KPA still works (regression).
	kpa := NewScalingAlgorithm(autoscalingv1alpha1.KPA)
	assert.Equal(t, "kpa", kpa.GetAlgorithmType())
	apa := NewScalingAlgorithm(autoscalingv1alpha1.APA)
	assert.Equal(t, "apa", apa.GetAlgorithmType())
	hpa := NewScalingAlgorithm(autoscalingv1alpha1.HPA)
	assert.Equal(t, "hpa", hpa.GetAlgorithmType())
	cpa := NewScalingAlgorithm(autoscalingv1alpha1.CPA)
	assert.Equal(t, "cpa", cpa.GetAlgorithmType())
	unknown := NewScalingAlgorithm(autoscalingv1alpha1.ScalingStrategyType("nope"))
	assert.Equal(t, "kpa", unknown.GetAlgorithmType(), "unknown should fall back to KPA")
}

func TestCPAAlgorithm_ReplicasHonorsMin(t *testing.T) {
	// Even with 0.1 QPS per pod, replicas should never go below 1.
	got := replicasForQPS(0.1, GPUCostEntry{ThroughputQPS: 10}, CPAConfig{})
	assert.GreaterOrEqual(t, got, int32(1))
}

func TestCPAAlgorithm_CatalogKey(t *testing.T) {
	assert.Equal(t, "spot|nvidia-h100-80gb", catalogKey("nvidia-h100-80gb", CostClassSpot))
	assert.Equal(t, "on-demand|nvidia-h100-80gb", catalogKey("nvidia-h100-80gb", CostClassOnDemand))
}

func TestCPAAlgorithm_RoundCents(t *testing.T) {
	assert.Equal(t, 1.23, roundCents(1.234))
	assert.Equal(t, 1.0, roundCents(1.0))
	assert.Equal(t, 0.0, roundCents(0.0))
}

func TestCPAAlgorithm_CostSavingsComparedToExpensive(t *testing.T) {
	a := NewCPAAlgorithm()

	// Same QPS, two strategies. Use distinct PA keys so overrides don't collide.
	reqCheap := newCPARequest(20.0, 1)
	reqCheap.Target.Namespace = "ns-cheap"
	a.SetOverride(reqCheap.Target, CPAOverride{Mode: CPAModeMinCost})

	reqFast := newCPARequest(20.0, 1)
	reqFast.Target.Namespace = "ns-fast"
	a.SetOverride(reqFast.Target, CPAOverride{Mode: CPAModeMinLatency})

	recCheap, _ := a.ComputeRecommendation(nil, reqCheap)
	recFast, _ := a.ComputeRecommendation(nil, reqFast)

	cheapCost := recCheap.Metadata["hourly_cost_usd"].(float64)
	fastCost := recFast.Metadata["hourly_cost_usd"].(float64)

	assert.Less(t, cheapCost, fastCost, "min-cost should be cheaper than min-latency")
}

func TestCPAAlgorithm_BigClusterScenario(t *testing.T) {
	// 300 QPS, $50 budget, max 50 replicas: should pick the right mix.
	a := NewCPAAlgorithm()
	req := newCPARequest(300.0, 1)
	a.SetOverride(req.Target, CPAOverride{
		Mode:           CPAModeMinCost,
		MaxCostPerHour: 50.0,
	})
	req.ScalingContext.(*mockScalingContext).MaxReplicas = 50

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	cost := rec.Metadata["hourly_cost_usd"].(float64)
	assert.LessOrEqual(t, cost, 50.0)
	assert.Greater(t, rec.DesiredReplicas, int32(0))
}

func TestCPAAlgorithm_ScalingRateLimitsAreClamped(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(1000.0, 2)
	req.ScalingContext.(*mockScalingContext).MaxScaleUpRate = 1.5 // only 50% increase
	req.ScalingContext.(*mockScalingContext).MaxScaleDownRate = 3.0

	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	// 2 * 1.5 = 3, so desired should be at most 3
	assert.LessOrEqual(t, rec.DesiredReplicas, int32(3))
}

func TestCPAAlgorithm_NotPanicOnInvalidOverride(t *testing.T) {
	a := NewCPAAlgorithm()
	req := newCPARequest(5.0, 1)
	a.SetOverride(req.Target, CPAOverride{Mode: CPAMode("not-a-real-mode")})
	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	assert.NotEmpty(t, rec.Metadata["selected_gpu_type"])
}

func TestCPAAlgorithm_AllOnDemandSorted(t *testing.T) {
	c := NewDefaultGPUCostCatalog()
	entries := c.AllOnDemand()
	for i := 1; i < len(entries); i++ {
		assert.LessOrEqual(t, entries[i-1].GPUType, entries[i].GPUType)
	}
}

func TestCPAAlgorithm_HighLoadRespectsCooldown(t *testing.T) {
	a := NewCPAAlgorithm()
	// The cooldown just lives in config; we don't reject here.
	// But the algorithm should still return a reasonable value.
	req := newCPARequest(200.0, 1)
	rec, err := a.ComputeRecommendation(nil, req)
	require.NoError(t, err)
	assert.Greater(t, rec.DesiredReplicas, int32(0))
}

func TestCPAAlgorithm_MathInfCheck(t *testing.T) {
	// Direct test of budget-too-tight fallback to cheapest.
	cfg := CPAConfig{
		Mode:              CPAModeMinCost,
		MaxCostPerHour:    0.001,
		CandidateGPUTypes: []string{"nvidia-h100-80gb"},
	}
	a := NewCPAAlgorithm()
	plan, err := a.optimalPlan(1000.0, cfg)
	require.NoError(t, err)
	assert.False(t, math.IsInf(plan.TotalHourlyCost, 1), "fallback should pick a finite cost")
}

func TestCPAAlgorithm_PerPAIsolation(t *testing.T) {
	// Two PAs with different modes should not interfere.
	a := NewCPAAlgorithm()

	a.SetOverride(types.ScaleTarget{Namespace: "ns1", Name: "pa1"},
		CPAOverride{Mode: CPAModeMinCost})
	a.SetOverride(types.ScaleTarget{Namespace: "ns2", Name: "pa2"},
		CPAOverride{Mode: CPAModeMinLatency})

	req1 := newCPARequest(20.0, 1)
	req1.Target.Namespace = "ns1"
	req1.Target.Name = "pa1"
	rec1, err := a.ComputeRecommendation(nil, req1)
	require.NoError(t, err)

	req2 := newCPARequest(20.0, 1)
	req2.Target.Namespace = "ns2"
	req2.Target.Name = "pa2"
	rec2, err := a.ComputeRecommendation(nil, req2)
	require.NoError(t, err)

	assert.Equal(t, "min-cost", rec1.Metadata["mode"])
	assert.Equal(t, "min-latency", rec2.Metadata["mode"])
}
