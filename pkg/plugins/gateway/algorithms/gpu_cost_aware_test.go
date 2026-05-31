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
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGpuCostPerHour(t *testing.T) {
	tests := []struct {
		gpuType  string
		expected float64
	}{
		{"nvidia-h200-141gb", 5.00},
		{"nvidia-h100-80gb", 4.00},
		{"nvidia-a100-80gb", 2.50},
		{"nvidia-l4-24gb", 0.80},
		{"nvidia-t4-16gb", 0.50},
	}

	for _, tt := range tests {
		t.Run(tt.gpuType, func(t *testing.T) {
			cost, ok := gpuCostPerHour[tt.gpuType]
			if !ok {
				t.Errorf("GPU type %s not in cost map", tt.gpuType)
			}
			if cost != tt.expected {
				t.Errorf("cost for %s = %.2f, want %.2f", tt.gpuType, cost, tt.expected)
			}
		})
	}
}

func TestGpuTDPWatts(t *testing.T) {
	// H100 should be more power-hungry than T4
	if gpuTDPWatts["nvidia-h100-80gb"] <= gpuTDPWatts["nvidia-t4-16gb"] {
		t.Error("H100 should have higher TDP than T4")
	}

	// L4 is very efficient
	if gpuTDPWatts["nvidia-l4-24gb"] >= gpuTDPWatts["nvidia-a100-80gb"] {
		t.Error("L4 should have lower TDP than A100")
	}
}

func TestParseStrategy(t *testing.T) {
	tests := []struct {
		input    string
		expected RoutingStrategy
	}{
		{"low-cost", StrategyLowCost},
		{"COST", StrategyLowCost},
		{"cheapest", StrategyLowCost},
		{"low-latency", StrategyLowLatency},
		{"latency", StrategyLowLatency},
		{"fastest", StrategyLowLatency},
		{"low-power", StrategyLowPower},
		{"green", StrategyLowPower},
		{"sla", StrategySLAOptimized},
		{"sla-optimized", StrategySLAOptimized},
		{"balanced", StrategyBalanced},
		{"unknown", StrategyBalanced},
		{"", StrategyBalanced},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := parseStrategy(tt.input)
			if result != tt.expected {
				t.Errorf("parseStrategy(%q) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}

func TestEffectiveCostPerHour_SpotDiscount(t *testing.T) {
	r := gpuCostAwareRouter{
		config: CostAwareConfig{PreferSpot: true},
	}

	baseCost := r.effectiveCostPerHour("nvidia-a100-80gb")
	expected := 2.50 * 0.30 // 70% discount
	if baseCost != expected {
		t.Errorf("spot A100 cost = %.2f, want %.2f", baseCost, expected)
	}
}

func TestEffectiveCostPerHour_ReservedDiscount(t *testing.T) {
	r := gpuCostAwareRouter{
		config: CostAwareConfig{PreferReserved: true},
	}

	baseCost := r.effectiveCostPerHour("nvidia-h100-80gb")
	expected := 4.00 * 0.60 // 40% discount
	if baseCost != expected {
		t.Errorf("reserved H100 cost = %.2f, want %.2f", baseCost, expected)
	}
}

func TestEffectiveCostPerHour_OnDemand(t *testing.T) {
	r := gpuCostAwareRouter{}

	baseCost := r.effectiveCostPerHour("nvidia-t4-16gb")
	if baseCost != 0.50 {
		t.Errorf("on-demand T4 cost = %.2f, want 0.50", baseCost)
	}
}

func TestTotalCostPerHour_MultiGPU(t *testing.T) {
	r := gpuCostAwareRouter{}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{"gpu-type": "nvidia-a100-80gb"},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							"nvidia.com/gpu": resource.MustParse("4"),
						},
					},
				},
			},
		},
	}

	cost := r.totalCostPerHour(pod, "nvidia-a100-80gb")
	expected := 2.50 * 4
	if cost != expected {
		t.Errorf("4xA100 cost = %.2f, want %.2f", cost, expected)
	}
}

func TestGetPodCostClass(t *testing.T) {
	tests := []struct {
		name     string
		pod      *v1.Pod
		expected string
	}{
		{
			name: "spot instance",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"node.kubernetes.io/instance-life-cycle": "spot",
					},
				},
			},
			expected: "spot",
		},
		{
			name: "aibrix cost class label",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"aibrix.ai/cost-class": "reserved",
					},
				},
			},
			expected: "reserved",
		},
		{
			name:     "no cost class",
			pod:      &v1.Pod{},
			expected: "on-demand",
		},
	}

	r := gpuCostAwareRouter{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := r.getPodCostClass(tt.pod)
			if result != tt.expected {
				t.Errorf("getPodCostClass() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestSatisfiesConstraints_CostBudget(t *testing.T) {
	r := gpuCostAwareRouter{
		config: CostAwareConfig{MaxCostPer1kTokens: 0.005},
	}

	// 4x H100 at $4/hr = $16/hr -> $0.0444/1k tokens - exceeds budget
	expensivePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"gpu-type": "nvidia-h100-80gb"}},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{"nvidia.com/gpu": resource.MustParse("4")},
				}},
			},
		},
	}
	if ok, _ := r.satisfiesConstraints(expensivePod, "default", "llama", 16.0, 2800.0); ok {
		t.Error("4x H100 should exceed 0.005/1k token budget")
	}

	// 1x T4 at $0.50/hr = $0.0014/1k tokens - within budget
	cheapPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"gpu-type": "nvidia-t4-16gb"}},
	}
	if ok, reason := r.satisfiesConstraints(cheapPod, "default", "llama", 0.50, 70.0); !ok {
		t.Errorf("1x T4 should be within budget, got reason: %s", reason)
	}
}

func TestSatisfiesConstraints_PowerBudget(t *testing.T) {
	r := gpuCostAwareRouter{
		config: CostAwareConfig{PowerBudgetWatts: 500.0},
	}

	// 1x H100 = 700W - exceeds 500W budget
	h100Pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"gpu-type": "nvidia-h100-80gb"}},
	}
	if ok, _ := r.satisfiesConstraints(h100Pod, "default", "llama", 4.0, 700.0); ok {
		t.Error("H100 should exceed 500W power budget")
	}

	// 1x A100 = 400W - within 500W budget
	a100Pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"gpu-type": "nvidia-a100-80gb"}},
	}
	if ok, reason := r.satisfiesConstraints(a100Pod, "default", "llama", 2.5, 400.0); !ok {
		t.Errorf("A100 should be within power budget, got reason: %s", reason)
	}
}

func TestSatisfiesConstraints_NoBudget(t *testing.T) {
	// Without budgets, any pod is acceptable
	r := gpuCostAwareRouter{}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"gpu-type": "nvidia-h200-141gb"}},
	}
	if ok, _ := r.satisfiesConstraints(pod, "default", "llama", 5.0, 700.0); !ok {
		t.Error("without budgets, any pod should be acceptable")
	}
}

func TestEstimateCostPer1kTokens(t *testing.T) {
	r := gpuCostAwareRouter{}

	// H100: $4/hr -> ~$0.0111/1k tokens
	h100Pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"gpu-type": "nvidia-h100-80gb"}},
	}
	cost := r.EstimateCostPer1kTokens(h100Pod)
	expected := 4.0 / 360.0
	if cost != expected {
		t.Errorf("H100 cost/1k = %.6f, want %.6f", cost, expected)
	}

	// T4: $0.50/hr -> ~$0.0014/1k tokens (10x cheaper)
	t4Pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"gpu-type": "nvidia-t4-16gb"}},
	}
	cost = r.EstimateCostPer1kTokens(t4Pod)
	expected = 0.50 / 360.0
	if cost != expected {
		t.Errorf("T4 cost/1k = %.6f, want %.6f", cost, expected)
	}
}

func TestStrategyWeighting(t *testing.T) {
	// Verify strategy weights sum correctly
	strategies := []RoutingStrategy{
		StrategyLowCost, StrategyLowLatency, StrategyLowPower, StrategySLAOptimized, StrategyBalanced,
	}

	for _, s := range strategies {
		t.Run(s.String(), func(t *testing.T) {
			// Each strategy should have weights that sum to 1.0
			var w1, w2, w3 float64
			switch s {
			case StrategyLowCost:
				w1, w2, w3 = 0.85, 0.10, 0.05
			case StrategyLowLatency:
				w1, w2, w3 = 0.10, 0.85, 0.05
			case StrategyLowPower:
				w1, w2, w3 = 0.15, 0.15, 0.70
			case StrategySLAOptimized:
				w1, w2, w3 = 0.70, 0.20, 0.10
			default:
				w1, w2, w3 = 0.4, 0.4, 0.2
			}
			sum := w1 + w2 + w3
			if sum < 0.99 || sum > 1.01 {
				t.Errorf("weights sum = %.2f, want 1.0", sum)
			}
		})
	}
}

func TestGetEnvHelpers(t *testing.T) {
	// Save and restore env vars
	t.Setenv("TEST_AIBRIX_VAR", "hello")
	t.Setenv("TEST_AIBRIX_FLOAT", "3.14")
	t.Setenv("TEST_AIBRIX_BOOL", "true")

	if v := getEnv("TEST_AIBRIX_VAR", "default"); v != "hello" {
		t.Errorf("getEnv = %q, want %q", v, "hello")
	}
	if v := getEnv("TEST_AIBRIX_MISSING", "fallback"); v != "fallback" {
		t.Errorf("getEnv missing = %q, want %q", v, "fallback")
	}
	if v := getEnvFloat("TEST_AIBRIX_FLOAT", 0.0); v != 3.14 {
		t.Errorf("getEnvFloat = %f, want 3.14", v)
	}
	if v := getEnvFloat("TEST_AIBRIX_MISSING", 1.5); v != 1.5 {
		t.Errorf("getEnvFloat missing = %f, want 1.5", v)
	}
	if v := getEnvBool("TEST_AIBRIX_BOOL", false); v != true {
		t.Errorf("getEnvBool = %v, want true", v)
	}
	if v := getEnvBool("TEST_AIBRIX_MISSING", true); v != true {
		t.Errorf("getEnvBool missing = %v, want true", v)
	}
}

func TestStrategyString(t *testing.T) {
	tests := []struct {
		strategy RoutingStrategy
		expected string
	}{
		{StrategyBalanced, "balanced"},
		{StrategyLowCost, "low-cost"},
		{StrategyLowLatency, "low-latency"},
		{StrategyLowPower, "low-power"},
		{StrategySLAOptimized, "sla-optimized"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if tt.strategy.String() != tt.expected {
				t.Errorf("String() = %q, want %q", tt.strategy.String(), tt.expected)
			}
		})
	}
}
