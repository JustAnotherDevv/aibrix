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

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/types"
	v1 "k8s.io/api/core/v1"
	klog "k8s.io/klog/v2"
)

const RouterGpuAware types.RoutingAlgorithm = "gpu-aware"

// GPU type memory capacities in GiB (common datacenter GPUs)
var gpuMemoryCapacity = map[string]float64{
	"nvidia-a100-80gb":  80.0,
	"nvidia-a100-40gb":  40.0,
	"nvidia-h100-80gb":  80.0,
	"nvidia-h200-141gb": 141.0,
	"nvidia-l4-24gb":    24.0,
	"nvidia-l40s-48gb":  48.0,
	"nvidia-a10g-24gb":  24.0,
	"nvidia-t4-16gb":    16.0,
	"nvidia-v100-16gb":  16.0,
	"nvidia-p100-16gb":  16.0,
}

// GPU FLOPS ratings relative to A100-80GB (for compute scoring)
var gpuComputePower = map[string]float64{
	"nvidia-a100-80gb":  1.0,
	"nvidia-a100-40gb":  0.5,
	"nvidia-h100-80gb":  2.0,
	"nvidia-h200-141gb": 2.2,
	"nvidia-l4-24gb":    0.12,
	"nvidia-l40s-48gb":  0.35,
	"nvidia-a10g-24gb":  0.12,
	"nvidia-t4-16gb":    0.08,
	"nvidia-v100-16gb":  0.15,
	"nvidia-p100-16gb":  0.1,
}

func init() {
	Register(RouterGpuAware, NewGpuAwareRouter)
}

type gpuAwareRouter struct {
	cache cache.Cache
}

func NewGpuAwareRouter() (types.Router, error) {
	c, err := cache.Get()
	if err != nil {
		return nil, err
	}

	return gpuAwareRouter{
		cache: c,
	}, nil
}

// getGpuComputePower returns the relative compute power for a GPU type
func (r gpuAwareRouter) getGpuComputePower(gpuType string) float64 {
	if power, ok := gpuComputePower[gpuType]; ok {
		return power
	}
	return 0.5 // Default for unknown
}

// ScoreAll scores pods based on GPU memory headroom relative to GPU type capacity.
// Pods with more available GPU memory relative to their GPU's total capacity get higher scores.
// This enables intelligent scheduling across heterogeneous GPU fleets.
func (r gpuAwareRouter) ScoreAll(ctx *types.RoutingContext, readyPodList types.PodList) ([]float64, []bool, error) {
	pods := readyPodList.All()
	scores := make([]float64, len(pods))
	scored := make([]bool, len(pods))

	for i, pod := range pods {
		gpuType := GetGpuTypeFromPod(pod)
		gpuCapacity := GetGpuCapacityFromPod(pod)
		gpuCount := gpuCount(pod)
		totalCapacity := gpuCapacity * float64(gpuCount)

		// Get current GPU memory utilization
		headroom, err := podHeadroom(pod, ctx, r.cache)
		if err != nil {
			klog.V(4).ErrorS(err, "failed to get GPU metrics for pod", "pod", pod.Name)
			scores[i] = 0
			scored[i] = false
			continue
		}
		currentUtilization := 1.0 - headroom

		// Calculate memory headroom: how much GPU memory is available
		// Score = (1 - utilization) * capacity_weight * compute_weight
		capacityWeight := totalCapacity / 80.0 // Normalize to A100-80GB
		computeWeight := r.getGpuComputePower(gpuType)

		// Composite score: headroom * capacity * compute power
		// Higher score = better candidate (more resources available)
		scores[i] = headroom * capacityWeight * computeWeight
		scored[i] = true

		klog.V(4).Infof("pod: %s, gpu: %s, capacity: %.0fGiB, count: %d, util: %.2f, score: %.4f",
			pod.Name, gpuType, totalCapacity, gpuCount, currentUtilization, scores[i])
	}

	return scores, scored, nil
}

// Polarity returns higher-is-better (more headroom = higher score = better)
func (r gpuAwareRouter) Polarity() types.Polarity {
	return types.PolarityMost
}

// Route selects the pod with the best GPU headroom for the request
func (r gpuAwareRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	var targetPod *v1.Pod
	maxScore := -math.MaxFloat64
	var candidatePods []*v1.Pod

	for _, pod := range readyPodList.All() {
		gpuType := GetGpuTypeFromPod(pod)
		gpuCapacity := GetGpuCapacityFromPod(pod)
		gpuCount := gpuCount(pod)
		totalCapacity := gpuCapacity * float64(gpuCount)

		// Get GPU cache utilization
		headroom, err := podHeadroom(pod, ctx, r.cache)
		if err != nil {
			klog.V(4).ErrorS(err, "failed to get GPU metrics for pod", "pod", pod.Name)
			continue
		}
		currentUtilization := 1.0 - headroom
		capacityWeight := totalCapacity / 80.0
		computeWeight := r.getGpuComputePower(gpuType)
		score := headroom * capacityWeight * computeWeight

		klog.V(4).Infof("pod: %s, gpu: %s, capacity: %.0fGiB, count: %d, util: %.2f, score: %.4f",
			pod.Name, gpuType, totalCapacity, gpuCount, currentUtilization, score)

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

	// Fallback to random if no valid metrics
	if targetPod == nil {
		var err error
		targetPod, err = SelectRandomPodAsFallback(ctx, readyPodList.All(), rand.Intn)
		if err != nil {
			return "", err
		}
		klog.V(4).Infof("fallback select targetPod: %s(%s)", targetPod.Name, targetPod.Status.PodIP)
	} else {
		klog.V(4).Infof("gpu-aware select targetPod: %s(%s) score: %.4f", targetPod.Name, targetPod.Status.PodIP, maxScore)
	}

	if targetPod == nil {
		return "", fmt.Errorf("no pods to forward request")
	}

	klog.V(4).Infof("targetPod: %s(%s)", targetPod.Name, targetPod.Status.PodIP)
	ctx.SetTargetPod(targetPod)
	return ctx.TargetAddress(), nil
}

// GetGpuTypeFromPod is exported for use in tests
func GetGpuTypeFromPod(pod *v1.Pod) string {
	if gpuType, ok := pod.Labels["gpu-type"]; ok {
		return gpuType
	}
	if gpuType, ok := pod.Labels["nvidia.com/gpu.product"]; ok {
		return gpuType
	}
	if gpuType, ok := pod.Annotations["gpu-type"]; ok {
		return gpuType
	}
	return "unknown"
}

// GetGpuCapacityFromPod is exported for use in tests
func GetGpuCapacityFromPod(pod *v1.Pod) float64 {
	gpuType := GetGpuTypeFromPod(pod)
	if cap, ok := gpuMemoryCapacity[gpuType]; ok {
		return cap
	}
	return 40.0
}

// FormatGpuInfo returns a human-readable GPU info string for a pod
func FormatGpuInfo(pod *v1.Pod) string {
	gpuType := GetGpuTypeFromPod(pod)
	capacity := GetGpuCapacityFromPod(pod)
	gpuCount := 1
	for _, container := range pod.Spec.Containers {
		if gpuQty, ok := container.Resources.Limits["nvidia.com/gpu"]; ok {
			gpuCount = int(gpuQty.Value())
			break
		}
	}
	return fmt.Sprintf("%s x%d (%.0fGiB)", gpuType, gpuCount, capacity*float64(gpuCount))
}
