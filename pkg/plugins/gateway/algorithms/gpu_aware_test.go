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

func TestGetGpuTypeFromPod(t *testing.T) {
	tests := []struct {
		name     string
		pod      *v1.Pod
		expected string
	}{
		{
			name: "pod with gpu-type label",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
					Labels: map[string]string{
						"gpu-type": "nvidia-h100-80gb",
					},
				},
			},
			expected: "nvidia-h100-80gb",
		},
		{
			name: "pod with nvidia.com/gpu.product label",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
					Labels: map[string]string{
						"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3",
					},
				},
			},
			expected: "NVIDIA-H100-80GB-HBM3",
		},
		{
			name: "pod with gpu-type annotation",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
					Annotations: map[string]string{
						"gpu-type": "nvidia-a100-40gb",
					},
				},
			},
			expected: "nvidia-a100-40gb",
		},
		{
			name: "pod without GPU info",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
			},
			expected: "unknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetGpuTypeFromPod(tt.pod)
			if result != tt.expected {
				t.Errorf("GetGpuTypeFromPod() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGetGpuCapacityFromPod(t *testing.T) {
	tests := []struct {
		name     string
		pod      *v1.Pod
		expected float64
	}{
		{
			name: "H100 pod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"gpu-type": "nvidia-h100-80gb"},
				},
			},
			expected: 80.0,
		},
		{
			name: "T4 pod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"gpu-type": "nvidia-t4-16gb"},
				},
			},
			expected: 16.0,
		},
		{
			name: "unknown GPU",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
			},
			expected: 24.0, // default for unknown GPU
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetGpuCapacityFromPod(tt.pod)
			if result != tt.expected {
				t.Errorf("GetGpuCapacityFromPod() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestFormatGpuInfo(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-pod",
			Labels: map[string]string{
				"gpu-type": "nvidia-h100-80gb",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "worker",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							"nvidia.com/gpu": resource.MustParse("4"),
						},
					},
				},
			},
		},
	}

	result := FormatGpuInfo(pod)
	expected := "nvidia-h100-80gb x4 (320GiB)"
	if result != expected {
		t.Errorf("FormatGpuInfo() = %v, want %v", result, expected)
	}
}

func TestGpuMemoryCapacity(t *testing.T) {
	// Verify all known GPU types have capacities defined
	knownGPUs := []string{
		"nvidia-a100-80gb",
		"nvidia-h100-80gb",
		"nvidia-h200-141gb",
		"nvidia-l4-24gb",
		"nvidia-t4-16gb",
	}

	for _, gpu := range knownGPUs {
		spec := lookupGPU(gpu)
		if spec.MemoryGB <= 0 {
			t.Errorf("GPU type %s has no memory entry (got %v)", gpu, spec)
		}
	}
}

func TestGpuComputePower(t *testing.T) {
	// Verify H100 has higher compute power than A100
	if lookupGPU("nvidia-h100-80gb").Compute <= lookupGPU("nvidia-a100-80gb").Compute {
		t.Error("H100 should have higher compute power than A100")
	}

	// Verify A100 has higher compute power than T4
	if lookupGPU("nvidia-a100-80gb").Compute <= lookupGPU("nvidia-t4-16gb").Compute {
		t.Error("A100 should have higher compute power than T4")
	}
}
