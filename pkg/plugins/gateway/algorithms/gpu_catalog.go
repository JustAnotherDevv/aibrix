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

// gpuSpec is the shared per-GPU-type catalog entry used by all three
// gateway routers (gpu-aware, gpu-cost-aware, tenant-quota). The same
// ten common datacenter SKUs are needed by each router, so we keep one
// authoritative map and look it up via lookupGPU.
type gpuSpec struct {
	MemoryGB  float64 // per-GPU memory capacity (GiB)
	Compute   float64 // relative FP16 TFLOPS, A100-80GB = 1.0
	CostPerHr float64 // on-demand USD/hour, us-east-1 reference
	TDPWatts  float64 // typical board power
}

// defaultGPUCatalog is the source of truth for GPU hardware specs and
// pricing. Adding a new SKU only requires one row.
var defaultGPUCatalog = map[string]gpuSpec{
	"nvidia-h200-141gb": {MemoryGB: 141, Compute: 2.2, CostPerHr: 5.00, TDPWatts: 700},
	"nvidia-h100-80gb":  {MemoryGB: 80, Compute: 2.0, CostPerHr: 4.00, TDPWatts: 700},
	"nvidia-a100-80gb":  {MemoryGB: 80, Compute: 1.0, CostPerHr: 2.50, TDPWatts: 400},
	"nvidia-a100-40gb":  {MemoryGB: 40, Compute: 0.5, CostPerHr: 1.80, TDPWatts: 400},
	"nvidia-l40s-48gb":  {MemoryGB: 48, Compute: 0.35, CostPerHr: 1.20, TDPWatts: 350},
	"nvidia-l4-24gb":    {MemoryGB: 24, Compute: 0.12, CostPerHr: 0.80, TDPWatts: 72},
	"nvidia-a10g-24gb":  {MemoryGB: 24, Compute: 0.12, CostPerHr: 0.75, TDPWatts: 150},
	"nvidia-t4-16gb":    {MemoryGB: 16, Compute: 0.08, CostPerHr: 0.50, TDPWatts: 70},
	"nvidia-v100-16gb":  {MemoryGB: 16, Compute: 0.15, CostPerHr: 0.45, TDPWatts: 300},
	"nvidia-p100-16gb":  {MemoryGB: 16, Compute: 0.10, CostPerHr: 0.40, TDPWatts: 250},
}

// lookupGPU returns the spec for gpuType, or a conservative default for
// unknown SKUs (mid-range memory, low compute, mid cost, mid power).
func lookupGPU(gpuType string) gpuSpec {
	if s, ok := defaultGPUCatalog[gpuType]; ok {
		return s
	}
	return gpuSpec{MemoryGB: 24, Compute: 0.1, CostPerHr: 2.0, TDPWatts: 300}
}
