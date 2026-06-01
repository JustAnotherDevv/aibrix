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
	"github.com/vllm-project/aibrix/pkg/cache"
	metrics "github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/types"
	v1 "k8s.io/api/core/v1"
)

// gpuCount returns the number of nvidia.com/gpu requested by the first
// container in the pod that sets the limit, defaulting to 1.
func gpuCount(pod *v1.Pod) int {
	for _, c := range pod.Spec.Containers {
		if q, ok := c.Resources.Limits["nvidia.com/gpu"]; ok {
			return int(q.Value())
		}
	}
	return 1
}

// userID extracts a stable per-tenant identifier from the routing context.
// Returns "anonymous" when no user is attached.
func userID(ctx *types.RoutingContext) string {
	if ctx.User != nil && *ctx.User != "" {
		return *ctx.User
	}
	return "anonymous"
}

// podHeadroom returns (1 - GPU cache utilization) for a pod, or an error
// if the metric is unavailable. The three GPU routers all score on headroom
// weighted by tenant priority / strategy, so this is the shared primitive.
func podHeadroom(pod *v1.Pod, ctx *types.RoutingContext, c cache.Cache) (float64, error) {
	v, err := c.GetMetricValueByPodModel(pod.Name, pod.Namespace, ctx.Model, metrics.GPUCacheUsagePerc)
	if err != nil {
		return 0, err
	}
	return 1.0 - v.GetSimpleValue(), nil
}
