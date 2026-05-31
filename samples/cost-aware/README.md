# Cost-Aware LLM Serving — AIBrix Samples

Sample Kubernetes manifests and ConfigMaps for deploying AIBrix with my
cost-aware contributions: the **cost-aware router** (`gpu-cost-aware`),
the **multi-tenant quota router** (`tenant-quota`), and the **cost-aware
autoscaler** (CPA, `scalingStrategy: CPA`).

## Files

| File | What it shows |
|------|---------------|
| `01-podautoscaler-cpa.yaml` | Minimal PodAutoscaler using the CPA strategy with budget cap + GPU allowlist |
| `02-cost-catalog-configmap.yaml` | ConfigMap with the cost+throughput catalog for production overrides |
| `03-gateway-plugin-config.yaml` | Gateway plugin config showing cost-aware router + multi-tenant chain |
| `04-tenant-configmap.yaml` | Multi-tenant quota tiers (free / standard / premium / enterprise) with per-tenant GPU allowlists |
| `05-full-deployment.yaml` | End-to-end: StormService + CPA PodAutoscaler + Service, ready to `kubectl apply` |

## Quick Start

```bash
# 1. Install AIBrix (gateway plugin + controller manager)
helm install aibrix dist/chart/ --namespace aibrix-system --create-namespace

# 2. Apply the cost catalog (production pricing override)
kubectl apply -f 02-cost-catalog-configmap.yaml

# 3. Apply the gateway plugin config (enables cost-aware router)
kubectl apply -f 03-gateway-plugin-config.yaml

# 4. Apply the tenant config (multi-tenant quotas)
kubectl apply -f 04-tenant-configmap.yaml

# 5. Deploy the workload with CPA autoscaling
kubectl apply -f 05-full-deployment.yaml

# 6. Watch CPA pick the cheapest GPU type
kubectl get podautoscaler llm-7b-cost-optimized-pa -n aibrix-system -w
# Look for: REPLICAS column, status.actualScale changes
```

## What CPA Does

CPA picks the cheapest (replica count × GPU type) combination that meets
your QPS target. Examples from the sample config:

| QPS | Cheapest feasible | Cost/hr | Why |
|-----|-------------------|---------|-----|
| 5   | 1× T4             | $0.50   | T4 (2.5 QPS) is cheapest, 1 pod handles 5 QPS |
| 50  | 7× L4             | $5.60   | 50/4.5=12 L4=$9.60 vs 7 L4 too many. 50/8=7 L40S=$8.40 — actually L40S wins |
| 200 | 25× L40S          | $30.00  | Over budget, CPA falls back to cheaper: 25 L40S=$30, or 200/14=15 A100=$37.50 |

CPA logs its decision in the PodAutoscaler status:

```bash
kubectl describe pa llm-7b-cost-optimized-pa -n aibrix-system
# Look at Status.ScalingHistory[].Reason for "cpa: 7x nvidia-l40s-48gb @ $8.40/hr"
```

## Modes Reference

Set via `spec.costOptimization.mode` on the PodAutoscaler, or cluster-wide
via the `AIBRIX_CPA_MODE` env var on the controller-manager.

| Mode | Behavior |
|------|----------|
| `min-cost` | Pick cheapest config that meets QPS (default) |
| `balanced` | Same as min-cost with default weights (40/40/20 cost/perf/power) |
| `min-latency` | Pick fastest GPU first, then cheapest within that class |
| `spot-first` | Force catalog lookup to use spot pricing |
| `reserved-first` | Force catalog lookup to use reserved pricing |

## Cost Catalog Reference

| GPU | On-Demand | Spot (×0.30) | Reserved (×0.60) | QPS/Pod | Memory |
|-----|-----------|--------------|-------------------|---------|--------|
| nvidia-h200-141gb | $5.00 | $1.50 | $3.00 | 22.0 | 141 GB |
| nvidia-h100-80gb | $4.00 | $1.20 | $2.40 | 20.0 | 80 GB |
| nvidia-a100-80gb | $2.50 | $0.75 | $1.50 | 14.0 | 80 GB |
| nvidia-l40s-48gb | $1.20 | $0.36 | $0.72 | 8.0 | 48 GB |
| nvidia-l4-24gb | $0.80 | $0.24 | $0.48 | 4.5 | 24 GB |
| nvidia-t4-16gb | $0.50 | $0.15 | $0.30 | 2.5 | 16 GB |

Throughput numbers are calibrated for a 7-8B chat model at batch=8.
Override per-environment via the ConfigMap (file 02).

## Verification

```bash
# CPA is making decisions
kubectl logs -n aibrix-system -l control-plane=controller-manager --tail=100 | grep cpa

# Prometheus metrics
kubectl port-forward -n aibrix-system svc/prometheus 9090:9090
# Then query: aibrix_podautoscaler_desired_replicas{strategy="cpa"}
```

## Interview Talking Points

1. **Why CPA matters:** "Existing autoscalers (HPA/KPA/APA) decide when to
   scale. CPA also decides what to scale to — switching to a cheaper GPU
   when QPS drops, or scaling up cheaper GPU types first."

2. **Real production tradeoff:** "The catalog is static for now. In a real
   deployment, the gpu-optimizer Python service would update it based on
   profiled throughput. We started static because the algorithm is the hard
   part — plugging in live data is a one-line catalog update."

3. **Graceful degradation:** "If a tenant's budget is too tight for any
   candidate GPU, CPA falls back to the absolute cheapest rather than
   refusing to scale. Better to over-spend than to melt the cluster."

4. **Why the cost catalog is a ConfigMap, not hardcoded:** "Pricing changes
   constantly. Enterprise customers negotiate reserved capacity. New GPU
   SKUs launch. ConfigMap lets ops tune pricing without a redeploy."
