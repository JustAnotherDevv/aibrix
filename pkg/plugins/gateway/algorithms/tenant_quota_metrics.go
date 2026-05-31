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
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Prometheus metrics for the tenant-quota router.
// All metrics are exported on the gateway's /metrics endpoint.

// tenantRoutingDecisions counts routing decisions per tenant/tier
var tenantRoutingDecisions = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "aibrix_tenant_routing_decisions_total",
		Help: "Number of routing decisions made per tenant and tier.",
	},
	[]string{"tenant_id", "tier"},
)

// tenantQuotaRejections counts quota rejections with reason
var tenantQuotaRejections = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "aibrix_tenant_quota_rejections_total",
		Help: "Number of requests rejected due to quota violations, by reason.",
	},
	[]string{"tenant_id", "tier", "reason"},
)

// tenantRequestLatency tracks request latency per tenant/tier
var tenantRequestLatency = promauto.NewHistogramVec(
	prometheus.HistogramOpts{
		Name:    "aibrix_tenant_request_latency_seconds",
		Help:    "Request latency per tenant/tier.",
		Buckets: prometheus.ExponentialBuckets(0.01, 2, 12), // 10ms to ~40s
	},
	[]string{"tenant_id", "tier"},
)

// tenantTokenUsage tracks token consumption
var tenantTokenUsage = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "aibrix_tenant_token_usage_total",
		Help: "Token usage by tenant, tier, and direction (input/output).",
	},
	[]string{"tenant_id", "tier", "direction"},
)

// tenantConcurrentGauge tracks current concurrent requests per tenant
var tenantConcurrentGauge = promauto.NewGaugeVec(
	prometheus.GaugeOpts{
		Name: "aibrix_tenant_concurrent_requests",
		Help: "Current number of in-flight requests per tenant.",
	},
	[]string{"tenant_id", "tier"},
)

// tenantHourCost tracks cost accrued per tenant in the current hour
var tenantHourCost = promauto.NewGaugeVec(
	prometheus.GaugeOpts{
		Name: "aibrix_tenant_hour_cost_usd",
		Help: "Cost accrued per tenant in the current hour (USD).",
	},
	[]string{"tenant_id", "tier"},
)

// fairShareShareRatio tracks what share of capacity each tenant got
var fairShareShareRatio = promauto.NewGaugeVec(
	prometheus.GaugeOpts{
		Name: "aibrix_tenant_fair_share_ratio",
		Help: "Share of total capacity allocated to each tenant under fair-share scheduling.",
	},
	[]string{"tenant_id", "tier"},
)
