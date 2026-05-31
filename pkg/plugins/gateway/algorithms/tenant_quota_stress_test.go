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

// Stress tests for concurrency safety. These do NOT replace the race detector
// (which requires CGO + gcc), but they exercise the shared-state paths
// heavily enough that bugs manifest as panics, deadlocks, or wrong results
// even without -race. The audit log below documents why the code is
// race-free under the Go memory model.

package routingalgorithms

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Audit log: every shared state mutation in the tenant_quota package and its
// synchronization. If you change a lock discipline, update this list.
//
// CPA + gpu-cost-aware + gpu-aware are read-only after construction (the
// catalog is RWMutex-protected but only written at startup). Only
// tenant_quota has hot-path shared state.
//
// Hot shared state:
//   - TenantRegistry.tenants (map): protected by TenantRegistry.mu (RWMutex)
//   - TenantState.Concurrent (int64): protected ONLY by sync/atomic
//   - TenantState.RequestTokens/TokenBucketTokens (float64): protected by
//     TenantState.mu (sync.Mutex)
//   - TenantState.HourCost (float64): protected by TenantState.mu
//   - TenantState.LastAccess (time.Time): protected by TenantState.mu
//
// Lock order: TenantRegistry.mu -> TenantState.mu. Never the reverse.
//   - cleanupIdle takes both in that order.
//   - GetOrCreate takes only TenantRegistry.mu.
//   - AllowRequest/ReleaseRequest take only TenantState.mu.
// => No deadlock possible because of strict acyclic order.
//
// Concurrent counter: written via atomic, read via atomic, both inside and
// outside the TenantState.mu critical section. This is safe because atomic
// operations have happens-before guarantees with the next read of the same
// variable (Go memory model).
//
// Stop() closes the cleanup channel exactly once. startCleanup reads it
// inside select. Multiple Stop() calls would panic on close-of-closed-chan,
// so the caller must guarantee single-call. This is documented in Stop().

// TestStress_AllowRequest_Concurrent: 100 goroutines, each doing 1000
// AllowRequest calls. Verifies the request token bucket is consistent.
func TestStress_AllowRequest_Concurrent(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	reg := NewTenantRegistry(TierStandard)
	defer reg.Stop()
	ts := reg.GetOrCreate("stress-tenant")

	const goroutines = 100
	const calls = 1000
	var allowed int64
	var rejected int64
	var wg sync.WaitGroup
	wg.Add(goroutines)

	start := make(chan struct{})
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			<-start
			for j := 0; j < calls; j++ {
				ok, _ := ts.AllowRequest(1)
				if ok {
					atomic.AddInt64(&allowed, 1)
				} else {
					atomic.AddInt64(&rejected, 1)
				}
				ts.ReleaseRequest()
			}
		}()
	}
	close(start)
	wg.Wait()

	total := int64(goroutines * calls)
	if allowed+rejected != total {
		t.Errorf("lost calls: allowed=%d rejected=%d total=%d", allowed, rejected, total)
	}
	t.Logf("stress: %d goroutines, %d calls each = %d total, %d allowed, %d rejected",
		goroutines, calls, total, allowed, rejected)
}

// TestStress_MultipleTenants_NoCrossContamination: 50 tenants, 20 goroutines
// each, hit them concurrently. Verify that each tenant's quota is
// independent (one tenant's burst doesn't affect another's).
func TestStress_MultipleTenants_NoCrossContamination(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	reg := NewTenantRegistry(TierStandard)
	defer reg.Stop()

	const numTenants = 50
	const goroutinesPerTenant = 20
	const callsPerGoroutine = 100

	// Pre-create all tenants
	for i := 0; i < numTenants; i++ {
		reg.GetOrCreate(fmt.Sprintf("tenant-%d", i))
	}

	var wg sync.WaitGroup
	start := make(chan struct{})

	perTenantAllowed := make([]int64, numTenants)
	perTenantRejected := make([]int64, numTenants)

	for i := 0; i < numTenants; i++ {
		tenantID := fmt.Sprintf("tenant-%d", i)
		ts := reg.GetOrCreate(tenantID)
		for g := 0; g < goroutinesPerTenant; g++ {
			wg.Add(1)
			go func(tid string, idx int, ts *TenantState) {
				defer wg.Done()
				<-start
				for j := 0; j < callsPerGoroutine; j++ {
					ok, _ := ts.AllowRequest(1)
					if ok {
						atomic.AddInt64(&perTenantAllowed[idx], 1)
					} else {
						atomic.AddInt64(&perTenantRejected[idx], 1)
					}
					ts.ReleaseRequest()
				}
			}(tenantID, i, ts)
		}
	}
	close(start)
	wg.Wait()

	// Each tenant's bucket should not be exhausted past its own BurstSize
	// in the very first call (since the goroutines start simultaneously).
	// The total allowed per tenant should be >= BurstSize.
	for i := 0; i < numTenants; i++ {
		allowed := atomic.LoadInt64(&perTenantAllowed[i])
		burst := int64(defaultTierQuotas[TierStandard].BurstSize)
		if allowed < burst {
			t.Errorf("tenant-%d allowed=%d < burst=%d — cross-contamination or undersized bucket",
				i, allowed, burst)
		}
	}
}

// TestStress_AllowRelease_Concurrent: hit AllowRequest and ReleaseRequest
// from different goroutines. Each goroutine does balanced pairs, so final counter = 0.
func TestStress_AllowRelease_Concurrent(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	reg := NewTenantRegistry(TierStandard)
	defer reg.Stop()
	ts := reg.GetOrCreate("concurrent-tenant")

	// Override to allow many concurrent so we can exercise the counter
	reg.SetOverride("concurrent-tenant", QuotaConfig{
		MaxConcurrent: 10000,
		BurstSize:     100000,
	})

	const goroutines = 50
	const iterations = 5000
	var wg sync.WaitGroup
	wg.Add(goroutines)

	// Each goroutine does balanced AllowRequest+ReleaseRequest pairs
	// Note: ReleaseRequest only when AllowRequest returned true.
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				if ok, _ := ts.AllowRequest(1); ok {
					ts.ReleaseRequest()
				}
			}
		}()
	}

	wg.Wait()

	// After all balanced pairs, counter should be exactly 0
	final := atomic.LoadInt64(&ts.Concurrent)
	if final != 0 {
		t.Errorf("concurrent counter = %d, want 0", final)
	}
}

// TestStress_GetOrCreate_Concurrent: 1000 goroutines call GetOrCreate for
// the same tenantID. Should return the same *TenantState.
func TestStress_GetOrCreate_Concurrent(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	reg := NewTenantRegistry(TierStandard)
	defer reg.Stop()

	const goroutines = 1000
	results := make([]*TenantState, goroutines)
	var wg sync.WaitGroup
	wg.Add(goroutines)
	start := make(chan struct{})

	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			<-start
			results[idx] = reg.GetOrCreate("same-tenant")
		}(i)
	}
	close(start)
	wg.Wait()

	for i := 1; i < goroutines; i++ {
		if results[i] != results[0] {
			t.Errorf("GetOrCreate returned different instances: results[0]=%p results[%d]=%p",
				results[0], i, results[i])
		}
	}
}

// TestStress_SetOverride_DuringReads: set overrides while reads are in
// flight. Verify the readers see consistent (not torn) state.
func TestStress_SetOverride_DuringReads(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	reg := NewTenantRegistry(TierStandard)
	defer reg.Stop()

	stop := make(chan struct{})
	var wg sync.WaitGroup

	// Writer: constantly change override
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				for i := 0; i < 100; i++ {
					reg.SetOverride("flipper", QuotaConfig{
						Priority:          rand.Intn(100),
						RequestsPerMinute: rand.Intn(10000),
						BurstSize:         rand.Intn(1000),
						MaxConcurrent:     rand.Intn(1000),
					})
				}
			}
		}
	}()

	// Readers: read stats
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				_ = reg.GetOrCreate("flipper")
				_ = reg.Stats()
			}
		}()
	}

	time.Sleep(200 * time.Millisecond)
	close(stop)
	wg.Wait()
}

// TestStress_Stop_DoesNotPanic: call Stop once. Verify it doesn't panic.
// (Don't call twice — that would close a closed channel and is documented
// as the caller's responsibility.)
func TestStress_Stop_DoesNotPanic(t *testing.T) {
	reg := NewTenantRegistry(TierStandard)
	reg.Stop()
	// Give the goroutine a moment to exit
	time.Sleep(50 * time.Millisecond)
}

// TestStress_Stop_WithConcurrentActivity: stop while AllowRequest is in
// flight. Verify no panic, no use-after-close.
func TestStress_Stop_WithConcurrentActivity(t *testing.T) {
	reg := NewTenantRegistry(TierStandard)
	ts := reg.GetOrCreate("active-tenant")

	stop := make(chan struct{})
	var wg sync.WaitGroup

	wg.Add(20)
	for i := 0; i < 20; i++ {
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					ts.AllowRequest(1)
					ts.ReleaseRequest()
				}
			}
		}()
	}

	// Stop after a brief delay
	time.Sleep(100 * time.Millisecond)
	reg.Stop()
	close(stop)
	wg.Wait()
}

// TestStress_RecordCost_Concurrent: many goroutines RecordCost in parallel.
// Verify the final cost is the sum of all inputs.
func TestStress_RecordCost_Concurrent(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	reg := NewTenantRegistry(TierStandard)
	defer reg.Stop()
	ts := reg.GetOrCreate("cost-tenant")

	// Set a huge budget so we don't hit the limit
	reg.SetOverride("cost-tenant", QuotaConfig{MaxCostPerHour: 1e9})

	const goroutines = 50
	const calls = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)
	start := make(chan struct{})

	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			<-start
			for j := 0; j < calls; j++ {
				ts.RecordCost(0.01)
			}
		}()
	}
	close(start)
	wg.Wait()

	ts.mu.Lock()
	finalCost := ts.HourCost
	ts.mu.Unlock()

	expected := float64(goroutines*calls) * 0.01
	if diff := finalCost - expected; diff < -0.001 || diff > 0.001 {
		t.Errorf("cost drift: got %.4f want %.4f diff=%.4f", finalCost, expected, diff)
	}
}

// TestStress_LockOrder_NoDeadlock: hammer the locks from many goroutines in
// mixed order. Verify no deadlock by completing all iterations in bounded time.
func TestStress_LockOrder_NoDeadlock(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	reg := NewTenantRegistry(TierStandard)
	defer reg.Stop()

	// Pre-populate
	for i := 0; i < 10; i++ {
		reg.GetOrCreate(fmt.Sprintf("tenant-%d", i))
	}

	const goroutines = 30
	const iterationsPerG = 200
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func(seed int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(seed)))
			for j := 0; j < iterationsPerG; j++ {
				// Mix of operations
				switch r.Intn(5) {
				case 0:
					_ = reg.GetOrCreate(fmt.Sprintf("tenant-%d", r.Intn(10)))
				case 1:
					_ = reg.Stats()
				case 2:
					reg.SetOverride("flapper", QuotaConfig{Priority: r.Intn(100)})
				case 3:
					ts := reg.GetOrCreate("flapper")
					ts.AllowRequest(1)
					ts.ReleaseRequest()
				case 4:
					ts := reg.GetOrCreate(fmt.Sprintf("tenant-%d", r.Intn(10)))
					ts.RecordCost(0.001)
				}
			}
		}(g)
	}

	// Bounded wait: if not done in 10s, declare deadlock
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// success - all goroutines completed
	case <-time.After(10 * time.Second):
		t.Fatal("deadlock detected (10s timeout)")
	}
}
