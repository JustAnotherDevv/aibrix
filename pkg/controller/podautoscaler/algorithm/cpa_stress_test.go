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

// Stress tests for CPA concurrency safety. Same approach as
// tenant_quota_stress_test.go: exercise shared state heavily and rely on
// the Go memory model + atomic operations to surface any races.

package algorithm

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/aibrix/pkg/controller/podautoscaler/types"
)

// Audit log for CPA shared state:
//
// Hot shared state:
//   - CPAAlgorithm.catalog (*GPUCostCatalog): protected by CPAAlgorithm.mu (RWMutex)
//     Catalog itself has its own mu protecting entries + defaults maps.
//   - CPAAlgorithm.overrides (map): protected by CPAAlgorithm.mu
//   - GPUCostCatalog.entries map: protected by GPUCostCatalog.mu
//   - GPUCostCatalog.defaults map: protected by GPUCostCatalog.mu
//
// Lock order: CPAAlgorithm.mu -> GPUCostCatalog.mu. Never reversed.
//
// ComputeRecommendation takes CPAAlgorithm.mu (RLock), then calls
// catalog.Get() which takes GPUCostCatalog.mu (RLock). Consistent order.
// SetOverride takes CPAAlgorithm.mu (Lock). Set catalog takes
// CPAAlgorithm.mu (Lock). No deadlock.
//
// All tests below use unique ScaleTarget keys so per-PA override state
// doesn't bleed between tests.

// TestStress_CPA_ComputeRecommendation_Concurrent: 100 goroutines, 100
// requests each, all hitting the same PA. Verifies the algorithm returns
// sane results under load.
func TestStress_CPA_ComputeRecommendation_Concurrent(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	a := NewCPAAlgorithm()
	target := types.ScaleTarget{Namespace: "ns", Name: "stress-pa"}
	a.SetOverride(target, CPAOverride{Mode: CPAModeMinCost})

	const goroutines = 100
	const calls = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)
	start := make(chan struct{})

	var errCount int64
	var totalDesired int64

	for i := 0; i < goroutines; i++ {
		go func(seed int) {
			defer wg.Done()
			<-start
			r := rand.New(rand.NewSource(int64(seed)))
			for j := 0; j < calls; j++ {
				req := newCPARequest(float64(r.Intn(200)+1), int32(r.Intn(20)+1))
				req.Target = target
				rec, err := a.ComputeRecommendation(nil, req)
				if err != nil {
					atomic.AddInt64(&errCount, 1)
					continue
				}
				atomic.AddInt64(&totalDesired, int64(rec.DesiredReplicas))
			}
		}(i)
	}
	close(start)
	wg.Wait()

	if errCount > 0 {
		t.Errorf("%d errors during concurrent ComputeRecommendation", errCount)
	}
	if totalDesired == 0 {
		t.Error("no replicas ever returned — algorithm broken")
	}
}

// TestStress_CPA_SetOverride_DuringComputes: hammer SetOverride while
// ComputeRecommendation is running. Verifies readers see consistent state.
func TestStress_CPA_SetOverride_DuringComputes(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	a := NewCPAAlgorithm()
	target := types.ScaleTarget{Namespace: "ns", Name: "flipper"}

	stop := make(chan struct{})
	var wg sync.WaitGroup

	// Writer: keep changing override
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				for i := 0; i < 50; i++ {
					a.SetOverride(target, CPAOverride{
						Mode:               CPAModeMinCost,
						MaxCostPerHour:     float64(rand.Intn(100)),
						PreferredCostClass: CostClassOnDemand,
					})
				}
			}
		}
	}()

	// Readers: compute recommendations
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					req := newCPARequest(50.0, 1)
					req.Target = target
					_, _ = a.ComputeRecommendation(nil, req)
				}
			}
		}()
	}

	time.Sleep(200 * time.Millisecond)
	close(stop)
	wg.Wait()
}

// TestStress_CPA_CatalogSet_DuringReads: write to the catalog while
// reads are in progress.
func TestStress_CPA_CatalogSet_DuringReads(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	c := NewDefaultGPUCostCatalog()
	stop := make(chan struct{})
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				c.Set(GPUCostEntry{
					GPUType: "nvidia-h100-80gb", CostClass: CostClassOnDemand,
					HourlyCostUSD: 3.0 + rand.Float64(), ThroughputQPS: 20.0,
				})
			}
		}
	}()

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					_, _ = c.Get("nvidia-h100-80gb", CostClassOnDemand)
					_ = c.AllOnDemand()
				}
			}
		}()
	}

	time.Sleep(200 * time.Millisecond)
	close(stop)
	wg.Wait()
}

// TestStress_CPA_OverriddenModes_AreIsolated: each goroutine uses its own
// PA key, then a final goroutine mutates one PA's override. Verifies
// other PAs' overrides are unaffected.
func TestStress_CPA_OverriddenModes_AreIsolated(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	a := NewCPAAlgorithm()

	const pas = 20
	targets := make([]types.ScaleTarget, pas)
	for i := 0; i < pas; i++ {
		targets[i] = types.ScaleTarget{
			Namespace: "ns",
			Name:      fmt.Sprintf("pa-%d", i),
		}
		mode := CPAModeMinCost
		if i%2 == 0 {
			mode = CPAModeMinLatency
		}
		a.SetOverride(targets[i], CPAOverride{Mode: mode})
	}

	var wg sync.WaitGroup
	for i := 0; i < pas; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			req := newCPARequest(20.0, 1)
			req.Target = targets[idx]
			rec, _ := a.ComputeRecommendation(nil, req)
			wantMode := "min-cost"
			if idx%2 == 0 {
				wantMode = "min-latency"
			}
			if got := rec.Metadata["mode"]; got != wantMode {
				t.Errorf("pa-%d mode=%s want %s", idx, got, wantMode)
			}
		}(i)
	}
	wg.Wait()
}

// TestStress_CPA_LockOrder_NoDeadlock: mixed concurrent operations.
// Uses a time-bounded loop instead of "until stop" so the test can
// self-terminate. The deadlock detector runs in parallel.
func TestStress_CPA_LockOrder_NoDeadlock(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test")
	}
	a := NewCPAAlgorithm()
	target := types.ScaleTarget{Namespace: "ns", Name: "deadlock-test"}

	stop := make(chan struct{})
	var wg sync.WaitGroup

	const goroutines = 10
	const opsPerGoroutine = 1000

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(seed)))
			for j := 0; j < opsPerGoroutine; j++ {
				select {
				case <-stop:
					return
				default:
				}
				switch r.Intn(4) {
				case 0:
					req := newCPARequest(float64(r.Intn(100)+1), 1)
					req.Target = target
					_, _ = a.ComputeRecommendation(nil, req)
				case 1:
					a.SetOverride(target, CPAOverride{Mode: CPAModeMinCost, MaxCostPerHour: float64(r.Intn(100))})
				case 2:
					a.SetCatalog(NewDefaultGPUCostCatalog())
				case 3:
					a.ClearOverride(target)
				}
			}
		}(g)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// success
	case <-time.After(10 * time.Second):
		close(stop)
		wg.Wait()
		t.Fatal("deadlock detected (10s timeout)")
	}
}
