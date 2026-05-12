/*
Copyright The Kubernetes Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"context"
	"errors"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

// ErrDeviceEnumerationTimeout is returned by enumerateDevicesWithRetry
// when the retry budget is exhausted without discovering any devices.
var ErrDeviceEnumerationTimeout = errors.New("device enumeration timed out before any GPU was discovered")

type deviceEnumerator interface {
	enumerateAllPossibleDevices() (*PerGPUAllocatableDevices, error)
}

// enumerateDevices performs a single GPU enumeration attempt and decides whether the result is ready or a retry is needed.
//
// The decision tree, per Varun's spec on PR #1009:
//   - empty allocatable, empty checkpoint               -> retry (first-boot, NVML initialising)
//   - empty allocatable, non-empty checkpoint           -> accept (only happens if GPU is unhealthy, retrying won't help)
//   - allocatable has any gpu/mig device                -> accept
//   - allocatable is vfio-only, non-empty checkpoint    -> accept
//   - allocatable is vfio-only, empty checkpoint        -> retry

// A non-empty checkpoint means we are past first boot — see comments in checkpointHasPreparedDevices.
func enumerateDevices(nvdevlib deviceEnumerator, cp *Checkpoint) (*PerGPUAllocatableDevices, error) {
	perGPU, err := nvdevlib.enumerateAllPossibleDevices()
	if err != nil {
		if isTransientNVMLError(err) {
			klog.Infof("Transient NVML error on enumeration attempt; will retry: %v", err)
			return nil, nil
		}
		return nil, fmt.Errorf("error enumerating all possible devices: %w", err)
	}

	if len(perGPU.allocatablesMap) == 0 {
		if checkpointHasPreparedDevices(cp) {
			klog.Infof("No GPU devices discovered via NVML but the checkpoint has prepared devices, not retrying (unhealthy device state, retry won't help)")
			return perGPU, nil
		}
		klog.Infof("No GPU devices discovered on enumeration attempt; will retry")
		return nil, nil
	}

	if allocatableHasNonVfio(perGPU) {
		return perGPU, nil
	}

	// Allocatable is vfio-only, if checkpoint is not empty, then GPU is ready
	if checkpointHasPreparedDevices(cp) {
		return perGPU, nil
	}
	klog.Infof("Only vfio devices visible and checkpoint is empty; will retry (NVML may still be initialising)")
	return nil, nil
}

// allocatableHasNonVfio reports whether perGPU contains any non-vfio (gpu or mig) device.
// Returns false for both empty maps and vfio-only maps.
func allocatableHasNonVfio(perGPU *PerGPUAllocatableDevices) bool {
	for _, devices := range perGPU.allocatablesMap {
		for _, dev := range devices {
			if dev.Type() != VfioDeviceType {
				return true
			}
		}
	}
	return false
}

// checkpointHasPreparedDevices reports whether cp contains any prepared device (of any type, in any claim state).
func checkpointHasPreparedDevices(cp *Checkpoint) bool {
	if cp == nil || cp.V2 == nil {
		return false
	}
	for _, claim := range cp.V2.PreparedClaims {
		for _, group := range claim.PreparedDevices {
			if len(group.Devices) > 0 {
				return true
			}
		}
	}
	return false
}

// enumerateDevicesWithRetry retries until at least one device is found, the context is cancelled, or the retry budget is exhausted.
// Transient NVML errors are retried, all other errors propagate immediately.
func enumerateDevicesWithRetry(ctx context.Context, nvdevlib deviceEnumerator, backoff wait.Backoff, cp *Checkpoint) (*PerGPUAllocatableDevices, error) {
	totalSteps := backoff.Steps
	var perGPUAllocatable *PerGPUAllocatableDevices
	err := wait.ExponentialBackoffWithContext(ctx, backoff, func(ctx context.Context) (bool, error) {
		var err error
		perGPUAllocatable, err = enumerateDevices(nvdevlib, cp)
		if err != nil {
			return false, err
		}
		return perGPUAllocatable != nil, nil
	})
	switch {
	case err == nil:
		return perGPUAllocatable, nil
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return nil, fmt.Errorf("context cancelled while waiting for GPU devices: %w", err)
	case wait.Interrupted(err):
		klog.Errorf("No GPU devices found after %d attempts; failing startup to avoid publishing an empty ResourceSlice", totalSteps)
		return nil, ErrDeviceEnumerationTimeout
	default:
		return nil, err
	}
}

// isTransientNVMLError reports whether err is an NVML "not ready yet" error expected during early driver init.
// Errors that may indicate real hardware problems are treated as permanent.
// errors.Is walks the %w-wrap chain and matches by == against the bare nvml.Return at the bottom.
func isTransientNVMLError(err error) bool {
	return errors.Is(err, nvml.ERROR_UNINITIALIZED) || errors.Is(err, nvml.ERROR_DRIVER_NOT_LOADED)
}

// deviceEnumerationBackoff builds the retry cadence for background GPU enumeration.
func deviceEnumerationBackoff(flags *Flags) wait.Backoff {
	return wait.Backoff{
		Duration: 1 * time.Second,
		Factor:   2.0,
		Jitter:   0.2,
		Cap:      flags.deviceEnumerationRetryMaxInterval,
		Steps:    flags.deviceEnumerationRetrySteps,
	}
}
