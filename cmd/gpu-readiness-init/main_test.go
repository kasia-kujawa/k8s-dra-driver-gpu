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
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/NVIDIA/go-nvlib/pkg/nvpci"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockNVML struct {
	initReturn  nvml.Return
	countReturn nvml.Return
	count       int
}

func (m *mockNVML) InitWithFlags(uint32) nvml.Return   { return m.initReturn }
func (m *mockNVML) Shutdown() nvml.Return              { return nvml.SUCCESS }
func (m *mockNVML) DeviceGetCount() (int, nvml.Return) { return m.count, m.countReturn }

func nvmlOK(count int) *mockNVML {
	return &mockNVML{initReturn: nvml.SUCCESS, countReturn: nvml.SUCCESS, count: count}
}

func nvmlInitErr(ret nvml.Return) *mockNVML {
	return &mockNVML{initReturn: ret}
}

func pciGPU(driver string) *nvpci.NvidiaPCIDevice {
	return &nvpci.NvidiaPCIDevice{Driver: driver}
}

func nvpcilWith(gpus ...*nvpci.NvidiaPCIDevice) *nvpci.InterfaceMock {
	return &nvpci.InterfaceMock{
		GetGPUsFunc: func() ([]*nvpci.NvidiaPCIDevice, error) { return gpus, nil },
	}
}

// makeDevNodes creates /dev/nvidia<indices> and /dev/nvidiactl under dir.
func makeDevNodes(t *testing.T, dir string, indices ...int) string {
	t.Helper()
	devDir := filepath.Join(dir, "dev")
	require.NoError(t, os.MkdirAll(devDir, 0o755))
	for _, i := range indices {
		require.NoError(t, os.WriteFile(filepath.Join(devDir, fmt.Sprintf("nvidia%d", i)), nil, 0o600))
	}
	require.NoError(t, os.WriteFile(filepath.Join(devDir, "nvidiactl"), nil, 0o600))
	return dir
}

func TestNvidiaDevNodesReady(t *testing.T) {
	tests := map[string]struct {
		setup     func(t *testing.T) string
		gpuCount  int
		wantReady bool
	}{
		"sequential indices 0,1 present, count=2 → ready": {
			setup:     func(t *testing.T) string { return makeDevNodes(t, t.TempDir(), 0, 1) },
			gpuCount:  2,
			wantReady: true,
		},
		"non-sequential indices 0,2 present, count=2 → ready": {
			setup:     func(t *testing.T) string { return makeDevNodes(t, t.TempDir(), 0, 2) },
			gpuCount:  2,
			wantReady: true,
		},
		"only one node present, count=2 → not ready": {
			setup:    func(t *testing.T) string { return makeDevNodes(t, t.TempDir(), 0) },
			gpuCount: 2,
		},
		"no nodes present, count=1 → not ready": {
			setup: func(t *testing.T) string {
				dir := t.TempDir()
				require.NoError(t, os.MkdirAll(filepath.Join(dir, "dev"), 0o755))
				return dir
			},
			gpuCount: 1,
		},
		"nvidia nodes present but nvidiactl missing → not ready": {
			setup: func(t *testing.T) string {
				dir := t.TempDir()
				devDir := filepath.Join(dir, "dev")
				require.NoError(t, os.MkdirAll(devDir, 0o755))
				require.NoError(t, os.WriteFile(filepath.Join(devDir, "nvidia0"), nil, 0o600))
				return dir
			},
			gpuCount: 1,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			devRoot := tc.setup(t)
			assert.Equal(t, tc.wantReady, nvidiaDevNodesReady(devRoot, tc.gpuCount))
		})
	}
}

func TestGpusReady(t *testing.T) {
	tests := map[string]struct {
		passthrough  bool
		nvmllib      *mockNVML
		nvpcil       *nvpci.InterfaceMock
		setupDevRoot func(t *testing.T) string
		wantReady    bool
		wantErr      bool
	}{
		"no-passthrough, nvml=0, pci=2 → retry": {
			nvmllib:      nvmlOK(0),
			nvpcil:       nvpcilWith(pciGPU(pciDriverNvidia), pciGPU(pciDriverNvidia)),
			setupDevRoot: func(t *testing.T) string { return t.TempDir() },
		},
		"no-passthrough, nvml=1, pci=2 → retry": {
			nvmllib:      nvmlOK(1),
			nvpcil:       nvpcilWith(pciGPU(pciDriverNvidia), pciGPU(pciDriverNvidia)),
			setupDevRoot: func(t *testing.T) string { return t.TempDir() },
		},
		"no-passthrough, nvml=2, pci=2, dev nodes present → ready": {
			nvmllib: nvmlOK(2),
			nvpcil:  nvpcilWith(pciGPU(pciDriverNvidia), pciGPU(pciDriverNvidia)),
			setupDevRoot: func(t *testing.T) string {
				return makeDevNodes(t, t.TempDir(), 0, 1)
			},
			wantReady: true,
		},
		"no-passthrough, nvml=2, pci=2, dev nodes missing → retry": {
			nvmllib:      nvmlOK(2),
			nvpcil:       nvpcilWith(pciGPU(pciDriverNvidia), pciGPU(pciDriverNvidia)),
			setupDevRoot: func(t *testing.T) string { return t.TempDir() },
		},
		"no-passthrough, nvml-transient-error → retry": {
			nvmllib:      nvmlInitErr(nvml.ERROR_DRIVER_NOT_LOADED),
			nvpcil:       nvpcilWith(pciGPU(pciDriverNvidia)),
			setupDevRoot: func(t *testing.T) string { return t.TempDir() },
		},
		"no-passthrough, nvml-permanent-error → error": {
			nvmllib:      nvmlInitErr(nvml.ERROR_UNKNOWN),
			nvpcil:       nvpcilWith(pciGPU(pciDriverNvidia)),
			setupDevRoot: func(t *testing.T) string { return t.TempDir() },
			wantErr:      true,
		},
		"no-passthrough, no-pci-gpus → permanent error": {
			nvmllib:      nvmlOK(0),
			nvpcil:       nvpcilWith(),
			setupDevRoot: func(t *testing.T) string { return t.TempDir() },
			wantErr:      true,
		},
		"passthrough, nvml=0, vfio=2, pci=2 → ready": {
			passthrough: true,
			nvmllib:     nvmlOK(0),
			nvpcil:      nvpcilWith(pciGPU(pciDriverVfioPCI), pciGPU(pciDriverVfioPCI)),
			// devRoot not consulted for passthrough path
			setupDevRoot: func(t *testing.T) string { return t.TempDir() },
			wantReady:    true,
		},
		"passthrough, nvml=1, unbound=1, pci=2 → retry": {
			passthrough:  true,
			nvmllib:      nvmlOK(1),
			nvpcil:       nvpcilWith(pciGPU(pciDriverNvidia), pciGPU(pciDriverNone)),
			setupDevRoot: func(t *testing.T) string { return t.TempDir() },
		},
		"passthrough, nvml=0, vfio=2 → ready": {
			passthrough:  true,
			nvmllib:      nvmlOK(0),
			nvpcil:       nvpcilWith(pciGPU(pciDriverVfioPCI), pciGPU(pciDriverVfioPCI)),
			setupDevRoot: func(t *testing.T) string { return t.TempDir() },
			wantReady:    true,
		},
		"passthrough, nvml=1, vfio=1, pci=2 → ready": {
			passthrough:  true,
			nvmllib:      nvmlOK(1),
			nvpcil:       nvpcilWith(pciGPU(pciDriverNvidia), pciGPU(pciDriverVfioPCI)),
			setupDevRoot: func(t *testing.T) string { return t.TempDir() },
			wantReady:    true,
		},
		"passthrough, unrecognised-driver, nvml=1, nouveau=1, pci=2 → retry": {
			passthrough:  true,
			nvmllib:      nvmlOK(1),
			nvpcil:       nvpcilWith(pciGPU(pciDriverNvidia), pciGPU("nouveau")),
			setupDevRoot: func(t *testing.T) string { return t.TempDir() },
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			devRoot := tc.setupDevRoot(t)
			done, err := gpusReady(tc.nvmllib, tc.nvpcil, tc.passthrough, devRoot)

			if tc.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, tc.wantReady, done)
		})
	}
}

func TestIsTransientNVMLError(t *testing.T) {
	tests := map[string]struct {
		err       error
		transient bool
	}{
		"ERROR_UNINITIALIZED":      {nvml.ERROR_UNINITIALIZED, true},
		"ERROR_DRIVER_NOT_LOADED":  {nvml.ERROR_DRIVER_NOT_LOADED, true},
		"ERROR_UNKNOWN":            {nvml.ERROR_UNKNOWN, false},
		"ERROR_INSUFFICIENT_POWER": {nvml.ERROR_INSUFFICIENT_POWER, false},
		"non-nvml error":           {assert.AnError, false},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.transient, isTransientNVMLError(tc.err))
		})
	}
}
