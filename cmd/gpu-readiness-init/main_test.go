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

func TestGpusReady(t *testing.T) {
	tests := map[string]struct {
		passthrough bool
		nvmllib     *mockNVML
		nvpcil      *nvpci.InterfaceMock
		wantReady   bool
		wantErr     bool
	}{
		"no-passthrough, nvml=0, pci=2 → retry": {
			nvmllib: nvmlOK(0),
			nvpcil:  nvpcilWith(pciGPU(pciDriverNvidia), pciGPU(pciDriverNvidia)),
		},
		"no-passthrough, nvml=1, pci=2 → retry": {
			nvmllib: nvmlOK(1),
			nvpcil:  nvpcilWith(pciGPU(pciDriverNvidia), pciGPU(pciDriverNvidia)),
		},
		"no-passthrough, nvml=2, pci=2 → ready": {
			nvmllib:   nvmlOK(2),
			nvpcil:    nvpcilWith(pciGPU(pciDriverNvidia), pciGPU(pciDriverNvidia)),
			wantReady: true,
		},
		"no-passthrough, nvml-transient-error → retry": {
			nvmllib: nvmlInitErr(nvml.ERROR_DRIVER_NOT_LOADED),
			nvpcil:  nvpcilWith(pciGPU(pciDriverNvidia)),
		},
		"no-passthrough, nvml-permanent-error → error": {
			nvmllib: nvmlInitErr(nvml.ERROR_UNKNOWN),
			nvpcil:  nvpcilWith(pciGPU(pciDriverNvidia)),
			wantErr: true,
		},
		"no-passthrough, no-pci-gpus → permanent error": {
			nvmllib: nvmlOK(0),
			nvpcil:  nvpcilWith(),
			wantErr: true,
		},
		"passthrough, nvml=0, vfio=2, pci=2 → ready": {
			passthrough: true,
			nvmllib:     nvmlOK(0),
			nvpcil:      nvpcilWith(pciGPU(pciDriverVfioPCI), pciGPU(pciDriverVfioPCI)),
			wantReady:   true,
		},
		"passthrough, nvml=1, unbound=1, pci=2 → retry": {
			passthrough: true,
			nvmllib:     nvmlOK(1),
			nvpcil:      nvpcilWith(pciGPU(pciDriverNvidia), pciGPU(pciDriverNone)),
		},
		"passthrough, nvml=0, vfio=2 → ready": {
			passthrough: true,
			nvmllib:     nvmlOK(0),
			nvpcil:      nvpcilWith(pciGPU(pciDriverVfioPCI), pciGPU(pciDriverVfioPCI)),
			wantReady:   true,
		},
		"passthrough, nvml=1, vfio=1, pci=2 → ready": {
			passthrough: true,
			nvmllib:     nvmlOK(1),
			nvpcil:      nvpcilWith(pciGPU(pciDriverNvidia), pciGPU(pciDriverVfioPCI)),
			wantReady:   true,
		},
		"passthrough, unrecognised-driver, nvml=1, nouveau=1, pci=2 → retry": {
			passthrough: true,
			nvmllib:     nvmlOK(1),
			nvpcil:      nvpcilWith(pciGPU(pciDriverNvidia), pciGPU("nouveau")),
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			done, err := gpusReady(tc.nvmllib, tc.nvpcil, tc.passthrough)

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
