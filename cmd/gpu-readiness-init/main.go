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

// gpu-readiness-init is an init-container entrypoint that blocks until all GPUs on the node are accounted for,
// either via NVML (nvidia driver) or via sysfs (vfio-pci driver, when PassthroughSupport is enabled).
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/NVIDIA/go-nvlib/pkg/nvpci"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/urfave/cli/v2"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"

	"sigs.k8s.io/dra-driver-nvidia-gpu/internal/driverroot"
	"sigs.k8s.io/dra-driver-nvidia-gpu/internal/info"
	"sigs.k8s.io/dra-driver-nvidia-gpu/pkg/featuregates"
	pkgflags "sigs.k8s.io/dra-driver-nvidia-gpu/pkg/flags"
)

const (
	pciDriverNvidia  = "nvidia"
	pciDriverVfioPCI = "vfio-pci"
	pciDriverNone    = ""
)

type Flags struct {
	containerDriverRoot               string
	deviceEnumerationRetrySteps       int
	deviceEnumerationRetryMaxInterval time.Duration
}

func main() {
	if err := newApp().Run(os.Args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func newApp() *cli.App {
	loggingConfig := pkgflags.NewLoggingConfig()
	flags := &Flags{}

	cliFlags := []cli.Flag{
		&cli.StringFlag{
			Name:        "container-driver-root",
			Value:       "/driver-root",
			Usage:       "path where the NVIDIA driver root is mounted in the container",
			Destination: &flags.containerDriverRoot,
			EnvVars:     []string{"DRIVER_ROOT_CTR_PATH"},
		},
		&cli.IntFlag{
			Name:        "device-enumeration-retry-steps",
			Value:       15,
			Usage:       "maximum number of enumeration retry attempts",
			Destination: &flags.deviceEnumerationRetrySteps,
			EnvVars:     []string{"DEVICE_ENUMERATION_RETRY_STEPS"},
		},
		&cli.DurationFlag{
			Name:        "device-enumeration-retry-max-interval",
			Value:       30 * time.Second,
			Usage:       "maximum wait between enumeration retry attempts",
			Destination: &flags.deviceEnumerationRetryMaxInterval,
			EnvVars:     []string{"DEVICE_ENUMERATION_RETRY_MAX_INTERVAL"},
		},
	}
	cliFlags = append(cliFlags, loggingConfig.Flags()...)

	app := &cli.App{
		Name:            "gpu-readiness-init",
		Usage:           "init container that blocks until all GPUs are accounted for via NVML or vfio-pci",
		ArgsUsage:       " ",
		HideHelpCommand: true,
		Flags:           cliFlags,
		Before: func(c *cli.Context) error {
			if c.Args().Len() > 0 {
				return fmt.Errorf("arguments not supported: %v", c.Args().Slice())
			}
			return loggingConfig.Apply()
		},
		Action: func(c *cli.Context) error {
			ctx, cancel := signal.NotifyContext(c.Context, syscall.SIGTERM, syscall.SIGINT)
			defer cancel()
			return run(ctx, flags)
		},
		Version: info.GetVersionString(),
	}

	f, ok := cli.VersionFlag.(*cli.BoolFlag)
	if ok {
		f.Aliases = nil
	}

	return app
}

func run(ctx context.Context, flags *Flags) error {
	libraryPath, err := driverroot.Root(flags.containerDriverRoot).GetDriverLibraryPath()
	if err != nil {
		return fmt.Errorf("locate libnvidia-ml.so.1: %w", err)
	}
	klog.Infof("using driver library: %s", libraryPath)

	nvmllib := nvml.New(nvml.WithLibraryPath(libraryPath))
	nvpcil := nvpci.New()

	backoff := wait.Backoff{
		Duration: 1 * time.Second,
		Factor:   2.0,
		Jitter:   0.2,
		Cap:      flags.deviceEnumerationRetryMaxInterval,
		Steps:    flags.deviceEnumerationRetrySteps,
	}

	passthroughEnabled := featuregates.Enabled(featuregates.PassthroughSupport)
	err = wait.ExponentialBackoffWithContext(ctx, backoff, func(ctx context.Context) (bool, error) {
		return gpusReady(nvmllib, nvpcil, passthroughEnabled)
	})

	switch {
	case err == nil:
		return nil
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return fmt.Errorf("context cancelled while waiting for GPU devices: %w", err)
	case wait.Interrupted(err):
		return fmt.Errorf("GPU readiness check did not succeed after %d attempts", flags.deviceEnumerationRetrySteps)
	default:
		return err
	}
}

// nvmlClient is the subset of nvml.Interface used by this package.
type nvmlClient interface {
	InitWithFlags(uint32) nvml.Return
	Shutdown() nvml.Return
	DeviceGetCount() (int, nvml.Return)
}

// gpusReady returns (true, nil) when every NVIDIA GPU on the node is accounted
// for by either NVML (bound to the nvidia driver) or sysfs showing vfio-pci binding (when PassthroughSupport is enabled).
//
// Returns (false, nil) for transient not-yet-ready states so the caller retries,
// and (false, err) for permanent failures.
//
// Decision matrix:
//
//	PassthroughSupport disabled:
//	  nvml_count == total_pci_count                 → ready
//	  nvml_count < total_pci_count                  → retry
//
//	PassthroughSupport enabled:
//	  nvml_count + vfio_count == total_pci_count    → ready
//	  some GPUs unbound                             → retry
//	  counts don't add up                           → retry
func gpusReady(nvmllib nvmlClient, nvpcil nvpci.Interface, passthroughEnabled bool) (bool, error) {
	nvmlCount, err := getNVMLDeviceCount(nvmllib)
	if err != nil {
		if isTransientNVMLError(err) {
			klog.Infof("transient NVML error (driver may still be initializing), retrying: %v", err)
			return false, nil
		}
		return false, err
	}

	pciGPUs, err := nvpcil.GetGPUs()
	if err != nil {
		return false, fmt.Errorf("enumerating PCI GPUs: %w", err)
	}

	totalPCI := len(pciGPUs)
	if totalPCI == 0 {
		return false, fmt.Errorf("no NVIDIA GPU PCI devices found on this node")
	}

	if !passthroughEnabled {
		if nvmlCount < totalPCI {
			klog.Infof("NVML returned %d GPU(s), %d expected — driver still initializing, retrying...", nvmlCount, totalPCI)
			return false, nil
		}
		klog.Infof("found %d GPU(s) via NVML", nvmlCount)
		return true, nil
	}

	// PassthroughSupport is enabled: GPUs bound to vfio-pci are invisible to NVML but visible in sysfs
	vfioCount, unboundCount := 0, 0
	for _, gpu := range pciGPUs {
		switch gpu.Driver {
		case pciDriverVfioPCI:
			vfioCount++
		case pciDriverNvidia:
			// accounted for by NVML
		case pciDriverNone:
			unboundCount++
		}
	}

	if unboundCount > 0 {
		klog.Infof("%d GPU(s) not yet bound to any driver, retrying...", unboundCount)
		return false, nil
	}

	if nvmlCount+vfioCount < totalPCI {
		// Some GPU is bound to an unrecognised driver — retry until budget exhausted.
		klog.Infof("GPU accounting mismatch: pci=%d nvml=%d vfio-pci=%d, retrying...", totalPCI, nvmlCount, vfioCount)
		return false, nil
	}

	klog.Infof("all %d GPU(s) accounted for: %d via NVML, %d via vfio-pci", totalPCI, nvmlCount, vfioCount)
	return true, nil
}

// getNVMLDeviceCount initializes NVML, returns the number of visible GPUs, then shuts NVML down.
// Any NVML failure is returned as an error for the caller to classify.
func getNVMLDeviceCount(nvmllib nvmlClient) (int, error) {
	// It's possible there are no GPUs available in NVML
	// (e.g. all GPUs prepared in passthrough mode).
	// We use INIT_FLAG_NO_GPUS to avoid failing if there are no GPUs.
	ret := nvmllib.InitWithFlags(nvml.INIT_FLAG_NO_GPUS)
	if ret != nvml.SUCCESS {
		return 0, ret
	}
	defer func() {
		if r := nvmllib.Shutdown(); r != nvml.SUCCESS {
			klog.Warningf("nvml Shutdown: %v", r)
		}
	}()

	count, ret := nvmllib.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return 0, ret
	}
	return count, nil
}

// isTransientNVMLError reports whether err is an NVML "not ready yet" error
// expected during early driver initialisation.
func isTransientNVMLError(err error) bool {
	var r nvml.Return
	if !errors.As(err, &r) {
		return false
	}
	return r == nvml.ERROR_UNINITIALIZED || r == nvml.ERROR_DRIVER_NOT_LOADED
}
