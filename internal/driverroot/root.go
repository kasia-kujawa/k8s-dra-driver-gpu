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

package driverroot

import (
	"fmt"
	"os"
	"path/filepath"
)

// Root represents a filesystem root under which NVIDIA driver files are located.
type Root string

// GetDriverLibraryPath returns the path to libnvidia-ml.so.1 under the root.
func (r Root) GetDriverLibraryPath() (string, error) {
	librarySearchPaths := []string{
		"/usr/lib64",
		"/usr/lib/x86_64-linux-gnu",
		"/usr/lib/aarch64-linux-gnu",
		"/lib64",
		"/lib/x86_64-linux-gnu",
		"/lib/aarch64-linux-gnu",
	}

	return r.FindFile("libnvidia-ml.so.1", librarySearchPaths...)
}

// GetNvidiaSMIPath returns the path to the nvidia-smi executable under the root.
func (r Root) GetNvidiaSMIPath() (string, error) {
	binarySearchPaths := []string{
		"/opt/bin",
		"/usr/bin",
		"/usr/sbin",
		"/bin",
		"/sbin",
	}

	return r.FindFile("nvidia-smi", binarySearchPaths...)
}

// IsDevRoot reports whether the root contains a /dev directory.
func (r Root) IsDevRoot() bool {
	stat, err := os.Stat(filepath.Join(string(r), "dev"))
	if err != nil {
		return false
	}
	return stat.IsDir()
}

// GetDevRoot returns the root itself if it contains a /dev directory, otherwise "/".
func (r Root) GetDevRoot() string {
	if r.IsDevRoot() {
		return string(r)
	}
	return "/"
}

// FindFile searches for name under the root, checking "/" and each of the
// searchIn subdirectories in order. Symlinks are resolved; the real path is returned.
func (r Root) FindFile(name string, searchIn ...string) (string, error) {
	for _, d := range append([]string{"/"}, searchIn...) {
		l := filepath.Join(string(r), d, name)
		resolved, err := filepath.EvalSymlinks(l)
		if err != nil {
			continue
		}
		return resolved, nil
	}
	return "", fmt.Errorf("error locating %q", name)
}
