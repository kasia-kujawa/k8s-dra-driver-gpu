/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSetMpsShmMountPath(t *testing.T) {
	testCases := map[string]struct {
		setupDriverRoot   func(t *testing.T) string
		expectedMountPath string
	}{
		"sh at /bin/sh": {
			setupDriverRoot: func(t *testing.T) string {
				t.Helper()
				dir := t.TempDir()
				require.NoError(t, os.MkdirAll(filepath.Join(dir, "bin"), 0755))
				require.NoError(t, os.WriteFile(filepath.Join(dir, "bin", "sh"), []byte{}, 0755))
				return dir
			},
			expectedMountPath: MpsChrootShmMountPath,
		},
		"sh at /usr/bin/sh": {
			setupDriverRoot: func(t *testing.T) string {
				t.Helper()
				dir := t.TempDir()
				require.NoError(t, os.MkdirAll(filepath.Join(dir, "usr", "bin"), 0755))
				require.NoError(t, os.WriteFile(filepath.Join(dir, "usr", "bin", "sh"), []byte{}, 0755))
				return dir
			},
			expectedMountPath: MpsChrootShmMountPath,
		},
		"no sh in driver root — case for GKE COS": {
			setupDriverRoot: func(t *testing.T) string {
				t.Helper()
				return t.TempDir()
			},
			expectedMountPath: MpsDefaultShmMountPath,
		},
		"driver root does not exist": {
			setupDriverRoot: func(t *testing.T) string {
				t.Helper()
				return filepath.Join(t.TempDir(), "nonexistent")
			},
			expectedMountPath: MpsDefaultShmMountPath,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			dir := tc.setupDriverRoot(t)
			driverRootMountDir = dir
			t.Cleanup(func() {
				driverRootMountDir = "/driver-root"
			})
			require.Equal(t, tc.expectedMountPath, setMpsShmMountPath())
		})
	}
}
