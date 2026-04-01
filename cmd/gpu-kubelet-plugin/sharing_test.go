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

	"github.com/stretchr/testify/require"
)

func TestRequiresHostIPC(t *testing.T) {
	tests := []struct {
		name     string
		caps     []string
		expected bool
	}{
		{
			name:     "three-part capability below threshold",
			caps:     []string{"6.1.0"},
			expected: true,
		},
		{
			name:     "exactly at threshold",
			caps:     []string{"7.0"},
			expected: false,
		},
		{
			name:     "above threshold",
			caps:     []string{"8.0"},
			expected: false,
		},
		{
			name:     "mixed: one below and one above threshold",
			caps:     []string{"8.0", "6.1.0"},
			expected: true,
		},
		{
			name:     "empty list",
			caps:     nil,
			expected: false,
		},
		{
			name:     "invalid capability string is skipped",
			caps:     []string{"not-a-version"},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.expected, requiresHostIPC(tt.caps))
		})
	}
}
