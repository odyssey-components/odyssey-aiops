package go1

import (
	"reflect"
	"testing"
)

func TestPreferredEnvironment(t *testing.T) {
	for _, tc := range []struct {
		available string
		attempts  []string
	}{
		{"cuda", []string{"cuda"}},
		{"rocm", []string{"cuda", "rocm"}},
		{"cpu", []string{"cuda", "rocm", "cpu"}},
		{"", []string{"cuda", "rocm", "cpu"}},
	} {
		var attempts []string
		result := preferredEnvironment(func(device string) map[string]any {
			attempts = append(attempts, device)
			return map[string]any{"device": device, "ready": device == tc.available}
		})
		if !reflect.DeepEqual(attempts, tc.attempts) || result["ready"] != (tc.available != "") {
			t.Fatalf("available %q: attempts %v, result %v", tc.available, attempts, result)
		}
	}
}
