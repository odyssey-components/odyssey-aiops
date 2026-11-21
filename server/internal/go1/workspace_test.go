package go1

import (
	"odyssey/aiops/internal/workspace"
	"os"
	"testing"
)

func testWorkspaceRoot(t *testing.T) string {
	t.Helper()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root, err := workspace.Root(cwd)
	if err != nil {
		t.Fatal(err)
	}
	return root
}
