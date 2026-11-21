package main

import (
	"fmt"
	aiopsworkspace "odyssey/aiops/internal/workspace"
	"os"
	"path/filepath"
	"runtime"
)

// Resolve the workspace at startup, before workers change their working directory.
// A standalone installation must explicitly configure its installed interpreter.
func workspacePython() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	executable, _ := os.Executable()
	for _, start := range []string{cwd, filepath.Dir(executable)} {
		root, err := aiopsworkspace.Root(start)
		if err != nil {
			continue
		}
		if runtime.GOOS == "windows" {
			return filepath.Join(root, ".venv", "Scripts", "python.exe"), nil
		}
		return filepath.Join(root, ".venv", "bin", "python"), nil
	}
	return "", fmt.Errorf("cannot locate workspace .venv; start inside the repository or set --python")
}
