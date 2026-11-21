package workspace

import (
	"fmt"
	"os"
	"path/filepath"
)

// Root finds the local workspace for either an Odyssey submodule or an
// independent AIOps checkout. Installed services can specify Python explicitly.
func Root(start string) (string, error) {
	for dir := start; ; dir = filepath.Dir(dir) {
		if regular(dir, "pyproject.toml") {
			if regular(dir, "go.work") {
				return dir, nil
			}
			if regular(dir, "server/go.mod") && regular(dir, "web/maze.json") {
				container := filepath.Dir(dir)
				parent := filepath.Dir(container)
				if filepath.Base(container) == "repositories" && regular(parent, "go.work") && regular(parent, "pyproject.toml") {
					return parent, nil
				}
				return dir, nil
			}
		}
		if filepath.Dir(dir) == dir {
			return "", fmt.Errorf("cannot locate workspace from %q", start)
		}
	}
}

func regular(root, name string) bool {
	info, err := os.Stat(filepath.Join(root, filepath.FromSlash(name)))
	return err == nil && info.Mode().IsRegular()
}
