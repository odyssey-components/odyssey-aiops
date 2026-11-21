// Package files reads bounded regular files without following symlinks.
package files

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
)

var ErrLarge = errors.New("file exceeds size limit")

func ID(s string) bool { return s != "" && s != "." && s != ".." && !strings.ContainsAny(s, "/\\\x00") }
func Read(root, relative string, limit int64) ([]byte, error) {
	if filepath.IsAbs(relative) {
		return nil, os.ErrNotExist
	}
	clean := filepath.Clean(relative)
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return nil, os.ErrNotExist
	}
	base, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	// OpenRoot anchors every lookup and prevents symlink escapes and rename races.
	r, err := os.OpenRoot(base)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	current := ""
	for _, part := range strings.Split(clean, string(filepath.Separator)) {
		current = filepath.Join(current, part)
		info, e := r.Lstat(current)
		if e != nil {
			return nil, e
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return nil, os.ErrNotExist
		}
	}
	f, err := r.Open(clean)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, os.ErrNotExist
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, ErrLarge
	}
	return b, nil
}
