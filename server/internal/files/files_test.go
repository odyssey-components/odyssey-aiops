package files

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestBoundedRegular(t *testing.T) {
	root := t.TempDir()
	if e := os.WriteFile(filepath.Join(root, "ok"), []byte("abcd"), 0600); e != nil {
		t.Fatal(e)
	}
	if _, e := Read(root, "ok", 3); !errors.Is(e, ErrLarge) {
		t.Fatal(e)
	}
	for _, p := range []string{"../outside", "."} {
		if _, e := Read(root, p, 10); e == nil {
			t.Fatal("accepted", p)
		}
	}
	outside := t.TempDir()
	if e := os.WriteFile(filepath.Join(outside, "secret"), []byte("secret"), 0600); e != nil {
		t.Fatal(e)
	}
	if e := os.Symlink(outside, filepath.Join(root, "link")); e == nil {
		if _, e := Read(root, "link/secret", 10); e == nil {
			t.Fatal("followed symlink")
		}
	}
}
