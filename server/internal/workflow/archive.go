package workflow

import (
	"archive/tar"
	"context"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/odyssey-components/odyssey-go/bootstrap/silo"
)

const maxBundle = 1 << 30

func writeArchive(root string, names []string, dst io.Writer) error {
	tw := tar.NewWriter(dst)
	var total int64
	for _, name := range names {
		if err := filepath.WalkDir(filepath.Join(root, name), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				return nil
			}
			info, err := entry.Info()
			if err != nil {
				return err
			}
			if !info.Mode().IsRegular() || info.Size() > maxBundle-total {
				return errors.New("invalid or oversized Workflow payload")
			}
			total += info.Size()
			rel, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			header := &tar.Header{Name: filepath.ToSlash(rel), Size: info.Size(), Mode: 0600, Typeflag: tar.TypeReg}
			if err := tw.WriteHeader(header); err != nil {
				return err
			}
			f, err := os.Open(path)
			if err != nil {
				return err
			}
			_, err = io.Copy(tw, f)
			return errors.Join(err, f.Close())
		}); err != nil {
			return err
		}
	}
	return tw.Close()
}

func unpack(src io.Reader, root string, allowed func(string) bool) error {
	tr := tar.NewReader(src)
	var size int64
	for i := 0; i < 4096; i++ {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if !fs.ValidPath(h.Name) || h.Name == "." || strings.Contains(h.Name, "\\") || h.Typeflag != tar.TypeReg ||
			h.Size < 0 || h.Size > maxBundle-size || !allowed(h.Name) {
			return errors.New("invalid Workflow archive entry")
		}
		size += h.Size
		path := filepath.Join(root, filepath.FromSlash(h.Name))
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			return err
		}
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			return err
		}
		_, err = io.CopyN(f, tr, h.Size)
		if err := errors.Join(err, f.Close()); err != nil {
			return err
		}
	}
	return errors.New("too many Workflow archive files")
}

func send(ctx context.Context, store *silo.Client, key, root string, names []string) error {
	f, err := os.CreateTemp("", "aiops-bundle-")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	if err := writeArchive(root, names, f); err != nil {
		return err
	}
	size, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	n, err := store.PutAsset(ctx, key, f, size, "application/x-tar")
	if err == nil && n != size {
		return io.ErrShortWrite
	}
	return err
}
func receive(ctx context.Context, store *silo.Client, key, root string, allowed func(string) bool) error {
	r, size, err := store.Get(ctx, key)
	if err != nil {
		return err
	}
	defer r.Close()
	if size > maxBundle+8<<20 {
		return errors.New("Workflow archive too large")
	}
	return unpack(r, root, allowed)
}
