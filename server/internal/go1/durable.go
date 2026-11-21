// PostgreSQL owns the Go1 catalog. Silo owns the immutable worker payloads.
package go1

import (
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/odyssey-components/odyssey-go/bootstrap/silo"
)

type Durable struct {
	DB   *sql.DB
	Silo *silo.Client
}

type storedFile struct {
	Path string
	Key  string
	Size int64
}

//go:embed catalog.sql
var durableDDL string

// Restore stages committed entries into a new, private worker directory.
func (d *Durable) Restore(ctx context.Context, root string) error {
	exists, err := d.Silo.SDK().BucketExists(ctx, d.Silo.Bucket())
	if err != nil {
		return fmt.Errorf("check Go1 Silo bucket: %w", err)
	}
	if !exists {
		return errors.New("create the configured Go1 Silo bucket before starting AIOps")
	}
	if err := d.prepareCatalog(ctx); err != nil {
		return fmt.Errorf("prepare Go1 catalog: %w", err)
	}
	return d.restoreCatalog(ctx, root)
}

func stagePath(root, path string) (string, error) {
	if path == "." || !fs.ValidPath(path) || strings.Contains(path, "\\") {
		return "", errors.New("invalid Go1 storage path")
	}
	target := filepath.Join(root, filepath.FromSlash(path))
	return target, os.MkdirAll(filepath.Dir(target), 0700)
}

func stageFile(root, path string, data []byte) error {
	target, err := stagePath(root, path)
	if err != nil {
		return err
	}
	return os.WriteFile(target, data, 0600)
}

func (d *Durable) restoreObject(ctx context.Context, root string, object storedFile) error {
	target, err := stagePath(root, object.Path)
	if err != nil {
		return err
	}
	reader, size, err := d.Silo.Get(ctx, object.Key)
	if err != nil {
		return err
	}
	defer reader.Close()
	if size != object.Size || size < 0 {
		return errors.New("Go1 Silo object size changed")
	}
	out, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	n, copyErr := io.Copy(out, io.LimitReader(reader, size+1))
	return errors.Join(copyErr, out.Close(), checkSize(n, size))
}

func checkSize(actual, expected int64) error {
	if actual != expected {
		return errors.New("Go1 Silo object size changed")
	}
	return nil
}

func (s *Service) storeRecord(path string, metadata []byte, artifactDir string) error {
	d := s.c.Durable
	if d == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	if artifactDir != "" {
		cancel()
		ctx, cancel = context.WithTimeout(context.Background(), 5*time.Minute)
	}
	defer cancel()
	var objects []storedFile
	if artifactDir != "" {
		prefix := "go1/artifacts/" + newID() + "/"
		err := filepath.WalkDir(filepath.Join(s.c.DataDir, artifactDir), func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() {
				return err
			}
			info, err := entry.Info()
			if err != nil {
				return err
			}
			if !info.Mode().IsRegular() {
				return errors.New("unexpected Go1 artifact type")
			}
			rel, err := filepath.Rel(s.c.DataDir, path)
			if err != nil {
				return err
			}
			rel = filepath.ToSlash(rel)
			if strings.HasSuffix(rel, "/job.json") || strings.HasSuffix(rel, "/job.json.tmp") {
				return nil
			}
			f, err := os.Open(path)
			if err != nil {
				return err
			}
			defer f.Close()
			key := prefix + rel
			n, err := d.Silo.PutAsset(ctx, key, f, info.Size(), "application/octet-stream")
			if err != nil {
				return err
			}
			if err := checkSize(n, info.Size()); err != nil {
				return err
			}
			objects = append(objects, storedFile{rel, key, n})
			return nil
		})
		if err != nil {
			return err
		}
	}
	tx, err := d.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	writer := catalogWriter{ctx: ctx, tx: tx, root: s.c.DataDir, d: d}
	if err := writer.write(path, metadata, objects); err != nil {
		return err
	}
	return tx.Commit()
}
