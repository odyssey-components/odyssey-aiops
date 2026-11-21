package models

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"github.com/pelletier/go-toml/v2"
	"net/http"
	"net/url"
	"odyssey/aiops/internal/files"
	"odyssey/aiops/internal/paging"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Service struct{ Root string }
type Entry struct {
	Run    string `json:"run"`
	Lesson string `json:"lesson"`
}

func JSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
func (s Service) artifact(run, kind string) ([]byte, error) {
	if !files.ID(run) {
		return nil, os.ErrNotExist
	}
	name, limit := "model.json", int64(1024*1024)
	switch kind {
	case "":
	case "manifest":
		name = "model.manifest.toml"
		limit = 65536
	case "onnx":
		name = "model.onnx"
	default:
		return nil, os.ErrInvalid
	}
	return files.Read(s.Root, filepath.Join(run, name), limit)
}
func (s Service) Manifest(run string) (map[string]any, error) {
	b, e := s.artifact(run, "manifest")
	if e != nil {
		return nil, e
	}
	var v map[string]any
	e = toml.Unmarshal(b, &v)
	return v, e
}
func (s Service) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		w.WriteHeader(405)
		return
	}
	run := strings.TrimPrefix(r.URL.Path, "/api/models")
	if run == "" || run == "/" {
		s.list(w, r.URL.Query())
		return
	}
	run = strings.TrimPrefix(run, "/")
	kind := r.URL.Query().Get("artifact")
	b, e := s.artifact(run, kind)
	if e != nil {
		status := 404
		if errors.Is(e, files.ErrLarge) {
			status = 413
		}
		if errors.Is(e, os.ErrInvalid) {
			status = 400
		}
		w.WriteHeader(status)
		return
	}
	if kind == "manifest" {
		var v map[string]any
		if toml.Unmarshal(b, &v) != nil {
			w.WriteHeader(422)
			return
		}
		JSON(w, 200, v)
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	if kind == "onnx" {
		w.Header().Set("Content-Type", "application/octet-stream")
	} else {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
	}
	_, _ = w.Write(b)
}
func (s Service) list(w http.ResponseWriter, query url.Values) {
	catalog := query.Get("catalog") == "1"
	paginated := catalog && query.Has("page")
	page := paging.Request{}
	if paginated {
		var err error
		page, err = paging.Parse(query)
		if err != nil {
			JSON(w, 400, map[string]string{"error": err.Error()})
			return
		}
	}
	entries, e := os.ReadDir(s.Root)
	if e != nil {
		if paginated {
			JSON(w, 200, map[string]any{"items": []Entry{}, "pagination": paging.Info{Page: 1, Size: page.Size}})
		} else if catalog {
			JSON(w, 200, []Entry{})
		} else {
			JSON(w, 200, []string{})
		}
		return
	}
	names := []string{}
	for _, entry := range entries {
		if entry.IsDir() && files.ID(entry.Name()) {
			names = append(names, entry.Name())
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(names)))
	result := []Entry{}
	legacy := []string{}
	for _, name := range names {
		if !catalog {
			if _, e = s.artifact(name, ""); e == nil {
				legacy = append(legacy, name)
			}
			continue
		}
		v, e := s.Manifest(name)
		if e != nil || !Valid(v) {
			continue
		}
		b, e := s.artifact(name, "onnx")
		if e != nil {
			continue
		}
		sum := sha256.Sum256(b)
		if hex.EncodeToString(sum[:]) != v["sha256"] {
			continue
		}
		if !paginated || query.Get("course") == "" || query.Get("course") == v["lesson"] {
			result = append(result, Entry{name, v["lesson"].(string)})
		}
	}
	if paginated {
		items, info := paging.Slice(result, page)
		JSON(w, 200, map[string]any{"items": items, "pagination": info})
	} else if catalog {
		JSON(w, 200, result)
	} else {
		JSON(w, 200, legacy)
	}
}
