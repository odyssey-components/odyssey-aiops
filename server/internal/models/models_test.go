package models

import (
	"crypto/sha256"
	"fmt"
	"github.com/pelletier/go-toml/v2"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(lesson string) map[string]any {
	version, width, encoding, steps := 1, 18, "one-hot-xy-direction/v1", 100
	switch lesson {
	case "random-map-v1":
		version, width, encoding = 2, 151, "grid-walls-goal-agent-direction/v1"
	case "local-map-v1":
		version, width, encoding, steps = 3, 249, "grid-known-walls-goal-visited-agent-direction/v1", 200
	case "memory-map-v1":
		version, width, encoding, steps = 4, 40, "local-view-xy-direction-previous-action/v1", 200
	}
	tensor := func(n string, w int) map[string]any {
		return map[string]any{"name": n, "dtype": "float32", "shape": map[string]any{"batch": "batch", "width": w}}
	}
	m := map[string]any{"schema": "odymaze-onnx/v1", "filename": "model.onnx", "sha256": fmt.Sprintf("%x", sha256.Sum256([]byte("onnx"))), "opset": 17, "lesson": lesson, "policy_format": fmt.Sprintf("odymaze-policy/v%d", version), "encoding": encoding, "actions": []string{"left", "right", "forward"}, "outputs_are_logits": true, "inputs": []any{tensor("observations", width)}, "outputs": []any{tensor("logits", 3)}, "metadata": map[string]any{"seed": 42, "epochs": 1, "samples": 10, "accuracy": .5, "success": false}}
	if version == 1 {
		m["map_ref"] = map[string]any{"id": "odymaze-first-steps-v1"}
	} else {
		m["generator"] = "lcg-corridor-7/v1"
		m["maze"] = generated(42, steps)
	}
	if version == 4 {
		m["inputs"] = append(m["inputs"].([]any), tensor("hidden", 32))
		m["outputs"] = append(m["outputs"].([]any), tensor("next_hidden", 32))
		m["recurrent"] = map[string]any{"reset": "per-maze", "update": "once-per-action", "initial_hidden": make([]int, 32)}
	}
	return m
}
func TestCatalog(t *testing.T) {
	root := t.TempDir()
	for _, lesson := range []string{"fixed-map-v1", "random-map-v1", "local-map-v1", "memory-map-v1"} {
		m := fixture(lesson)
		b, e := toml.Marshal(m)
		if e != nil {
			t.Fatal(e)
		}
		dir := filepath.Join(root, lesson)
		_ = os.Mkdir(dir, 0700)
		_ = os.WriteFile(filepath.Join(dir, "model.manifest.toml"), b, 0600)
		_ = os.WriteFile(filepath.Join(dir, "model.onnx"), []byte("onnx"), 0600)
		parsed, e := (Service{root}).Manifest(lesson)
		if e != nil || !Valid(parsed) {
			t.Fatalf("invalid %s %v %#v", lesson, e, parsed)
		}
	}
	w := httptest.NewRecorder()
	Service{root}.ServeHTTP(w, httptest.NewRequest("GET", "/api/models?catalog=1", nil))
	if w.Code != 200 || strings.Count(w.Body.String(), `"run"`) != 4 {
		t.Fatal(w.Body.String())
	}
	_ = os.WriteFile(filepath.Join(root, "fixed-map-v1", "model.onnx"), []byte("bad"), 0600)
	w = httptest.NewRecorder()
	Service{root}.ServeHTTP(w, httptest.NewRequest("GET", "/api/models?catalog=1", nil))
	if strings.Count(w.Body.String(), `"run"`) != 3 {
		t.Fatal(w.Body.String())
	}
}
func TestMalformedContracts(t *testing.T) {
	for _, key := range []string{"schema", "sha256", "encoding", "inputs", "metadata", "map_ref"} {
		m := fixture("fixed-map-v1")
		m[key] = "bad"
		if Valid(m) {
			t.Fatal("accepted", key)
		}
	}
}
