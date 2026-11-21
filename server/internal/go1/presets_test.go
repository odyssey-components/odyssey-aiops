package go1

import (
	"encoding/json"
	"net/http/httptest"
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestFixedScenesAvailableWithoutParametersAndSurviveRestart(t *testing.T) {
	executor := runner.New()
	defer executor.Close()
	cfg := Config{DataDir: t.TempDir(), Python: "/missing/python", Executor: executor}
	service, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	read := func(s *Service) []Scene {
		response := httptest.NewRecorder()
		s.ServeHTTP(response, httptest.NewRequest("GET", "/api/go1", nil))
		var snapshot struct {
			Scenes []Scene `json:"scenes"`
		}
		if response.Code != 200 || json.Unmarshal(response.Body.Bytes(), &snapshot) != nil {
			t.Fatal(response.Body.String())
		}
		return snapshot.Scenes
	}
	scenes := read(service)
	if len(scenes) != 6 {
		t.Fatalf("expected six fixed scenes, got %d", len(scenes))
	}
	expected := map[string]map[string]float64{
		"flat":  {"forward_m_s": .3, "yaw_rate_rad_s": 0},
		"turn":  {"angle_deg": 90, "road_width_m": .8, "distance_m": 1.2},
		"uturn": {"road_width_m": .8, "turn_at_m": 1.2},
		"slope": {"angle_deg": 8, "length_m": 1.2},
		"duck":  {"distance_m": 1.2, "height_m": .28},
		"jump":  {"distance_m": 1.2, "height_m": .2},
	}
	kinds := map[string]bool{}
	for _, scene := range scenes {
		if kinds[scene.Kind] {
			t.Fatal("duplicate kind", scene.Kind)
		}
		kinds[scene.Kind] = true
		if !reflect.DeepEqual(scene.Parameters, expected[scene.Kind]) {
			t.Fatal("preset parameters changed", scene)
		}
		if scene.Kind != "flat" && scene.Criteria["stable_seconds"] != 1 {
			t.Fatal("preset stability changed", scene)
		}
		if err := validateScene(scene); err != nil {
			t.Fatal(err)
		}
		loaded, err := service.loadScene(scene.ID)
		if err != nil || !reflect.DeepEqual(scene, loaded) {
			t.Fatal(loaded, err)
		}
	}
	legacy, err := service.createScene(json.RawMessage(`{"title":"历史自定义","kind":"turn","parameters":{"angle_deg":120,"road_width_m":2,"distance_m":4}}`))
	if err != nil {
		t.Fatal(err)
	}
	restarted, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scenes, read(restarted)) {
		t.Fatal("preset identity changed or custom scene leaked into choices")
	}
	if loaded, err := restarted.loadScene(legacy.ID); err != nil || loaded.Parameters["angle_deg"] != 120 {
		t.Fatal("legacy scene changed", err)
	}
}

func TestFixedScenesDoNotReadOrWriteStorage(t *testing.T) {
	executor := runner.New()
	defer executor.Close()
	root := t.TempDir()
	cfg := Config{DataDir: root, Python: "/missing/python", Executor: executor, Durable: &Durable{}}
	service, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(filepath.Join(root, "scenes"))
	if err != nil || len(entries) != 0 {
		t.Fatal("startup persisted fixed scenes", entries, err)
	}
	response := httptest.NewRecorder()
	service.ServeHTTP(response, httptest.NewRequest("POST", "/api/go1", strings.NewReader(`{"action":"create-scene","scene":{"title":"custom","kind":"flat","parameters":{"forward_m_s":0.3,"yaw_rate_rad_s":0}}}`)))
	if response.Code != 400 || !strings.Contains(response.Body.String(), "固定预设") {
		t.Fatal(response.Code, response.Body.String())
	}
	// A broken persisted copy must have no influence on code-defined scenes.
	scene := service.fixedScenes()[0]
	if err := os.WriteFile(filepath.Join(root, "scenes", scene.ID+".toml"), []byte("broken"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Join(root, "scenes")); err != nil {
		t.Fatal(err)
	}
	// No database/Silo handles and no data directory: lookup must still work.
	service.c = Config{DataDir: filepath.Join(root, "missing"), Durable: &Durable{}}
	if got := service.fixedScenes(); len(got) != 6 {
		t.Fatal(got)
	}
	loaded, err := service.loadScene(scene.ID)
	if err != nil || !reflect.DeepEqual(loaded, scene) {
		t.Fatal(loaded, err)
	}
	loaded.Parameters["forward_m_s"] = 99
	again, err := service.loadScene(scene.ID)
	if err != nil || again.Parameters["forward_m_s"] != .3 {
		t.Fatal("caller mutated fixed definitions", again, err)
	}
}
