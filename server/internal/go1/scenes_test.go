package go1

import (
	"encoding/json"
	"github.com/pelletier/go-toml/v2"
	"net/http/httptest"
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestScenesPersistAndValidateWithoutStartingTraining(t *testing.T) {
	executor := runner.New()
	defer executor.Close()
	root := t.TempDir()
	service, e := New(Config{DataDir: root, Python: "/missing/python", Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	body := `{"action":"create-scene","scene":{"title":"窄路练习","kind":"uturn","parameters":{"road_width_m":0.8,"turn_at_m":1.2}}}`
	var command struct {
		Scene json.RawMessage `json:"scene"`
	}
	if err := json.Unmarshal([]byte(body), &command); err != nil {
		t.Fatal(err)
	}
	scene, e := service.createScene(command.Scene)
	if e != nil {
		t.Fatal(e)
	}
	restarted, e := New(Config{DataDir: root, Python: "/missing/python", Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	loaded, e := restarted.loadScene(scene.ID)
	if e != nil || loaded.Title != "窄路练习" || loaded.Parameters["road_width_m"] != .8 {
		t.Fatal(loaded, e)
	}
	for _, invalid := range []string{strings.Replace(body, `0.8`, `0.1`, 1), strings.Replace(body, `"uturn"`, `"unknown"`, 1), strings.Replace(body, `"title":`, `"path":"/tmp/map","title":`, 1)} {
		response := httptest.NewRecorder()
		service.ServeHTTP(response, httptest.NewRequest("POST", "/api/go1", strings.NewReader(invalid)))
		if response.Code != 400 {
			t.Fatal(response.Code, response.Body.String())
		}
	}
	if _, e = service.loadScene("../../scene"); e == nil {
		t.Fatal("accepted unsafe scene ID")
	}
	if len(restarted.scenes()) != 1 {
		t.Fatal("invalid configurations persisted")
	}
}
func TestTaskSnapshotsSceneAndUsesVisionModule(t *testing.T) {
	executor := runner.New()
	defer executor.Close()
	root := t.TempDir()
	python := filepath.Join(root, "python")
	script := "#!/bin/sh\ncase \"$*\" in *--check*) echo '{\"ready\":true}';; *) echo \"$*\"; exit 1;; esac\n"
	if e := os.WriteFile(python, []byte(script), 0700); e != nil {
		t.Fatal(e)
	}
	service, e := New(Config{DataDir: root, Python: python, Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	scene, e := service.createScene(json.RawMessage(`{"title":"小坡","kind":"slope","parameters":{"angle_deg":8,"length_m":1.2}}`))
	if e != nil {
		t.Fatal(e)
	}
	payload, _ := json.Marshal(map[string]any{"action": "train", "title": "坡面任务", "scene": scene.ID, "config": map[string]float64{}})
	response := httptest.NewRecorder()
	service.ServeHTTP(response, httptest.NewRequest("POST", "/api/go1", strings.NewReader(string(payload))))
	if response.Code != 202 {
		t.Fatal(response.Code, response.Body.String())
	}
	var job Job
	_ = json.Unmarshal(response.Body.Bytes(), &job)
	if job.Scene == nil || job.Scene.ID != scene.ID {
		t.Fatal(job)
	}
	data, e := os.ReadFile(filepath.Join(root, "jobs", job.ID, "training.toml"))
	var config map[string]any
	if e != nil || toml.Unmarshal(data, &config) != nil || config["goal_kind"] != "slope" {
		t.Fatal(string(data), e)
	}
	_ = os.WriteFile(filepath.Join(root, "scenes", scene.ID+".toml"), []byte("broken"), 0600)
	for attempt := 0; attempt < 100; attempt++ {
		service.mu.Lock()
		stored := service.live[job.ID]
		finished := stored.State == "failed"
		same := stored.Scene != nil && stored.Scene.Parameters["angle_deg"] == 8
		logs := stored.Logs
		service.mu.Unlock()
		if finished {
			if !same || !strings.Contains(logs, "aiops_go1_training.vision.train") {
				t.Fatal(stored)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("fake visual worker did not finish")
}

func TestSceneDimensionsHaveNoFixedMaximum(t *testing.T) {
	for _, scene := range []Scene{
		{Title: "远距离转弯", Kind: "turn", Parameters: map[string]float64{"angle_deg": 90, "road_width_m": 100, "distance_m": 1000}},
		{Title: "远距离掉头", Kind: "uturn", Parameters: map[string]float64{"road_width_m": 100, "turn_at_m": 1000}},
		{Title: "长坡", Kind: "slope", Parameters: map[string]float64{"angle_deg": 8, "length_m": 1000}},
	} {
		if err := validateScene(scene); err != nil {
			t.Fatalf("%s: %v", scene.Kind, err)
		}
		job := &Job{Scene: &scene, Config: map[string]float64{}}
		if got := visionConfig(job)["target_distance_m"]; got != float64(1000) {
			t.Fatalf("distance changed: %v", got)
		}
	}
}

func TestBeamSceneConfiguration(t *testing.T) {
	for _, tc := range []struct {
		kind   string
		height float64
	}{{"duck", .28}, {"jump", .20}} {
		t.Run(tc.kind, func(t *testing.T) {
			scene := Scene{Title: "横梁", Kind: tc.kind, Parameters: map[string]float64{"distance_m": 1.2, "height_m": tc.height}, Criteria: defaultCriteria(tc.kind)}
			if err := validateScene(scene); err != nil {
				t.Fatal(err)
			}
			cfg := visionConfig(&Job{Scene: &scene})
			if cfg["goal_kind"] != tc.kind || cfg["beam_height_m"] != tc.height || cfg["target_distance_m"] != 1.2 {
				t.Fatalf("wrong beam configuration: %v", cfg)
			}
			scene.Parameters["height_m"] = .4
			if validateScene(scene) == nil {
				t.Fatal("accepted invalid beam height")
			}
		})
	}
}

func TestSceneSetPersistsDistinctScenes(t *testing.T) {
	executor := runner.New()
	defer executor.Close()
	root := t.TempDir()
	python := filepath.Join(root, "python")
	if e := os.WriteFile(python, []byte("#!/bin/sh\ncase \"$*\" in *--check*) echo '{\"ready\":true}';; *) exit 1;; esac\n"), 0700); e != nil {
		t.Fatal(e)
	}
	service, e := New(Config{DataDir: root, Python: python, Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	ids := []string{}
	for _, raw := range []string{
		`{"title":"小坡","kind":"slope","parameters":{"angle_deg":8,"length_m":1.2}}`,
		`{"title":"转弯","kind":"turn","parameters":{"angle_deg":90,"road_width_m":0.8,"distance_m":1.2}}`,
	} {
		scene, e := service.createScene(json.RawMessage(raw))
		if e != nil {
			t.Fatal(e)
		}
		ids = append(ids, scene.ID)
	}
	post := func(body map[string]any) *httptest.ResponseRecorder {
		data, _ := json.Marshal(body)
		response := httptest.NewRecorder()
		service.ServeHTTP(response, httptest.NewRequest("POST", "/api/go1", strings.NewReader(string(data))))
		return response
	}
	bad := post(map[string]any{"action": "create-scene-set", "sceneSet": map[string]any{"title": "重复", "scenes": []string{ids[0], ids[0]}}})
	if bad.Code != 400 {
		t.Fatal(bad.Code, bad.Body.String())
	}
	saved := post(map[string]any{"action": "create-scene-set", "sceneSet": map[string]any{"title": "场景组合", "scenes": ids}})
	var set SceneSet
	if saved.Code != 201 || json.Unmarshal(saved.Body.Bytes(), &set) != nil {
		t.Fatal(saved.Code, saved.Body.String())
	}
	if sets := service.sceneSets(); len(sets) != 1 || sets[0].ID != set.ID {
		t.Fatal(sets)
	}
	if len(set.Scenes) != 2 || set.Scenes[0] != ids[0] || set.Scenes[1] != ids[1] {
		t.Fatal(set.Scenes)
	}
}
