package go1

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"odyssey/aiops/internal/runner"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func simulationFixture(t *testing.T) (*Service, *runner.Executor, string, string) {
	t.Helper()
	root := testWorkspaceRoot(t)
	parent := filepath.Join(root, "tmp/go1-simulation")
	_ = os.MkdirAll(parent, 0700)
	base, e := os.MkdirTemp(parent, "service-")
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { os.RemoveAll(base) })
	source := filepath.Join(base, "source")
	modelID := "12345678-1234-1234-1234-123456789abc"
	model := filepath.Join(base, "runs", modelID)
	_ = os.MkdirAll(model, 0700)
	python := filepath.Join(root, ".venv/bin/python")
	code := `import json,numpy as np,sys
from pathlib import Path
from aiops_go1_training.locomotion.model_manifest import create_manifest
from aiops_go1_training.runtime.provenance import model_digest
source=Path(sys.argv[1]);p=Path(sys.argv[2]);(source/'unitree_go1').mkdir(parents=True)
joints=''.join(f'<body name="leg{i}" pos="0 0 -.1"><joint name="j{i}"/><geom size=".01"/></body>' for i in range(12))
motors=''.join(f'<motor joint="j{i}" ctrllimited="true" ctrlrange="-1 1"/>' for i in range(12))
(source/'unitree_go1/scene_torque.xml').write_text(f'<mujoco><option timestep=".002"/><worldbody><geom type="plane" size="5 5 .1"/><body name="trunk" pos="0 0 .3"><freejoint/><geom type="box" size=".1 .05 .05"/>{joints}</body></worldbody><actuator>{motors}</actuator></mujoco>')
shapes={'l1/weight':(45,64),'l1/bias':(64,),'l2/weight':(64,64),'l2/bias':(64,),'actor/weight':(64,12),'actor/bias':(12,),'critic/weight':(64,1),'critic/bias':(1,),'log_std':(12,)}
np.savez_compressed(p/'policy.npz',**{k:np.zeros(s,dtype=np.float32) for k,s in shapes.items()})
e=dict(source_commit='f4fb0d12ac68842c1567bf1de6f771ce8e6671b5',model_sha256=model_digest(source/'unitree_go1'),config={},jax='synthetic',mujoco='synthetic',device='synthetic')
(p/'model-manifest.json').write_text(json.dumps(create_manifest(p/'policy.npz',e,{})))`
	if output, e := exec.Command(python, "-c", code, source, model).CombinedOutput(); e != nil {
		t.Fatalf("fixture: %v %s", e, output)
	}
	shim := filepath.Join(base, "synthetic-python")
	script := "#!" + python + "\n" + `import sys
from pathlib import Path
from aiops_go1_training.runtime.simulation import run_regression, run_simulation
args=sys.argv
def arg(name):return Path(args[args.index(name)+1])
if '--inject-vision' in args:
    run_simulation(arg('--source'),arg('--model'),arg('--output'),arg('--config'),arg('--recording') if '--recording' in args else None,inject_cases='--recording' not in args)
elif '--recording' in args:
    run_regression(arg('--source'),arg('--model'),arg('--output'),arg('--recording'))
else:
    run_simulation(arg('--source'),arg('--model'),arg('--output'),arg('--config'))
`
	if e = os.WriteFile(shim, []byte(script), 0700); e != nil {
		t.Fatal(e)
	}
	executor := runner.New()
	t.Cleanup(executor.Close)
	s, e := New(Config{DataDir: base, Python: shim, Source: source, Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	scene, e := s.createScene(json.RawMessage(`{"title":"转弯场景","kind":"turn","parameters":{"angle_deg":90,"road_width_m":0.8,"distance_m":1.2}}`))
	if e != nil {
		t.Fatal(e)
	}
	return s, executor, modelID, scene.ID
}
func simulationPost(s *Service, body string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest("POST", "/api/go1", strings.NewReader(body)))
	return w
}
func waitSimulation(t *testing.T, s *Service, id string) *Simulation {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for {
		s.mu.Lock()
		j := s.simulations[id]
		b, _ := json.Marshal(j)
		s.mu.Unlock()
		var copy Simulation
		_ = json.Unmarshal(b, &copy)
		if !active(copy.State) {
			return &copy
		}
		if time.Now().After(deadline) {
			t.Fatal("simulation timeout")
		}
		time.Sleep(10 * time.Millisecond)
	}
}
func TestSimulationCPUAndReadOnlyModel(t *testing.T) {
	s, _, model, scene := simulationFixture(t)
	original, e := os.ReadFile(filepath.Join(s.c.DataDir, "runs", model, "policy.npz"))
	if e != nil {
		t.Fatal(e)
	}
	w := simulationPost(s, fmt.Sprintf(`{"action":"simulate","title":"CPU仿真","model":"%s","scene":"%s","seed":7,"steps":50}`, model, scene))
	if w.Code != 202 {
		t.Fatal(w.Code, w.Body.String())
	}
	var created Simulation
	_ = json.Unmarshal(w.Body.Bytes(), &created)
	result := waitSimulation(t, s, created.ID)
	if result.State != "succeeded" {
		t.Fatal(result.State, result.Error, result.Logs)
	}
	if result.Result["backend"] != "mujoco-cpu" || result.Result["trained"] != false || result.Progress < 1 {
		t.Fatal(result)
	}
	after, _ := os.ReadFile(filepath.Join(s.c.DataDir, "runs", model, "policy.npz"))
	if string(original) != string(after) {
		t.Fatal("simulation changed policy")
	}
	request := httptest.NewRequest("GET", "/api/go1?simulation-replay="+created.ID, nil)
	replay := httptest.NewRecorder()
	s.ServeHTTP(replay, request)
	if replay.Code != 200 {
		t.Fatal(replay.Code, replay.Body.String())
	}
	saveSet := func(id string) *httptest.ResponseRecorder {
		return simulationPost(s, fmt.Sprintf(`{"action":"create-test-case-set","testCaseSet":{"title":"回归用例","cases":["%s"]}}`, id))
	}
	if response := saveSet(scene); response.Code != 400 {
		t.Fatal("scene cannot be used as a test case", response.Code, response.Body.String())
	}
	if response := saveSet(created.ID); response.Code != 201 {
		t.Fatal("could not save completed test case", response.Code, response.Body.String())
	}
	restarted, e := New(s.c)
	if e != nil {
		t.Fatal(e)
	}
	if sets := restarted.testCaseSets(); len(sets) != 1 || len(sets[0].Cases) != 1 || sets[0].Cases[0] != created.ID {
		t.Fatal("test case set did not persist", sets)
	}
	replay = httptest.NewRecorder()
	restarted.ServeHTTP(replay, request)
	if replay.Code != 200 {
		t.Fatal("saved replay unavailable", replay.Body.String())
	}
	if simulationPost(s, `{"action":"cancel-simulation","id":"`+created.ID+`"}`).Code != 409 {
		t.Fatal("completed simulation cancelled")
	}
}
func TestSimulationQueuedCancelAndModelMutation(t *testing.T) {
	for _, cancel := range []bool{true, false} {
		s, executor, model, scene := simulationFixture(t)
		if e := executor.Submit(runner.Task{ID: "block", Module: "maze", Command: "/bin/sh", Args: []string{"-c", "sleep 2"}, Timeout: 3 * time.Second}); e != nil {
			t.Fatal(e)
		}
		j, e := s.simulate(SimulationCommand{Action: "simulate", Title: "queued", Model: model, Scene: scene, Seed: 7, Steps: 50})
		if e != nil {
			t.Fatal(e)
		}
		if cancel {
			w := simulationPost(s, `{"action":"cancel-simulation","id":"`+j.ID+`"}`)
			if w.Code != 202 {
				t.Fatal(w.Code, w.Body.String())
			}
		} else {
			path := filepath.Join(s.c.DataDir, "simulations", j.ID, "model", "policy.npz")
			_ = os.WriteFile(path, []byte("corrupt"), 0600)
		}
		executor.Cancel("block")
		done := waitSimulation(t, s, j.ID)
		if cancel && done.State != "cancelled" || !cancel && done.State != "failed" {
			t.Fatal(done.State, done.Error)
		}
		if _, e = os.Stat(filepath.Join(s.c.DataDir, "simulations", j.ID, "output")); !os.IsNotExist(e) {
			t.Fatal("queued cancelled/mutated task executed")
		}
	}
}
func TestSimulationStrictFields(t *testing.T) {
	for _, body := range []string{`{"action":"simulate","model":"../escape"}`, `{"action":"simulate","command":"x"}`, `{"action":"simulate","model":"12345678-1234-1234-1234-123456789abc","scene":"12345678-1234-1234-1234-123456789abc","title":"x","seed":null,"steps":50}`} {
		var raw map[string]json.RawMessage
		_ = json.Unmarshal([]byte(body), &raw)
		if _, e := parseSimulation(raw); e == nil {
			t.Fatal("invalid simulation accepted", body)
		}
	}
}

func TestVisualSimulationUsesRealDepthAndCPUInference(t *testing.T) {
	s, _, motionModel, scene := simulationFixture(t)
	script, e := os.ReadFile(s.c.Python)
	if e != nil {
		t.Fatal(e)
	}
	python := strings.TrimPrefix(strings.SplitN(string(script), "\n", 2)[0], "#!")
	id := "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
	directory := filepath.Join(s.c.DataDir, "runs", id)
	_ = os.Mkdir(directory, 0700)
	code := `import jax,sys
from pathlib import Path
from aiops_go1_training.vision.policy import initialize
from aiops_go1_training.vision.configuration import VisionConfig
from aiops_go1_training.vision.artifacts import save_policy
from aiops_go1_training.runtime.provenance import model_digest
p=Path(sys.argv[1]);source=Path(sys.argv[2]);c=VisionConfig(episode_steps=50)
save_policy(p,initialize(jax.random.PRNGKey(0),c.height,c.width),c,model_digest(source/'unitree_go1'))`
	command := exec.Command(python, "-c", code, directory, s.c.Source)
	command.Env = append(os.Environ(), "JAX_PLATFORMS=cpu", "MUJOCO_GL=egl")
	if output, e := command.CombinedOutput(); e != nil {
		t.Fatalf("vision fixture: %v %s", e, output)
	}
	created, e := s.simulate(SimulationCommand{Action: "simulate", Title: "视觉CPU仿真", Model: id, Scene: scene, Seed: 7, Steps: 50})
	if e != nil {
		t.Fatal(e)
	}
	result := waitSimulation(t, s, created.ID)
	if result.State != "succeeded" {
		t.Fatal(result.State, result.Error, result.Logs)
	}
	if result.Result["steps"].(float64) < 1 || result.Result["trained"] != false {
		t.Fatal(result.Result)
	}
	if !result.Recorded {
		t.Fatal("visual simulation did not retain MCAP")
	}
	w := simulationPost(s, fmt.Sprintf(`{"action":"regress","title":"视觉回灌","model":"%s","recording":"%s"}`, id, created.ID))
	if w.Code != 202 {
		t.Fatal(w.Code, w.Body.String())
	}
	var regression Simulation
	if e := json.Unmarshal(w.Body.Bytes(), &regression); e != nil {
		t.Fatal(e)
	}
	compared := waitSimulation(t, s, regression.ID)
	if compared.State != "succeeded" || compared.Result["max_abs_delta"] != float64(0) ||
		compared.Result["steps"] != result.Result["steps"] {
		t.Fatal(compared.State, compared.Result, compared.Logs)
	}
	report := httptest.NewRecorder()
	s.ServeHTTP(report, httptest.NewRequest("GET", "/api/go1?regression-report="+regression.ID, nil))
	if report.Code != 200 || !strings.Contains(report.Body.String(), `"candidate_action"`) {
		t.Fatal(report.Code, report.Body.String())
	}

	if !strings.Contains(report.Body.String(), `"candidate_state"`) {
		t.Fatal("regression report omitted candidate input state")
	}
	depth := httptest.NewRecorder()
	s.ServeHTTP(depth, httptest.NewRequest("GET", "/api/go1?regression-report="+regression.ID+"&step=1", nil))
	var input struct {
		Width  int
		Height int
		F32LE  string
	}
	if depth.Code != 200 || json.Unmarshal(depth.Body.Bytes(), &input) != nil ||
		input.Width != 64 || input.Height != 48 || len(input.F32LE) != 16384 {
		t.Fatal("regression depth input missing or malformed", depth.Code, input.Width, input.Height, len(input.F32LE))
	}
	badStep := httptest.NewRecorder()
	s.ServeHTTP(badStep, httptest.NewRequest("GET", "/api/go1?regression-report="+regression.ID+"&step=501", nil))
	if badStep.Code != 400 {
		t.Fatal("accepted out-of-range regression step", badStep.Code)
	}

	frame := httptest.NewRecorder()
	s.ServeHTTP(frame, httptest.NewRequest("GET", "/api/go1?recording-frame="+created.ID+"&step=1", nil))
	if frame.Code != 200 || !strings.Contains(frame.Body.String(), `"width":64`) ||
		!strings.Contains(frame.Body.String(), `"depth_f32le_base64"`) {
		t.Fatal("MCAP frame inspection failed", frame.Code)
	}
	metrics := httptest.NewRecorder()
	s.ServeHTTP(metrics, httptest.NewRequest("GET", "/api/go1?recording-frame="+created.ID+"&step=metrics", nil))
	var recordingSummary map[string]float64
	if metrics.Code != 200 || json.Unmarshal(metrics.Body.Bytes(), &recordingSummary) != nil ||
		recordingSummary["frames"] != result.Result["steps"] || recordingSummary["observedTopics"] != 1 ||
		recordingSummary["maxGapMs"] != 20 {
		t.Fatal("MCAP metrics invalid", metrics.Code, metrics.Body.String())
	}
	depthStream := httptest.NewRecorder()
	s.ServeHTTP(depthStream, httptest.NewRequest("GET", "/api/go1?recorded-depth="+created.ID, nil))
	if depthStream.Code != 200 || depthStream.Body.Len() != 8+64*48*4*int(result.Result["steps"].(float64)) {
		t.Fatal("recorded depth stream invalid", depthStream.Code, depthStream.Body.Len())
	}
	injection := simulationPost(s, fmt.Sprintf(`{"action":"inject-vision","title":"视觉闭环回灌","model":"%s","recording":"%s"}`, id, created.ID))
	if injection.Code != 202 {
		t.Fatal(injection.Code, injection.Body.String())
	}
	var injected Simulation
	if e := json.Unmarshal(injection.Body.Bytes(), &injected); e != nil {
		t.Fatal(e)
	}
	finished := waitSimulation(t, s, injected.ID)
	if finished.State != "succeeded" || !finished.VisionInjected || finished.Result["steps"] != result.Result["steps"] {
		artifact, _ := os.ReadFile(filepath.Join(s.c.DataDir, "simulations", injected.ID, "output", "result.json"))
		var fields map[string]any
		_ = json.Unmarshal(artifact, &fields)
		t.Fatal(finished.State, finished.Error, finished.Logs, fields["format"], fields["seed"], fields["goal_kind"])
	}
	baselineReplay, err := os.ReadFile(filepath.Join(s.c.DataDir, "simulations", created.ID, "output", "replay.json"))
	if err != nil {
		t.Fatal(err)
	}
	injectedReplay, err := os.ReadFile(filepath.Join(s.c.DataDir, "simulations", injected.ID, "output", "replay.json"))
	if err != nil || string(baselineReplay) != string(injectedReplay) {
		t.Fatal("same policy and injected observations should reproduce the physics trace", err)
	}
	vision := httptest.NewRecorder()
	s.ServeHTTP(vision, httptest.NewRequest("GET", "/api/go1?injected-vision="+injected.ID, nil))
	if vision.Code != 200 || vision.Body.Len() != 8+64*48*4*int(finished.Result["steps"].(float64)) {
		t.Fatal("injected vision playback payload mismatch", vision.Code, vision.Body.Len())
	}

	setResponse := simulationPost(s, fmt.Sprintf(`{"action":"create-test-case-set","testCaseSet":{"title":"两个回灌用例","cases":["%s","%s"]}}`, created.ID, injected.ID))
	var testSet TestCaseSet
	if setResponse.Code != 201 || json.Unmarshal(setResponse.Body.Bytes(), &testSet) != nil {
		t.Fatal(setResponse.Code, setResponse.Body.String())
	}
	before := len(s.simulationHistory())
	batchResponse := simulationPost(s, fmt.Sprintf(`{"action":"inject-vision","title":"集合回灌测试","model":"%s","testCaseSet":"%s"}`, id, testSet.ID))
	var batch Simulation
	if batchResponse.Code != 202 || json.Unmarshal(batchResponse.Body.Bytes(), &batch) != nil {
		t.Fatal(batchResponse.Code, batchResponse.Body.String())
	}
	batchDone := waitSimulation(t, s, batch.ID)
	if batchDone.State != "succeeded" || batchDone.Progress != 2 || len(s.simulationHistory()) != before+1 {
		t.Fatal(batchDone.State, batchDone.Error, batchDone.Logs)
	}
	for _, caseID := range testSet.Cases {
		for _, query := range []string{"simulation-replay", "injected-vision", "recording-frame", "recorded-depth"} {
			path := "/api/go1?" + query + "=" + batch.ID + "&case=" + caseID
			if query == "recording-frame" {
				path += "&step=1"
			}
			response := httptest.NewRecorder()
			s.ServeHTTP(response, httptest.NewRequest("GET", path, nil))
			if response.Code != 200 {
				t.Fatal(query, response.Code, response.Body.String())
			}
		}
	}

	motion, e := s.simulate(SimulationCommand{Action: "simulate", Title: "运动模型深度录制", Model: motionModel, Scene: scene, Seed: 7, Steps: 50})
	if e != nil {
		t.Fatal(e)
	}
	recorded := waitSimulation(t, s, motion.ID)
	if recorded.State != "succeeded" || !recorded.Recorded {
		t.Fatal(recorded.State, recorded.Error, recorded.Logs)
	}
	for _, candidate := range []string{motionModel, id} {
		task, err := s.simulate(SimulationCommand{Action: "regress", Title: "运动录制回灌", Model: candidate, Recording: motion.ID})
		if err != nil {
			t.Fatal(err)
		}
		result := waitSimulation(t, s, task.ID)
		if result.State != "succeeded" || result.Result["steps"] != recorded.Result["steps"] {
			t.Fatal(result.State, result.Error, result.Logs)
		}
		if candidate == motionModel && result.Result["max_abs_delta"] != float64(0) {
			t.Fatal("motion regression changed when depth was present", result.Result)
		}
	}

}
