package go1

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/pelletier/go-toml/v2"
	"odyssey/aiops/internal/files"
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func (s *Service) simulate(c SimulationCommand) (Simulation, error) {
	var scene Scene
	var set *SceneSet
	var testSet *TestCaseSet
	var caseInputs []SimulationCaseInput
	var caseRecordings [][]byte
	var cases []Scene
	var recording []byte
	var e error
	if c.TestCaseSet != "" {
		for _, available := range s.testCaseSets() {
			if available.ID == c.TestCaseSet {
				testSet = &available
				break
			}
		}
		if testSet == nil {
			return Simulation{}, errors.New("未找到测试用例集")
		}
		totalBytes := 0
		for _, id := range testSet.Cases {
			member, seed, steps, data, err := s.regressionInput(id)
			if err != nil {
				return Simulation{}, fmt.Errorf("测试用例 %s：%w", id, err)
			}
			totalBytes += len(data)
			if totalBytes > 128<<20 {
				return Simulation{}, errors.New("测试用例集录制总大小超过 128 MiB")
			}
			member.ID = id
			cases = append(cases, member)
			caseInputs = append(caseInputs, SimulationCaseInput{Seed: seed, Steps: steps})
			caseRecordings = append(caseRecordings, data)
		}
		scene, c.Seed, c.Steps = cases[0], caseInputs[0].Seed, caseInputs[0].Steps
	} else if c.SceneSet != "" {
		group, err := s.loadSceneSet(c.SceneSet)
		if err != nil {
			return Simulation{}, err
		}
		for _, id := range group.Scenes {
			member, err := s.loadScene(id)
			if err != nil {
				return Simulation{}, err
			}
			cases = append(cases, member)
		}
		set, scene = &group, cases[0]
	} else if c.Recording == "" {
		scene, e = s.loadScene(c.Scene)
		if e != nil {
			return Simulation{}, e
		}
	} else {
		scene, c.Seed, c.Steps, recording, e = s.regressionInput(c.Recording)
		if e != nil {
			return Simulation{}, e
		}
	}
	modelRoot := filepath.Join(s.c.DataDir, "runs", c.Model)
	model, e := manifest(filepath.Dir(modelRoot), filepath.Base(modelRoot))
	if e != nil {
		if path, err := s.baseLocation(c.Model); err == nil {
			modelRoot = path
			model, e = upstreamManifest(filepath.Dir(path), filepath.Base(path))
		}
	}
	if e != nil && validID.MatchString(c.Model) {
		root := s.publicPolicyRoot()
		if imported, err := publicPolicyManifest(root, c.Model); err == nil {
			modelRoot, model, e = filepath.Join(root, c.Model), imported, nil
		}
	}
	if e != nil {
		return Simulation{}, e
	}
	parkour := model["format"] == parkourPolicyFormat
	if isPublicPolicy(model) && c.Action != "simulate" && !(parkour && c.Action == "inject-vision") {
		return Simulation{}, errors.New("公开位置控制策略目前仅支持闭环仿真")
	}
	if parkour {
		for _, member := range append([]Scene{scene}, cases...) {
			if member.Kind != "flat" && member.Kind != "duck" && member.Kind != "jump" {
				return Simulation{}, errors.New("Robot Parkour Learning 仅支持空旷场地、低横杆和跳跃场景")
			}
		}
	}
	if c.Action == "inject-vision" && !parkour && model["format"] != "go1-depth/v1" && model["format"] != "go1-depth-residual/v2" {
		return Simulation{}, errors.New("仿真回灌需要视觉模型")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.simulations == nil {
		s.simulations = map[string]*Simulation{}
	}
	count := 0
	for _, j := range s.simulations {
		if active(j.State) {
			count++
		}
	}
	if count >= 16 {
		return Simulation{}, errors.New("仿真排队任务过多")
	}
	j := &Simulation{Format: "go1-simulation/v1", ID: newID(), Title: c.Title, Recording: c.Recording, VisionInjected: c.Action == "inject-vision", State: "queued", StartedAt: time.Now().UTC().Format(time.RFC3339Nano), Model: c.Model, ModelManifest: model, ModelFiles: map[string]string{}, Scene: scene, SceneSet: set, TestCaseSet: testSet, CaseInputs: caseInputs, Cases: cases, Seed: c.Seed, Steps: c.Steps}
	work := filepath.Join(s.c.DataDir, "simulations", j.ID)
	if e = os.Mkdir(work, 0700); e != nil {
		return Simulation{}, e
	}
	if e = os.Mkdir(filepath.Join(work, "model"), 0700); e != nil {
		return Simulation{}, e
	}
	names := []string{"policy.npz", "model-manifest.json"}
	snapshotLimit := int64(4 << 20)
	if isPublicPolicy(model) {
		names = []string{"policy.npz", "public-policy.json"}
		snapshotLimit = publicPolicySize
	}
	if model["format"] == "go1-depth/v1" || model["format"] == "go1-depth-residual/v2" {
		names = []string{"policy.npz", "config.toml", "model.manifest.toml"}
	}
	if model["format"] == "go1-upstream-policy/v1" {
		names = []string{"policy.npz", "model.manifest.toml"}
		if model["config_sha256"] != nil {
			names = append(names, "config.toml")
		}
	}
	if model["scene_id"] != nil {
		names = append(names, "scene.toml")
	}
	if model["format"] == "go1-depth-residual/v2" {
		names = append(names, "base/policy.npz", "base/model.manifest.toml")
		if _, e := os.Stat(filepath.Join(modelRoot, "base/config.toml")); e == nil {
			names = append(names, "base/config.toml")
		}
	}
	for _, name := range names {
		b, err := files.Read(modelRoot, name, snapshotLimit)
		if err != nil {
			return Simulation{}, err
		}
		if e = os.MkdirAll(filepath.Dir(filepath.Join(work, "model", name)), 0700); e != nil {
			return Simulation{}, e
		}
		if e = os.WriteFile(filepath.Join(work, "model", name), b, 0600); e != nil {
			return Simulation{}, e
		}
		j.ModelFiles[name] = hashBytes(b)
	}
	if e = s.verifySimulationModel(j); e != nil {
		return Simulation{}, e
	}
	j.ModelManifest, e = simulationModelManifest(work, j)
	if e != nil {
		return Simulation{}, e
	}
	recordingHash := ""
	if recording != nil {
		recordingHash = hashBytes(recording)
		if e = os.WriteFile(filepath.Join(work, "policy-input.mcap"), recording, 0600); e != nil {
			return Simulation{}, e
		}
	}
	cfg := simulationConfig(c, scene)
	limit := int64(16384)
	timeout := 5 * time.Minute
	caseHashes := make([]string, len(caseRecordings))
	for index, data := range caseRecordings {
		caseHashes[index] = hashBytes(data)
		if e = os.WriteFile(filepath.Join(work, fmt.Sprintf("input-case-%d.mcap", index+1)), data, 0600); e != nil {
			return Simulation{}, e
		}
	}
	if len(cases) > 0 {
		configs := make([]map[string]any, 0, len(cases))
		for index, member := range cases {
			command := c
			if testSet != nil {
				command.Seed, command.Steps = caseInputs[index].Seed, caseInputs[index].Steps
			}
			configs = append(configs, simulationConfig(command, member))
		}
		cfg = map[string]any{"cases": configs}
		limit *= sceneSetLimit
		timeout *= time.Duration(len(cases))
	}
	b, e := toml.Marshal(cfg)
	if e != nil {
		return Simulation{}, e
	}
	if e = os.WriteFile(filepath.Join(work, "simulation.toml"), b, 0600); e != nil {
		return Simulation{}, e
	}
	j.ConfigSHA = hashBytes(b)
	s.simulations[j.ID] = j
	if e = s.saveSimulation(j); e != nil {
		delete(s.simulations, j.ID)
		return Simulation{}, e
	}
	env := []string{}
	for _, value := range s.env() {
		key, _, _ := strings.Cut(value, "=")
		if key != "JAX_PLATFORMS" && key != "MUJOCO_GL" && !(isPublicPolicy(model) && key == "OPENBLAS_NUM_THREADS") {
			env = append(env, value)
		}
	}
	env = append(env, "JAX_PLATFORMS=cpu", "MUJOCO_GL=egl")
	if isPublicPolicy(model) {
		env = append(env, "OPENBLAS_NUM_THREADS=1")
	}
	partial := ""
	output := func(text, stream string) {
		s.mu.Lock()
		defer s.mu.Unlock()
		j.Logs += text
		if len(j.Logs) > 24000 {
			j.Logs = strings.ToValidUTF8(j.Logs[len(j.Logs)-24000:], "")
		}
		if stream == "stdout" {
			partial += text
			if len(partial) > 65536 {
				partial = partial[len(partial)-65536:]
			}
			lines := strings.Split(partial, "\n")
			partial = lines[len(lines)-1]
			for _, line := range lines[:len(lines)-1] {
				var event struct {
					Event       string
					Step, Total int
				}
				if json.Unmarshal([]byte(line), &event) == nil && event.Event == "simulation.progress" && event.Total == j.Steps && event.Step >= j.Progress && event.Step <= j.Steps {
					if len(cases) == 0 {
						j.Progress = event.Step
					}
				} else if len(cases) > 0 && event.Event == "simulation.case-complete" && event.Total == len(cases) && event.Step == j.Progress+1 && event.Step <= len(cases) {
					j.Progress = event.Step
				}
			}
		}
		_ = s.saveSimulation(j)
	}
	task := runner.Task{ID: j.ID, Module: "go1-simulation", Command: s.c.Python, Args: []string{"-u", "-m", "aiops_go1_training.runtime.worker", "--parent-pid", fmt.Sprint(os.Getpid()), "--", "aiops_go1_training.runtime.simulation", "--source", s.c.Source, "--model", filepath.Join(work, "model"), "--config", filepath.Join(work, "simulation.toml"), "--output", filepath.Join(work, "output")}, Dir: work, Env: env, Timeout: timeout, OnOutput: output,
		OnStart: func() error {
			s.mu.Lock()
			defer s.mu.Unlock()
			if j.State == "cancelling" {
				return errors.New("仿真已取消")
			}
			configBytes, err := files.Read(s.c.DataDir, filepath.Join("simulations", j.ID, "simulation.toml"), limit)
			if err != nil || hashBytes(configBytes) != j.ConfigSHA {
				return errors.New("排队后的仿真配置已变化")
			}
			if e := s.verifySimulationModel(j); e != nil {
				return e
			}
			if recordingHash != "" {
				b, err := files.Read(work, "policy-input.mcap", recordingLimit)
				if err != nil || hashBytes(b) != recordingHash {
					return errors.New("排队后的 MCAP 快照已变化")
				}
			}
			for index, digest := range caseHashes {
				data, err := files.Read(work, fmt.Sprintf("input-case-%d.mcap", index+1), recordingLimit)
				if err != nil || hashBytes(data) != digest {
					return errors.New("排队后的测试用例 MCAP 快照已变化")
				}
			}

			j.State = "running"
			return s.saveSimulation(j)
		},
		OnExit: func(result runner.Result) {
			output("\n", "stdout")
			s.mu.Lock()
			defer s.mu.Unlock()
			switch {
			case result.Cancelled || j.State == "cancelling":
				j.State = "cancelled"
			case result.TimedOut:
				j.State = "failed"
				j.Error = "仿真超过运行时限，已停止"
			case result.Err != nil || result.Code != 0:
				j.State = "failed"
				j.Error = "仿真失败，请查看日志"
				if result.Err != nil {
					j.Error += "：" + result.Err.Error()
				}
			default:
				j.State = "finalizing"
				report, err := s.completeSimulation(j)
				if err != nil {
					j.State = "failed"
					j.Error = err.Error()
				} else {
					j.Result = report
					j.State = "succeeded"
				}
			}
			j.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
			if err := s.saveSimulation(j); err != nil {
				j.State = "failed"
				j.Error = "无法保存仿真任务"
				j.Result = nil
				_ = s.saveSimulation(j)
			}
		}}
	if j.Recording != "" {
		task.Args = append(task.Args, "--recording", filepath.Join(work, "policy-input.mcap"))
	}
	if j.VisionInjected {
		task.Args = append(task.Args, "--inject-vision")
	}
	if e = s.c.Executor.Submit(task); e != nil {
		j.State = "failed"
		j.Error = e.Error()
		j.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
		_ = s.saveSimulation(j)
		return Simulation{}, e
	}
	encoded, _ := json.Marshal(j)
	var copy Simulation
	_ = json.Unmarshal(encoded, &copy)
	return copy, nil
}
