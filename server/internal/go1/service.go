package go1

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"odyssey/aiops/internal/files"
	"odyssey/aiops/internal/models"
	"odyssey/aiops/internal/runner"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

type Config struct {
	DataDir, Python, Source string
	PublicPolicyDir         string
	Executor                *runner.Executor
	Durable                 *Durable
	Workflow                bool
}
type Service struct {
	c                    Config
	mu                   sync.Mutex
	live                 map[string]*Job
	simulations          map[string]*Simulation
	environment          map[string]any
	environmentOnce      sync.Once
	trainingEnvironments map[string]map[string]any
}

func New(c Config) (*Service, error) {
	if c.Executor == nil || c.Python == "" {
		return nil, errors.New("Python 和执行器不能为空")
	}
	for _, dir := range []string{".", "jobs", "runs", "scenes", "scene-sets", "test-case-sets", "simulations", "bases"} {
		if e := os.MkdirAll(filepath.Join(c.DataDir, dir), 0700); e != nil {
			return nil, e
		}
		p := filepath.Join(c.DataDir, dir)
		resolved, e := filepath.EvalSymlinks(p)
		absolute, _ := filepath.Abs(p)
		if e != nil || resolved != absolute {
			return nil, errors.New("Go1 数据目录不允许符号链接")
		}
	}
	service := &Service{c: c, live: map[string]*Job{}, simulations: map[string]*Simulation{}}
	if err := service.installPublicPolicies(); err != nil {
		return nil, err
	}
	return service, nil
}
func (s *Service) env(device ...string) []string {
	env := []string{}
	for _, value := range os.Environ() {
		key, _, _ := strings.Cut(value, "=")
		if key != "AIOPS_GO1_DEVICE" && key != "PYTHONPATH" && key != "PYTHONHOME" && key != "JAX_PLATFORMS" && key != "XLA_PYTHON_CLIENT_PREALLOCATE" && key != "PYTHONUNBUFFERED" {
			env = append(env, value)
		}
	}
	return append(env, "JAX_PLATFORMS="+selectedDevice(device), "AIOPS_GO1_DEVICE="+selectedDevice(device), "XLA_PYTHON_CLIENT_PREALLOCATE=false", "PYTHONUNBUFFERED=1")
}
func (s *Service) visionEnv(device ...string) []string {
	result := []string{}
	hasGL := false
	for _, value := range s.env(device...) {
		if strings.HasPrefix(value, "MUJOCO_GL=") {
			hasGL = true
		}
		result = append(result, value)
	}
	if !hasGL {
		result = append(result, "MUJOCO_GL=egl")
	}
	return result
}
func (s *Service) capture(args ...string) ([]byte, error) { return s.captureDevice("cpu", args...) }
func (s *Service) captureDevice(device string, args ...string) ([]byte, error) {
	vision := len(args) > 1 && args[1] == "aiops_go1_training.vision.train"
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	// Probes and numeric validation also stop if the owning Go process exits.
	if len(args) >= 2 && args[0] == "-m" && (args[1] == "aiops_go1_training.runtime.local" || args[1] == "aiops_go1_training.runtime.upstream" || args[1] == "aiops_go1_training.vision.train") {
		args = append([]string{"-u", "-m", "aiops_go1_training.runtime.worker", "--parent-pid", fmt.Sprint(os.Getpid()), "--", args[1]}, args[2:]...)
	}
	cmd := exec.CommandContext(ctx, s.c.Python, args...)
	cmd.Env = s.env(device)
	if vision {
		cmd.Env = s.visionEnv(device)
	}
	cmd.Dir = s.c.DataDir
	var out capped
	cmd.Stdout = &out
	cmd.Stderr = io.Discard
	e := cmd.Run()
	return out.b.Bytes(), e
}

type capped struct{ b bytes.Buffer }

func (c *capped) Write(p []byte) (int, error) {
	n := len(p)
	if c.b.Len()+len(p) > 1<<20 {
		return n, errors.New("进程输出过大")
	}
	_, e := c.b.Write(p)
	return n, e
}
func (s *Service) probeModule(module, device string) map[string]any {
	b, e := s.captureDevice(device, "-m", module, "--check", "--source", s.c.Source)
	v := map[string]any{"ready": false, "errors": []string{"Go1 Python 环境或源码不可用"}, "devices": []string{}, "source_commit": Commit}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) > 0 {
		var parsed map[string]any
		if json.Unmarshal([]byte(lines[len(lines)-1]), &parsed) == nil {
			if ready, ok := parsed["ready"].(bool); ok && (e == nil || !ready) {
				v = parsed
			}
		}
	}
	v["device"] = device
	return v
}
func (s *Service) persist(j *Job) error {
	dir := filepath.Join(s.c.DataDir, "jobs", j.ID)
	b, e := json.Marshal(j)
	if e != nil {
		return e
	}
	artifacts := ""
	if j.State == "succeeded" {
		artifacts = filepath.Join("runs", j.ID)
	}
	if e = s.storeRecord(filepath.Join("jobs", j.ID, "job.json"), b, artifacts); e != nil {
		return e
	}
	if e = os.WriteFile(filepath.Join(dir, "job.json.tmp"), b, 0600); e != nil {
		return e
	}
	return os.Rename(filepath.Join(dir, "job.json.tmp"), filepath.Join(dir, "job.json"))
}
func (s *Service) snapshot() any { return s.snapshotPage(nil) }

func (s *Service) snapshotPage(page *listPage) any {
	s.mu.Lock()
	defer s.mu.Unlock()
	jobs := map[string]*Job{}
	entries, _ := os.ReadDir(filepath.Join(s.c.DataDir, "jobs"))
	for _, entry := range entries {
		if !entry.IsDir() || !validID.MatchString(entry.Name()) {
			continue
		}
		b, e := files.Read(s.c.DataDir, filepath.Join("jobs", entry.Name(), "job.json"), 512<<10)
		var j Job
		if e != nil || json.Unmarshal(b, &j) != nil || j.ID != entry.Name() || j.Format != "go1-local-job/v1" || utf8.RuneCountInString(j.Title) > 80 || len(j.Logs) > 24000 || len(j.Metrics) > 2000 {
			continue
		}
		if _, e = time.Parse(time.RFC3339Nano, j.StartedAt); e != nil {
			continue
		}
		if !active(j.State) && j.State != "succeeded" && j.State != "failed" && j.State != "cancelled" {
			continue
		}
		if j.Scene != nil && (validateScene(*j.Scene) != nil || !validID.MatchString(j.Scene.ID)) {
			continue
		}
		if j.Base != nil && (!validID.MatchString(j.Base.ID) || !regexpHash(j.Base.PolicySHA256) || !regexpHash(j.Base.ManifestSHA256)) {
			continue
		}
		if _, e = config(j.Config); e != nil {
			continue
		}
		if active(j.State) {
			j.State = "failed"
			j.Error = "服务已重启，旧任务已中断，未自动恢复"
		}
		jobs[j.ID] = &j
	}
	for id, j := range s.live {
		jobs[id] = j
	}
	tasks := []*Job{}
	for _, j := range jobs {
		tasks = append(tasks, j)
	}
	sort.Slice(tasks, func(i, j int) bool {
		if tasks[i].StartedAt == tasks[j].StartedAt {
			return tasks[i].ID > tasks[j].ID
		}
		return tasks[i].StartedAt > tasks[j].StartedAt
	})
	value := map[string]any{"tasks": tasks, "models": s.availableModels(), "scenes": s.fixedScenes(), "sceneSets": s.sceneSets(), "testCaseSets": s.testCaseSets(), "baseModels": s.baseModels(jobs), "simulations": s.simulationHistory(), "environment": s.environment, "activity": s.c.Executor.Activity()}
	if page != nil {
		page.apply(value, jobs)
	}
	encoded, _ := json.Marshal(value)
	var copy map[string]any
	_ = json.Unmarshal(encoded, &copy)
	return copy
}
func (s *Service) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	fail := func(status int, e error) { models.JSON(w, status, map[string]string{"error": e.Error()}) }
	if r.Method == "GET" {
		q := r.URL.Query()
		if q.Has("list") {
			page, err := parseListPage(q)
			if err != nil {
				fail(400, err)
				return
			}
			models.JSON(w, 200, s.snapshotPage(&page))
			return
		}
		if len(q) > 0 && (s.serveRecordingFrame(w, r) || s.serveRecordedDepth(w, r) || s.serveInjectedVision(w, r) || s.serveRegression(w, r) || s.serveSimulationReplay(w, r)) {
			return
		}
		if len(q) > 0 {
			if len(q) != 1 || len(q["replay"]) != 1 {
				fail(400, errors.New("查询参数无效"))
				return
			}
			frames, e := replay(filepath.Join(s.c.DataDir, "runs"), q.Get("replay"))
			if e != nil {
				fail(400, e)
				return
			}
			models.JSON(w, 200, map[string]any{"frames": frames})
			return
		}
		models.JSON(w, 200, s.snapshot())
		return
	}
	if r.Method != "POST" {
		w.WriteHeader(405)
		return
	}
	var raw map[string]json.RawMessage
	d := json.NewDecoder(r.Body)
	if d.Decode(&raw) != nil || d.Decode(new(any)) != io.EOF {
		fail(400, errors.New("JSON格式无效"))
		return
	}
	if s.serveSimulationCommand(w, r, raw) {
		return
	}
	var action string
	_ = json.Unmarshal(raw["action"], &action)
	if action == "create-scene" {
		fail(400, errors.New("场景使用固定预设，不支持创建或修改"))
		return
	}
	if action == "create-test-case-set" && len(raw) == 2 {
		set, err := s.createTestCaseSet(raw["testCaseSet"])
		if err != nil {
			fail(400, err)
			return
		}
		models.JSON(w, 201, set)
		return
	}
	if action == "create-scene-set" && len(raw) == 2 {
		set, e := s.createSceneSet(raw["sceneSet"])
		if e != nil {
			fail(400, e)
			return
		}
		models.JSON(w, 201, set)
		return
	}
	if action == "check" && len(raw) == 1 {
		s.DetectTrainingEnvironment()
		models.JSON(w, 200, s.environment)
		return
	}
	if action == "cancel" && len(raw) == 2 {
		var id string
		_ = json.Unmarshal(raw["id"], &id)
		s.mu.Lock()
		j := s.live[id]
		if j == nil || !active(j.State) || j.State == "finalizing" {
			s.mu.Unlock()
			fail(409, errors.New("没有可取消的本模块任务"))
			return
		}
		j.State = "cancelling"
		s.c.Executor.Cancel(id)
		_ = s.persist(j)
		b, _ := json.Marshal(j)
		var copy Job
		_ = json.Unmarshal(b, &copy)
		s.mu.Unlock()
		models.JSON(w, 202, copy)
		return
	}
	if action != "train" {
		fail(400, errors.New("参数无效"))
		return
	}
	for k := range raw {
		if k != "action" && k != "title" && k != "config" && k != "scene" && k != "base" {
			fail(400, errors.New("未知参数"))
			return
		}
	}
	var title string
	var cfg map[string]float64
	if json.Unmarshal(raw["title"], &title) != nil || strings.TrimSpace(title) == "" || len([]rune(title)) > 80 || json.Unmarshal(raw["config"], &cfg) != nil || cfg == nil {
		fail(400, errors.New("请填写任务名称和有效训练参数"))
		return
	}
	cfg, e := config(cfg)
	if e != nil {
		fail(400, e)
		return
	}
	var base *BaseModel
	if value, ok := raw["base"]; ok {
		var id string
		if json.Unmarshal(value, &id) != nil {
			fail(400, errors.New("基础模型引用无效"))
			return
		}
		selectedBase, e := s.loadBase(id)
		if e != nil {
			fail(400, e)
			return
		}
		base = &selectedBase
	}
	selected, e := s.resolveTrainingScenes(raw, cfg)
	if e != nil {
		fail(400, e)
		return
	}
	if cfg["random_commands"] != 0 && (base == nil || len(selected) > 0) {
		fail(400, errors.New("随机速度指令需要基础模型与空旷训练环境"))
		return
	}
	checkModule := "aiops_go1_training.runtime.local"
	if base != nil {
		checkModule = "aiops_go1_training.runtime.upstream"
	}
	if len(selected) > 0 {
		checkModule = "aiops_go1_training.vision.train"
	}
	s.DetectTrainingEnvironment()
	environment := s.trainingEnvironments[checkModule]
	device, _ := environment["device"].(string)
	if environment["ready"] != true {
		fail(503, fmt.Errorf("Go1 %s 环境未就绪：%v", device, environment["errors"]))
		return
	}
	j, e := s.trainWithBase(device, strings.TrimSpace(title), cfg, base, selected...)
	if e != nil {
		fail(409, e)
		return
	}
	models.JSON(w, 202, j)
}
func newID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	b[6] = (b[6] & 15) | 64
	b[8] = (b[8] & 63) | 128
	h := hex.EncodeToString(b)
	return fmt.Sprintf("%s-%s-%s-%s-%s", h[:8], h[8:12], h[12:16], h[16:20], h[20:])
}
