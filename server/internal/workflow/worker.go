package workflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/odyssey-components/odyssey-go/bootstrap/silo"
)

var taskID = regexp.MustCompile(`^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$`)

const go1Source = "/opt/aiops/go1-source"

type request struct {
	Module string
	Args   []string
	Output string
}

func allowedModule(module string) bool {
	switch module {
	case "aiops_go1_training.runtime.local", "aiops_go1_training.runtime.upstream", "aiops_go1_training.vision.train",
		"aiops_go1_training.runtime.simulation", "aiops_maze_training.common.datasets", "aiops_maze_training.runtime.train":
		return true
	}
	return false
}

func prefix(id string) string { return "aiops/workflows/v1/" + id + "/" }

func validOutput(name string) bool {
	return name == "output" || (strings.HasSuffix(name, ".jsonl") && taskID.MatchString(strings.TrimSuffix(name, ".jsonl")))
}

func RunWorker(ctx context.Context, id string) error {
	if !taskID.MatchString(id) {
		return errors.New("invalid task ID")
	}
	store, err := silo.New(silo.Config{Endpoint: os.Getenv("AIOPS_SILO_ENDPOINT"), Bucket: os.Getenv("AIOPS_SILO_BUCKET"), AccessKey: os.Getenv("AIOPS_SILO_ACCESS_KEY"), SecretKey: os.Getenv("AIOPS_SILO_SECRET_KEY")})
	if err != nil {
		return err
	}
	if store == nil {
		return errors.New("missing worker Silo configuration")
	}
	defer store.Close()
	root, err := os.MkdirTemp("", "aiops-task-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(root)
	if err := receive(ctx, store, prefix(id)+"input.tar", root, func(name string) bool {
		return name == "request.json" || strings.HasPrefix(name, "inputs/")
	}); err != nil {
		return err
	}
	b, err := os.ReadFile(filepath.Join(root, "request.json"))
	if err != nil {
		return err
	}
	var req request
	if json.Unmarshal(b, &req) != nil || !allowedModule(req.Module) || !validOutput(req.Output) {
		return errors.New("invalid worker request")
	}
	logPath := filepath.Join(root, "worker.log")
	log, err := os.OpenFile(logPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	run := func(args ...string) error {
		cmd := exec.CommandContext(ctx, "/usr/local/bin/python", args...)
		cmd.Dir = root
		cmd.Env = append(os.Environ(), "JAX_PLATFORMS=cpu", "AIOPS_GO1_DEVICE=cpu", "XLA_PYTHON_CLIENT_PREALLOCATE=false", "PYTHONUNBUFFERED=1", "MUJOCO_GL=egl", "PYOPENGL_PLATFORM=egl")
		cmd.Stdout = io.MultiWriter(os.Stdout, log)
		cmd.Stderr = io.MultiWriter(os.Stderr, log)
		return cmd.Run()
	}
	err = run(append([]string{"-u", "-m", req.Module}, req.Args...)...)
	if err == nil {
		switch req.Module {
		case "aiops_go1_training.runtime.local":
			err = run("-m", req.Module, "--validate-output", req.Output)
		case "aiops_go1_training.runtime.upstream", "aiops_go1_training.vision.train":
			err = run("-m", req.Module, "--source", go1Source, "--validate-output", req.Output)
		}
	}
	if closeErr := log.Close(); closeErr != nil {
		return errors.Join(err, closeErr)
	}
	if err != nil {
		return fmt.Errorf("worker task failed: %w", err)
	}
	names := []string{req.Output, "worker.log"}
	if req.Module == "aiops_maze_training.common.datasets" {
		names = append(names, strings.TrimSuffix(req.Output, ".jsonl")+".toml")
	}
	return send(ctx, store, prefix(id)+"output.tar", root, names)
}
