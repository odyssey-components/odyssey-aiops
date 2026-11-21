package workflow

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"odyssey/aiops/internal/runner"

	"github.com/odyssey-components/odyssey-go/bootstrap/silo"
)

type Dispatch struct {
	k     kube
	store *silo.Client
	db    *sql.DB
}

func New(ctx context.Context, store *silo.Client, db *sql.DB) (*Dispatch, error) {
	k, err := inCluster()
	if err != nil {
		return nil, err
	}
	for _, name := range []string{trainTemplate, simulationTemplate} {
		if err := k.call(ctx, "GET", "/workflowtemplates/"+name, nil, nil); err != nil {
			return nil, err
		}
	}
	_, err = db.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS aiops_go1.workflow_executions (
task_id uuid PRIMARY KEY, namespace text NOT NULL, name text UNIQUE NOT NULL, uid text UNIQUE NOT NULL,
template text NOT NULL, phase text NOT NULL, submitted_at timestamptz NOT NULL DEFAULT now())`)
	if err != nil {
		return nil, err
	}
	d := &Dispatch{k: k, store: store, db: db}
	if err := d.stopInterrupted(ctx); err != nil {
		return nil, err
	}
	return d, nil
}

func (d *Dispatch) stopInterrupted(ctx context.Context) error {
	rows, err := d.db.QueryContext(ctx, "SELECT name,uid FROM aiops_go1.workflow_executions WHERE phase='active'")
	if err != nil {
		return err
	}
	defer rows.Close()
	var workflows [][2]string
	for rows.Next() {
		var id [2]string
		if err := rows.Scan(&id[0], &id[1]); err != nil {
			return err
		}
		workflows = append(workflows, id)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if err := rows.Close(); err != nil {
		return err
	}
	for _, wf := range workflows {
		if err := d.k.update(ctx, wf[0], wf[1], map[string]any{"shutdown": "Terminate", "suspend": false}); err != nil {
			var response apiError
			if !errors.As(err, &response) || response.status != 404 {
				return err
			}
		}
		if _, err := d.db.ExecContext(ctx, "UPDATE aiops_go1.workflow_executions SET phase='interrupted' WHERE uid=$1", wf[1]); err != nil {
			return err
		}
	}
	return nil
}

func stageOne(source, target string) error {
	return filepath.WalkDir(source, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		dest := filepath.Join(target, rel)
		if entry.IsDir() {
			return os.MkdirAll(dest, 0700)
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() || info.Size() > maxBundle {
			return errors.New("invalid Workflow input")
		}
		if err := os.MkdirAll(filepath.Dir(dest), 0700); err != nil {
			return err
		}
		src, err := os.Open(path)
		if err != nil {
			return err
		}
		defer src.Close()
		out, err := os.OpenFile(dest, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			return err
		}
		_, err = io.Copy(out, src)
		return errors.Join(err, out.Close())
	})
}

func stageTask(task runner.Task, root string) (request, string, error) {
	var req request
	var output string
	if !taskID.MatchString(task.ID) {
		return req, output, errors.New("invalid Workflow task ID")
	}
	index := -1
	for i, arg := range task.Args {
		if arg == "--" {
			index = i
			break
		}
	}
	if index < 0 || index+1 >= len(task.Args) || !allowedModule(task.Args[index+1]) {
		return req, output, errors.New("invalid Workflow task module")
	}
	req.Module = task.Args[index+1]
	req.Args = append([]string(nil), task.Args[index+2:]...)
	if err := os.MkdirAll(filepath.Join(root, "inputs"), 0700); err != nil {
		return req, output, err
	}
	for i := 0; i < len(req.Args); i++ {
		flag := req.Args[i]
		switch flag {
		case "--source", "--config", "--output", "--base-policy", "--model", "--dataset", "--recording":
			i++
			if i >= len(req.Args) {
				return req, output, errors.New("missing Workflow path")
			}
			original := req.Args[i]
			switch flag {
			case "--source":
				req.Args[i] = go1Source
			case "--output":
				output = original
				req.Output = filepath.Base(original)
				req.Args[i] = req.Output
			default:
				name := filepath.Join("inputs", filepath.Base(original))
				if err := stageOne(original, filepath.Join(root, name)); err != nil {
					return req, output, err
				}
				req.Args[i] = name
				if flag == "--config" && req.Module == "aiops_go1_training.runtime.simulation" {
					files, err := filepath.Glob(filepath.Join(filepath.Dir(original), "input-case-*.mcap"))
					if err != nil {
						return req, output, err
					}
					for _, f := range files {
						if err := stageOne(f, filepath.Join(root, "inputs", filepath.Base(f))); err != nil {
							return req, output, err
						}
					}
				}
			}
		}
	}
	if !validOutput(req.Output) || !filepath.IsAbs(output) {
		return req, output, errors.New("invalid Workflow output")
	}
	return req, output, nil
}

func (d *Dispatch) Run(task runner.Task, cancel <-chan struct{}) (result runner.Result) {
	result.Code = -1
	timeout := task.Timeout
	if timeout <= 0 {
		timeout = 15 * time.Minute
	}
	ctx, stop := context.WithTimeout(context.Background(), timeout)
	defer stop()
	go func() {
		select {
		case <-cancel:
			stop()
		case <-ctx.Done():
		}
	}()
	root, err := os.MkdirTemp("", "aiops-dispatch-")
	if err != nil {
		result.Err = err
		return
	}
	defer os.RemoveAll(root)
	req, output, err := stageTask(task, root)
	if err != nil {
		result.Err = err
		return
	}
	data, err := json.Marshal(req)
	if err == nil {
		err = os.WriteFile(filepath.Join(root, "request.json"), data, 0600)
	}
	if err == nil {
		err = send(ctx, d.store, prefix(task.ID)+"input.tar", root, []string{"request.json", "inputs"})
	}
	if err != nil {
		result.Err = err
		return
	}
	template := trainTemplate
	if req.Module == "aiops_go1_training.runtime.simulation" {
		template = simulationTemplate
	}
	name := "aiops-" + task.ID
	wf, err := d.k.submit(ctx, name, template, task.ID, timeout)
	if err != nil || wf.Metadata.Name != name || wf.Metadata.UID == "" {
		result.Err = errors.Join(err, errors.New("Workflow submission did not return identity"))
		return
	}
	uid := wf.Metadata.UID
	phase := "active"
	defer func() {
		end, endCancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer endCancel()
		if phase == "active" {
			select {
			case <-cancel:
				result.Cancelled = true
			default:
			}
			result.TimedOut = errors.Is(ctx.Err(), context.DeadlineExceeded)
			if err := d.k.update(end, name, uid, map[string]any{"shutdown": "Terminate", "suspend": false}); err != nil {
				result.Err = errors.Join(result.Err, err)
				return
			}
			phase = "terminated"
		}
		if _, err := d.db.ExecContext(end, "UPDATE aiops_go1.workflow_executions SET phase=$2 WHERE uid=$1", uid, phase); err != nil {
			result.Err = errors.Join(result.Err, err)
		}
	}()
	_, err = d.db.ExecContext(ctx, "INSERT INTO aiops_go1.workflow_executions(task_id,namespace,name,uid,template,phase) VALUES ($1,$2,$3,$4,$5,'active')", task.ID, d.k.namespace, name, uid, template)
	if err == nil {
		err = d.k.update(ctx, name, uid, map[string]any{"suspend": false})
	}
	if err != nil {
		result.Err = err
		return
	}
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		var current execution
		err = d.k.call(ctx, "GET", "/workflows/"+name, nil, &current)
		if err != nil {
			if !retryable(err) {
				result.Err = err
				return
			}
			select {
			case <-ctx.Done():
				result.Err = errors.Join(ctx.Err(), err)
				return
			case <-ticker.C:
				continue
			}
		}
		if current.Metadata.UID != uid {
			result.Err = errors.New("Workflow UID changed")
			return
		}
		switch current.Status.Phase {
		case "Succeeded":
			phase = "succeeded"
			result.Err = d.collect(ctx, task, req, output)
			if result.Err == nil {
				result.Code = 0
			}
			return
		case "Error", "Failed":
			phase = "failed"
			result.Err = fmt.Errorf("Workflow %s: %s", name, current.Status.Phase)
			return
		}
		select {
		case <-ctx.Done():
			result.Err = ctx.Err()
			return
		case <-ticker.C:
		}
	}
}

func (d *Dispatch) collect(ctx context.Context, task runner.Task, req request, output string) error {
	resultDir, err := os.MkdirTemp(filepath.Dir(output), ".aiops-result-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(resultDir)
	sidecar := strings.TrimSuffix(req.Output, ".jsonl") + ".toml"
	dataset := req.Module == "aiops_maze_training.common.datasets"
	if err := receive(ctx, d.store, prefix(task.ID)+"output.tar", resultDir, func(name string) bool {
		return name == "worker.log" || name == req.Output || strings.HasPrefix(name, req.Output+"/") || dataset && name == sidecar
	}); err != nil {
		return err
	}
	if task.OnOutput != nil {
		if log, err := os.Open(filepath.Join(resultDir, "worker.log")); err == nil {
			buf := make([]byte, 4096)
			for {
				n, err := log.Read(buf)
				if n > 0 {
					task.OnOutput(string(buf[:n]), "stdout")
				}
				if err != nil {
					break
				}
			}
			_ = log.Close()
		}
	}
	if dataset {
		if err := os.Rename(filepath.Join(resultDir, sidecar), strings.TrimSuffix(output, ".jsonl")+".toml"); err != nil {
			return err
		}
	}
	return os.Rename(filepath.Join(resultDir, req.Output), output)
}
