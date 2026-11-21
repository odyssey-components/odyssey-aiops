package go1

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"path/filepath"
	"strings"
)

type catalogWriter struct {
	ctx  context.Context
	tx   *sql.Tx
	root string
	d    *Durable
}

func (w catalogWriter) exec(query string, args ...any) error {
	_, err := w.tx.ExecContext(w.ctx, query, args...)
	return err
}

func (w catalogWriter) write(key string, data []byte, objects []storedFile) error {
	p := strings.Split(filepath.ToSlash(key), "/")
	if len(p) < 2 {
		return errors.New("invalid Go1 catalog record")
	}
	id := strings.TrimSuffix(strings.TrimSuffix(p[1], ".json"), ".toml")
	if !validID.MatchString(id) {
		return errors.New("invalid Go1 catalog ID")
	}
	switch p[0] {
	case "scene-sets":
		var set SceneSet
		if err := json.Unmarshal(data, &set); err != nil {
			return err
		}
		return w.set("scene_sets", "scene_set_members", "scene_id", id, set.Title, set.CreatedAt, set.Scenes)
	case "test-case-sets":
		var set TestCaseSet
		if err := json.Unmarshal(data, &set); err != nil {
			return err
		}
		return w.set("test_case_sets", "test_case_set_members", "simulation_id", id, set.Title, set.CreatedAt, set.Cases)
	case "public-policies":
		return w.model("public-policies", id, nil, objects)
	case "catalog":
		return w.model("bases", id, nil, objects)
	case "jobs":
		var job Job
		if err := json.Unmarshal(data, &job); err != nil {
			return err
		}
		return w.training(job, data, objects)
	case "simulations":
		var job Simulation
		if err := json.Unmarshal(data, &job); err != nil {
			return err
		}
		return w.simulation(job, data, objects)
	default:
		return errors.New("unsupported Go1 catalog record")
	}
}

func (w catalogWriter) set(table, members, memberColumn, id, title, created string, ids []string) error {
	if err := w.exec("INSERT INTO aiops_go1."+table+" VALUES ($1,$2,$3) ON CONFLICT DO NOTHING", id, title, created); err != nil {
		return err
	}
	for i, member := range ids {
		query := fmt.Sprintf("INSERT INTO aiops_go1.%s (set_id,ordinal,%s) VALUES ($1,$2,$3) ON CONFLICT DO NOTHING", members, memberColumn)
		if err := w.exec(query, id, i+1, member); err != nil {
			return err
		}
	}
	return nil
}

func (w catalogWriter) snapshot(kind, id, state string, data []byte) (string, error) {
	if active(state) && state != "queued" {
		return "", nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return "", err
	}
	for _, field := range []string{"id", "title", "state", "startedAt", "finishedAt", "logs", "error", "progress", "config", "metrics"} {
		delete(fields, field)
	}
	payload, err := json.Marshal(fields)
	if err != nil {
		return "", err
	}
	key := path.Join("go1/snapshots", kind, id, hashBytes(payload)+".json")
	_, err = w.d.Silo.PutIfAbsent(w.ctx, key, payload, "application/json")
	return key, err
}

func (w catalogWriter) training(j Job, data []byte, objects []storedFile) error {
	key, err := w.snapshot("training", j.ID, j.State, data)
	if err != nil {
		return err
	}
	var scene, base any
	if j.Scene != nil {
		scene = j.Scene.ID
	}
	if j.Base != nil {
		base = j.Base.ID
	}
	if err = w.exec(`INSERT INTO aiops_go1.training_tasks VALUES ($1,$2,$3,$4,$5,NULLIF($6,'')::timestamptz,$7,$8,$9,$10,$11)
ON CONFLICT (id) DO UPDATE SET state=EXCLUDED.state, finished_at=EXCLUDED.finished_at,
logs=EXCLUDED.logs,error=EXCLUDED.error,snapshot_key=COALESCE(NULLIF(EXCLUDED.snapshot_key,''),training_tasks.snapshot_key)`,
		j.ID, j.Title, j.Device, j.State, j.StartedAt, j.FinishedAt, scene, base, j.Logs, j.Error, key); err != nil {
		return err
	}
	if key != "" {
		for name, value := range j.Config {
			if err := w.exec(`INSERT INTO aiops_go1.training_parameters VALUES ($1,$2,$3) ON CONFLICT DO NOTHING`, j.ID, name, value); err != nil {
				return err
			}
		}
	}
	// Persist each metric when it arrives; materialized snapshots preserve the API history format.
	metrics := j.Metrics
	if key == "" && len(metrics) > 0 {
		metrics = metrics[len(metrics)-1:]
	}
	for _, point := range metrics {
		for name, value := range point {
			if err := w.exec(`INSERT INTO aiops_go1.training_metrics VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING`, j.ID, int(point["iteration"]), name, value); err != nil {
				return err
			}
		}
	}
	if j.State == "succeeded" {
		return w.model("runs", j.ID, j.ID, objects)
	}
	return nil
}

func (w catalogWriter) model(area, id string, training any, objects []storedFile) error {
	loader := manifest
	if area == "public-policies" {
		loader = publicPolicyManifest
	}
	m, err := loader(filepath.Join(w.root, area), id)
	if err != nil {
		return err
	}
	format, _ := m["format"].(string)
	if format == "" {
		format, _ = m["kind"].(string)
	}
	title, _ := m["title"].(string)
	if title == "" {
		title = id
	}
	if err := w.exec(`INSERT INTO aiops_go1.models VALUES ($1,$2,$3,$4,$5) ON CONFLICT DO NOTHING`, id, title, format, m["policy_sha256"], training); err != nil {
		return err
	}
	if area == "public-policies" {
		var existing string
		if err := w.tx.QueryRowContext(w.ctx, "SELECT policy_sha256 FROM aiops_go1.models WHERE id=$1 AND format=$2", id, format).Scan(&existing); err != nil {
			return err
		}
		if existing != m["policy_sha256"] {
			return errors.New("公开模型权重与登记记录不一致")
		}
	}
	return w.artifacts(id, nil, objects)
}

// Public policies live outside the training catalog. Register their verified
// identity before inserting a simulation that references them.
func (w catalogWriter) registerSimulationPolicy(j Simulation) error {
	if !isPublicPolicy(j.ModelManifest) {
		return nil
	}
	title, _ := j.ModelManifest["title"].(string)
	format, _ := j.ModelManifest["format"].(string)
	digest, _ := j.ModelManifest["policy_sha256"].(string)
	if title == "" || !regexpHash(digest) {
		return errors.New("公开模型清单不完整")
	}
	result, err := w.tx.ExecContext(w.ctx, `INSERT INTO aiops_go1.models AS m (id,title,format,policy_sha256,training_id)
VALUES ($1,$2,$3,$4,NULL) ON CONFLICT (id) DO UPDATE SET title=m.title
WHERE m.format=EXCLUDED.format AND m.policy_sha256=EXCLUDED.policy_sha256 AND m.training_id IS NULL`,
		j.Model, title, format, digest)
	if err != nil {
		return err
	}
	count, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if count != 1 {
		return errors.New("公开模型 ID 与现有模型冲突")
	}
	return nil
}

func (w catalogWriter) simulation(j Simulation, data []byte, objects []storedFile) error {
	if err := w.registerSimulationPolicy(j); err != nil {
		return err
	}
	key, err := w.snapshot("simulation", j.ID, j.State, data)
	if err != nil {
		return err
	}
	var sceneSet, testSet any
	if j.SceneSet != nil {
		sceneSet = j.SceneSet.ID
	}
	if j.TestCaseSet != nil {
		testSet = j.TestCaseSet.ID
	}
	if err = w.exec(`INSERT INTO aiops_go1.simulation_tasks VALUES ($1,$2,$3,$4,NULLIF($5,'')::timestamptz,$6,$7,$8,$9,NULLIF($10,'')::uuid,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
ON CONFLICT (id) DO UPDATE SET state=EXCLUDED.state,finished_at=EXCLUDED.finished_at,
progress=EXCLUDED.progress,recorded=EXCLUDED.recorded,outcome=EXCLUDED.outcome,passed=EXCLUDED.passed,
logs=EXCLUDED.logs,error=EXCLUDED.error,snapshot_key=COALESCE(NULLIF(EXCLUDED.snapshot_key,''),simulation_tasks.snapshot_key)`,
		j.ID, j.Title, j.State, j.StartedAt, j.FinishedAt, j.Model, j.Scene.ID, sceneSet, testSet, j.Recording,
		j.Seed, j.Steps, j.Progress, j.Recorded, j.VisionInjected, j.Result["outcome"], j.Result["passed"], j.Logs, j.Error, key); err != nil {
		return err
	}
	if key != "" {
		for i, scene := range j.Cases {
			seed, steps := j.Seed, j.Steps
			if i < len(j.CaseInputs) {
				seed, steps = j.CaseInputs[i].Seed, j.CaseInputs[i].Steps
			}
			if err := w.exec(`INSERT INTO aiops_go1.simulation_cases VALUES ($1,$2,$3,$4,$5) ON CONFLICT DO NOTHING`, j.ID, i+1, scene.ID, seed, steps); err != nil {
				return err
			}
		}
	}
	return w.artifacts(nil, j.ID, objects)
}

func (w catalogWriter) artifacts(model, simulation any, objects []storedFile) error {
	for _, object := range objects {
		if err := w.exec(`INSERT INTO aiops_go1.artifacts VALUES ($1,$2,$3,$4,$5)
ON CONFLICT (path) DO UPDATE SET object_key=EXCLUDED.object_key,size_bytes=EXCLUDED.size_bytes`,
			object.Path, object.Key, object.Size, model, simulation); err != nil {
			return err
		}
	}
	return nil
}

func (d *Durable) readSnapshot(ctx context.Context, key string) (map[string]json.RawMessage, error) {
	b, err := d.Silo.Read(ctx, key, 2<<20)
	if err != nil {
		return nil, err
	}
	var record map[string]json.RawMessage
	err = json.NewDecoder(bytes.NewReader(b)).Decode(&record)
	return record, err
}
