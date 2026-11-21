package go1

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"path"
	"time"
)

func (d *Durable) prepareCatalog(ctx context.Context) error {
	tx, err := d.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx, durableDDL); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, "INSERT INTO aiops_go1.catalog_versions VALUES (3) ON CONFLICT DO NOTHING"); err != nil {
		return err
	}
	return tx.Commit()
}

func (d *Durable) restoreCatalog(ctx context.Context, root string) error {
	if err := d.restoreArtifacts(ctx, root); err != nil {
		return err
	}
	if err := d.restoreSets(ctx, root); err != nil {
		return err
	}
	if err := d.restoreJobs(ctx, root, "training_tasks", "jobs"); err != nil {
		return err
	}
	return d.restoreJobs(ctx, root, "simulation_tasks", "simulations")
}

func (d *Durable) restoreArtifacts(ctx context.Context, root string) error {
	rows, err := d.DB.QueryContext(ctx, "SELECT path,object_key,size_bytes FROM aiops_go1.artifacts ORDER BY path")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var obj storedFile
		if err := rows.Scan(&obj.Path, &obj.Key, &obj.Size); err != nil {
			return err
		}
		if err := d.restoreObject(ctx, root, obj); err != nil {
			return err
		}
	}
	return rows.Err()
}

func (d *Durable) restoreSets(ctx context.Context, root string) error {
	for _, s := range []struct{ table, members, column, folder, field string }{
		{"scene_sets", "scene_set_members", "scene_id", "scene-sets", "scenes"},
		{"test_case_sets", "test_case_set_members", "simulation_id", "test-case-sets", "cases"},
	} {
		query := fmt.Sprintf(`SELECT s.id,s.title,s.created_at,json_agg(m.%s ORDER BY m.ordinal)
FROM aiops_go1.%s s JOIN aiops_go1.%s m ON m.set_id=s.id GROUP BY s.id`, s.column, s.table, s.members)
		rows, err := d.DB.QueryContext(ctx, query)
		if err != nil {
			return err
		}
		err = func() error {
			defer rows.Close()
			for rows.Next() {
				var id, title string
				var created time.Time
				var refs []byte
				if err := rows.Scan(&id, &title, &created, &refs); err != nil {
					return err
				}
				data, err := json.Marshal(map[string]any{"id": id, "title": title, "createdAt": created.UTC().Format(time.RFC3339Nano), s.field: json.RawMessage(refs)})
				if err != nil {
					return err
				}
				if err := stageFile(root, path.Join(s.folder, id+".json"), data); err != nil {
					return err
				}
			}
			return rows.Err()
		}()
		if err != nil {
			return err
		}
	}
	return nil
}

func (d *Durable) restoreJobs(ctx context.Context, root, table, folder string) error {
	extra := `NULL::integer,
(SELECT COALESCE(json_object_agg(name,value),'{}') FROM aiops_go1.training_parameters WHERE training_id=t.id),
(SELECT COALESCE(json_agg(point ORDER BY iteration),'[]') FROM (
    SELECT iteration,json_object_agg(name,value) AS point
    FROM aiops_go1.training_metrics WHERE training_id=t.id
    GROUP BY iteration ORDER BY iteration DESC LIMIT 2000
) recent)`
	if table == "simulation_tasks" {
		extra = "progress,NULL::json,NULL::json"
	}
	rows, err := d.DB.QueryContext(ctx, "SELECT id,title,state,started_at,finished_at,logs,error,snapshot_key,"+extra+" FROM aiops_go1."+table+" t")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var id, title, state, logs, failure, key string
		var started time.Time
		var config, metrics []byte
		var finished sql.NullTime
		var progress sql.NullInt64
		if err := rows.Scan(&id, &title, &state, &started, &finished, &logs, &failure, &key, &progress, &config, &metrics); err != nil {
			return err
		}
		record, err := d.readSnapshot(ctx, key)
		if err != nil {
			return err
		}
		overlay := map[string]any{"id": id, "title": title, "state": state, "startedAt": started.UTC().Format(time.RFC3339Nano), "logs": logs, "error": failure}
		if config != nil {
			overlay["config"] = json.RawMessage(config)
			overlay["metrics"] = json.RawMessage(metrics)
		}
		if finished.Valid {
			overlay["finishedAt"] = finished.Time.UTC().Format(time.RFC3339Nano)
		}
		if progress.Valid {
			overlay["progress"] = progress.Int64
		}
		for k, v := range overlay {
			record[k], err = json.Marshal(v)
			if err != nil {
				return err
			}
		}
		data, err := json.Marshal(record)
		if err != nil {
			return err
		}
		if err := stageFile(root, path.Join(folder, id, "job.json"), data); err != nil {
			return err
		}
	}
	return rows.Err()
}
