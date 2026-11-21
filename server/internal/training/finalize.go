package training

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"github.com/pelletier/go-toml/v2"
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"time"
)

func (s *Service) finish(j *Job, c Command, metrics *collector, r runner.Result) {
	s.mu.Lock()
	defer s.mu.Unlock()
	metrics.end()
	var err error
	switch {
	case r.Cancelled || j.State == "cancelling":
		j.State = "cancelled"
	case r.TimedOut:
		err = errors.New("任务超过时间限制，已停止")
	case r.Err != nil || r.Code != 0:
		err = errors.New("Python 任务失败，请查看日志")
		if r.Err != nil {
			err = errors.New("Python 任务失败：" + r.Err.Error())
		}
	default:
		j.State = "finalizing"
		err = s.save(j, c)
		if err == nil {
			if c.Action == "prepare" {
				var d DatasetInfo
				d, err = s.dataset(j.ID)
				if err == nil && (d.Course != c.Course || d.Seed != c.Seed) {
					err = errors.New("地图组课时或种子不匹配")
				}
				if err == nil {
					j.Dataset = d.ID
					j.DatasetSha256 = d.Sha256
				}
			} else {
				d, e := s.dataset(c.Dataset)
				if e != nil || d.Sha256 != j.DatasetSha256 {
					err = errors.New("地图集在训练期间已更改")
				} else {
					output := filepath.Join(s.jobs, j.ID, "output")
					err = validateOutput(output, c.Course)
					if err == nil {
						run := time.Now().UTC().Format("20060102150405000000") + "-" + c.Course + "-" + j.ID
						err = directory(s.runs)
						if err == nil {
							err = os.Rename(output, filepath.Join(s.runs, run))
						}
						if err == nil {
							j.Run = run
						}
					}
				}
			}
		}
		if err == nil {
			j.State = "succeeded"
		}
	}
	if err != nil {
		j.State = "failed"
		j.Error = err.Error()
	}
	j.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
	if err = s.save(j, c); err != nil {
		j.State = "failed"
		j.Error = "无法保存任务元信息：" + err.Error()
	}
}
func validateOutput(output, course string) error {
	resolved, err := filepath.EvalSymlinks(output)
	if err != nil || resolved != output {
		return errors.New("训练产物目录无效")
	}
	var onnx []byte
	var manifest map[string]any
	for _, name := range []string{"model.json", "model.onnx", "model.manifest.toml", "samples.json", "report.json"} {
		limit := int64(8 << 20)
		if name == "model.manifest.toml" {
			limit = 64 << 10
		}
		b, e := bounded(filepath.Join(output, name), limit)
		if e != nil || len(b) == 0 {
			return errors.New("训练产物不完整，未发布模型")
		}
		if name == "model.onnx" {
			onnx = b
		} else if name == "model.manifest.toml" {
			if toml.Unmarshal(b, &manifest) != nil {
				return errors.New("模型清单无效")
			}
		} else {
			var value any
			if json.Unmarshal(b, &value) != nil {
				return errors.New("训练产物 JSON 无效")
			}
		}
	}
	sum := sha256.Sum256(onnx)
	if manifest["schema"] != "odymaze-onnx/v1" || manifest["filename"] != "model.onnx" || manifest["lesson"] != course {
		return errors.New("模型清单规格不匹配")
	}
	if manifest["sha256"] != hex.EncodeToString(sum[:]) {
		return errors.New("模型 SHA-256 不匹配")
	}
	if l, ok := manifest["lesson"].(string); ok && l != course {
		return errors.New("模型课时不匹配")
	}
	return nil
}
