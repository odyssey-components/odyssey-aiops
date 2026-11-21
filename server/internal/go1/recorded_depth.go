package go1

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"net/http"
	"path/filepath"

	"github.com/foxglove/mcap/go/mcap"
	"odyssey/aiops/internal/files"
	"odyssey/aiops/internal/models"
)

func recordedDepth(data []byte, count int) ([]byte, error) {
	if count < 1 || count > 500 {
		return nil, errors.New("录制步数无效")
	}
	var result []byte
	var stride int
	sequence := 0
	err := visitRecording(data, func(camera map[string]any, message *mcap.Message) (bool, error) {
		if sequence == 0 {
			width, wx := camera["width"].(float64)
			height, hx := camera["height"].(float64)
			if !wx || !hx || width < 16 || height < 16 || width > 128 || height > 128 || float64(int(width)) != width || float64(int(height)) != height {
				return false, errors.New("该录制没有有效的深度规格")
			}
			stride = int(width*height) * 4
			result = make([]byte, 8+stride*count)
			binary.LittleEndian.PutUint32(result, uint32(width))
			binary.LittleEndian.PutUint32(result[4:], uint32(height))
		}
		if int(message.Sequence) != sequence {
			return false, errors.New("录制深度帧顺序无效")
		}
		var frame struct {
			Depth string `json:"depth_f32le_base64"`
		}
		if err := json.Unmarshal(message.Data, &frame); err != nil {
			return false, err
		}
		if len(frame.Depth) != base64.StdEncoding.EncodedLen(stride) {
			return false, errors.New("录制深度帧尺寸无效")
		}
		target := result[8+sequence*stride : 8+(sequence+1)*stride]
		if n, err := base64.StdEncoding.Decode(target, []byte(frame.Depth)); err != nil || n != stride {
			return false, errors.New("录制深度数据无效")
		}
		sequence++
		return sequence == count, nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

func (s *Service) serveRecordedDepth(w http.ResponseWriter, r *http.Request) bool {
	q := r.URL.Query()
	if !q.Has("recorded-depth") {
		return false
	}
	id := q.Get("recorded-depth")
	if len(q) == 1+len(q["case"]) && len(q["case"]) <= 1 && len(q["recorded-depth"]) == 1 && validID.MatchString(id) {
		s.mu.Lock()
		history := s.simulationHistory()
		s.mu.Unlock()
		for _, job := range history {
			if job.ID != id || job.State != "succeeded" || (job.Recording != "" && !job.VisionInjected) {
				continue
			}
			output, result := "output", job.Result
			if len(job.Cases) > 0 {
				index := simulationCaseIndex(job, q.Get("case"))
				if index < 0 {
					break
				}
				output = sceneCaseOutput(index)
				member := simulationCase(job, index)
				var err error
				result, err = s.simulationOutputResult(&member, output)
				if err != nil {
					break
				}
			} else if q.Has("case") {
				break
			}
			steps, ok := result["steps"].(float64)
			digest, hasDigest := result["recording_sha256"].(string)
			if !ok || !hasDigest || !regexpHash(digest) {
				break
			}
			data, err := files.Read(s.c.DataDir, filepath.Join("simulations", id, output, "policy-input.mcap"), recordingLimit)
			if err != nil || hashBytes(data) != digest {
				break
			}
			frames, err := recordedDepth(data, int(steps))
			if err != nil {
				break
			}
			w.Header().Set("Content-Type", "application/octet-stream")
			w.Header().Set("Cache-Control", "no-store")
			_, _ = w.Write(frames)
			return true
		}
	}
	models.JSON(w, 404, map[string]string{"error": "无可用的录制深度帧"})
	return true
}
