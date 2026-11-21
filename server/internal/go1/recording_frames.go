package go1

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"path/filepath"
	"strconv"

	"github.com/foxglove/mcap/go/mcap"
	"odyssey/aiops/internal/files"
	"odyssey/aiops/internal/models"
)

func recordingFrame(data []byte, step int) (map[string]any, error) {
	var result map[string]any
	err := visitRecording(data, func(camera map[string]any, message *mcap.Message) (bool, error) {
		if int(message.Sequence) != step-1 {
			return false, nil
		}
		var frame map[string]any
		if err := json.Unmarshal(message.Data, &frame); err != nil {
			return false, err
		}
		result = map[string]any{"step": step, "time": float64(message.LogTime) / 1e9, "camera": camera, "frame": frame}
		return true, nil
	})
	return result, err
}

func visitRecording(data []byte, visit func(map[string]any, *mcap.Message) (bool, error)) error {
	lexer, err := mcap.NewLexer(bytes.NewReader(data), &mcap.LexerOptions{
		ValidateChunkCRCs: true, MaxDecompressedChunkSize: 4 << 20, MaxRecordSize: 4 << 20,
	})
	if err != nil {
		return err
	}
	defer lexer.Close()
	channels := map[uint16]bool{}
	var camera map[string]any
	var buffer []byte
	for {
		token, payload, err := lexer.Next(buffer)
		if err != nil {
			return err
		}
		buffer = payload
		switch token {
		case mcap.TokenMetadata:
			meta, err := mcap.ParseMetadata(payload)
			if err != nil {
				return err
			}
			if meta.Name == "go1-policy-input/v1" {
				var contract struct {
					Camera map[string]any `json:"camera"`
				}
				if err := json.Unmarshal([]byte(meta.Metadata["contract"]), &contract); err != nil {
					return err
				}
				camera = contract.Camera
			}
		case mcap.TokenChannel:
			channel, err := mcap.ParseChannel(payload)
			if err != nil {
				return err
			}
			channels[channel.ID] = channel.Topic == "/go1/policy_step" && channel.MessageEncoding == "json"
		case mcap.TokenMessage:
			message, err := mcap.ParseMessage(payload)
			if err != nil {
				return err
			}
			if !channels[message.ChannelID] {
				continue
			}
			if message.LogTime != uint64(message.Sequence)*20_000_000 {
				return errors.New("录制帧时间无效")
			}
			done, err := visit(camera, message)
			if done || err != nil {
				return err
			}
		}
	}
}

func recordingMetrics(data []byte, expected int) (map[string]any, error) {
	frames, depth := 0, 0
	var previous, maxGap uint64
	err := visitRecording(data, func(_ map[string]any, message *mcap.Message) (bool, error) {
		if frames >= expected || (frames > 0 && message.LogTime <= previous) {
			return false, errors.New("录制帧超限或时间顺序无效")
		}
		var frame struct {
			Depth string `json:"depth_f32le_base64"`
		}
		if err := json.Unmarshal(message.Data, &frame); err != nil {
			return false, err
		}
		if frame.Depth != "" {
			depth++
		}
		if frames > 0 && message.LogTime-previous > maxGap {
			maxGap = message.LogTime - previous
		}
		previous = message.LogTime
		frames++
		return false, nil
	})
	if !errors.Is(err, io.EOF) {
		return nil, err
	}
	var gap any
	if frames > 1 {
		gap = float64(maxGap) / 1e6
	}
	topics := 0
	if frames > 0 {
		topics = 1
	}
	return map[string]any{"frames": frames, "expectedFrames": expected, "depthFrames": depth, "observedTopics": topics, "requiredTopics": 1, "maxGapMs": gap}, nil
}

func (s *Service) serveRecordingFrame(w http.ResponseWriter, r *http.Request) bool {
	q := r.URL.Query()
	if !q.Has("recording-frame") {
		return false
	}
	id, caseID := q.Get("recording-frame"), q.Get("case")
	step, err := strconv.Atoi(q.Get("step"))
	metrics := q.Get("step") == "metrics"
	if len(q["recording-frame"]) == 1 && len(q["step"]) == 1 && len(q["case"]) <= 1 &&
		len(q) == 2+len(q["case"]) && validID.MatchString(id) && (metrics || (err == nil && step >= 1 && step <= 500)) {
		s.mu.Lock()
		history := s.simulationHistory()
		s.mu.Unlock()
		for _, job := range history {
			if job.ID != id || job.State != "succeeded" || (job.Recording != "" && !job.VisionInjected) {
				continue
			}
			output := "output"
			result := job.Result
			if len(job.Cases) > 0 {
				index := simulationCaseIndex(job, caseID)
				if index < 0 {
					break
				}
				member := simulationCase(job, index)
				output = sceneCaseOutput(index)
				result, err = s.simulationOutputResult(&member, output)
				if err != nil {
					break
				}
			} else if q.Has("case") {
				break
			}
			count, ok := result["steps"].(float64)
			digest, hasDigest := result["recording_sha256"].(string)
			if !ok || count < 1 || count > 500 || float64(step) > count || !hasDigest || !regexpHash(digest) {
				break
			}
			data, err := files.Read(s.c.DataDir, filepath.Join("simulations", id, output, "policy-input.mcap"), recordingLimit)
			if err != nil || hashBytes(data) != digest {
				break
			}
			if metrics {
				if value, err := recordingMetrics(data, int(count)); err == nil {
					models.JSON(w, 200, value)
					return true
				}
				break
			}
			if frame, err := recordingFrame(data, step); err == nil {
				models.JSON(w, 200, frame)
				return true
			}
			break
		}
	}
	models.JSON(w, 400, map[string]string{"error": "无法读取 MCAP 录制帧"})
	return true
}
