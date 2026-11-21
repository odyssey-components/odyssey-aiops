package training

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"github.com/pelletier/go-toml/v2"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"
	"unicode/utf8"
)

func directory(path string) error {
	if err := os.MkdirAll(path, 0700); err != nil {
		return err
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return err
	}
	abs, err := filepath.Abs(path)
	if err != nil || resolved != abs {
		return errors.New("目录不可为符号链接")
	}
	st, err := os.Lstat(path)
	if err != nil || !st.IsDir() {
		return errors.New("目录无效")
	}
	return nil
}
func bounded(path string, limit int64) ([]byte, error) {
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil || resolved != path {
		return nil, errors.New("文件路径无效")
	}
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	defer f.Close()
	st, err := f.Stat()
	if err != nil || !st.Mode().IsRegular() || st.Size() > limit {
		return nil, errors.New("文件无效或过大")
	}
	current, err := os.Lstat(path)
	if err != nil || !os.SameFile(st, current) {
		return nil, errors.New("文件已更改")
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if int64(len(b)) > limit {
		return nil, errors.New("文件过大")
	}
	return b, err
}
func atomicTOML(path string, v any) error {
	b, err := toml.Marshal(v)
	if err != nil {
		return err
	}
	return atomicWrite(path, b)
}

func atomicWrite(path string, b []byte) error {
	f, err := os.CreateTemp(filepath.Dir(path), ".record-*")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	if err = f.Chmod(0600); err == nil {
		_, err = f.Write(b)
	}
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(name, path)
}

type datasetRecord struct {
	ID, Format, Lesson, Sha256, Title, Description, Generator string
	CreatedAt                                                 string `toml:"created_at"`
	Seed                                                      int
	Rows, Columns                                             int
	WallValue                                                 int                                   `toml:"wall_value"`
	RoadValue                                                 int                                   `toml:"road_value"`
	GoalValue                                                 int                                   `toml:"goal_value"`
	MapFormat                                                 string                                `toml:"map_format"`
	MaxSteps                                                  int                                   `toml:"max_steps"`
	Counts                                                    struct{ Train, Validation, Test int } `toml:"split_counts"`
}

func (s *Service) dataset(id string) (DatasetInfo, error) {
	var info DatasetInfo
	if !idPattern.MatchString(id) {
		return info, errors.New("地图组标识无效")
	}
	maps, err := bounded(filepath.Join(s.datasets, id+".jsonl"), 4<<20)
	if err != nil {
		return info, err
	}
	meta, err := bounded(filepath.Join(s.datasets, id+".toml"), 16384)
	if err != nil {
		return info, err
	}
	var d datasetRecord
	if !utf8.Valid(meta) || !utf8.Valid(maps) || toml.Unmarshal(meta, &d) != nil {
		return info, errors.New("地图组格式无效")
	}
	sha := sha256.Sum256(maps)
	steps := 100
	if d.Lesson == "local-map-v1" || d.Lesson == "memory-map-v1" {
		steps = 200
	}
	generator := "lcg-corridor-7/v1"
	counts := d.Counts.Train >= 16 && d.Counts.Train <= 512 && d.Counts.Validation == 64 && d.Counts.Test == 100
	if d.Lesson == "fixed-map-v1" {
		generator = "fixed-map/v1"
		counts = d.Counts.Train == 1 && d.Counts.Validation == 0 && d.Counts.Test == 0
	}
	_, dateErr := time.Parse(time.RFC3339Nano, d.CreatedAt)
	if d.ID != id || d.Format != "odymaze-maps/v2" || !lesson(d.Lesson) || d.Sha256 != hex.EncodeToString(sha[:]) || !counts || d.Rows != 7 || d.Columns != 7 || d.WallValue != -1 || d.RoadValue != 1 || d.GoalValue != 0 || d.MapFormat != "numeric-grid/v1" || d.MaxSteps != steps || d.Generator != generator || strings.TrimSpace(d.Title) == "" || utf8.RuneCountInString(d.Title) > 80 || utf8.RuneCountInString(d.Description) > 500 || dateErr != nil || d.Seed < 0 || d.Seed > 2147483647 {
		return info, errors.New("地图组规格或校验和无效")
	}
	return DatasetInfo{id, d.Lesson, d.Seed, d.Counts.Train, d.Counts.Validation, d.Counts.Test, d.Sha256, d.CreatedAt, d.Title, d.Description}, nil
}
