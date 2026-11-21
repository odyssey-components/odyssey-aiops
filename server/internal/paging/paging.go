package paging

import (
	"errors"
	"net/url"
	"strconv"
)

type Request struct{ Page, Size int }
type Info struct {
	Page  int `json:"page"`
	Size  int `json:"size"`
	Total int `json:"total"`
}

func Parse(q url.Values) (Request, error) {
	result := Request{Page: 1, Size: 20}
	for key, value := range map[string]*int{"page": &result.Page, "pageSize": &result.Size} {
		if !q.Has(key) {
			continue
		}
		n, err := strconv.Atoi(q.Get(key))
		if err != nil || n < 1 || n > 1000000 || len(q[key]) != 1 {
			return Request{}, errors.New("分页参数无效")
		}
		*value = n
	}
	if result.Size > 100 {
		return Request{}, errors.New("每页最多100条")
	}
	return result, nil
}

func Slice[T any](rows []T, request Request) ([]T, Info) {
	page := min(request.Page, max(1, (len(rows)+request.Size-1)/request.Size))
	start := (page - 1) * request.Size
	return rows[start:min(start+request.Size, len(rows))], Info{Page: page, Size: request.Size, Total: len(rows)}
}
