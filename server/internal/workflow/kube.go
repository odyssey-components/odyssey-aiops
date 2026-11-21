package workflow

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const account = "/var/run/secrets/kubernetes.io/serviceaccount/"
const trainTemplate = "aiops-train-v1"
const simulationTemplate = "aiops-simulate-v1"

type apiError struct {
	method string
	status int
}

func (e apiError) Error() string { return fmt.Sprintf("Workflow API %s: HTTP %d", e.method, e.status) }

func retryable(err error) bool {
	var response apiError
	return !errors.As(err, &response) || response.status == 429 || response.status >= 500
}

type kube struct {
	client    *http.Client
	base      string
	namespace string
}

type execution struct {
	Metadata struct {
		Name string
		UID  string
	} `json:"metadata"`
	Status struct {
		Phase   string
		Message string
	} `json:"status"`
}

func inCluster() (kube, error) {
	ca, err := os.ReadFile(account + "ca.crt")
	if err != nil {
		return kube{}, err
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(ca) {
		return kube{}, errors.New("invalid cluster CA")
	}
	ns, err := os.ReadFile(account + "namespace")
	if err != nil {
		return kube{}, err
	}
	namespace := strings.TrimSpace(string(ns))
	if namespace != "aiops" {
		return kube{}, errors.New("Workflow mode requires the aiops namespace")
	}
	client := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}}}
	return kube{client: client, base: "https://kubernetes.default.svc/apis/argoproj.io/v1alpha1/namespaces/" + namespace, namespace: namespace}, nil
}

func (k kube) call(ctx context.Context, method, path string, body any, response any) error {
	var buf io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return err
		}
		buf = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, k.base+path, buf)
	if err != nil {
		return err
	}
	token, err := os.ReadFile(account + "token")
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(string(token)))
	content := "application/json"
	if method == "PATCH" {
		content = "application/merge-patch+json"
	}
	req.Header.Set("Content-Type", content)
	resp, err := k.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return apiError{method, resp.StatusCode}
	}
	if response != nil {
		return json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(response)
	}
	return nil
}

func (k kube) submit(ctx context.Context, name, template, id string, timeout time.Duration) (execution, error) {
	body := map[string]any{
		"apiVersion": "argoproj.io/v1alpha1", "kind": "Workflow",
		"metadata": map[string]any{"name": name, "labels": map[string]string{"app.kubernetes.io/name": "aiops"}},
		"spec": map[string]any{
			"suspend": true, "workflowTemplateRef": map[string]string{"name": template},
			"activeDeadlineSeconds": int(timeout.Seconds()), "arguments": map[string]any{"parameters": []map[string]string{{"name": "task-id", "value": id}}},
		},
	}
	var wf execution
	err := k.call(ctx, "POST", "/workflows", body, &wf)
	return wf, err
}

func (k kube) update(ctx context.Context, name, uid string, spec map[string]any) error {
	return k.call(ctx, "PATCH", "/workflows/"+name, map[string]any{"metadata": map[string]string{"uid": uid}, "spec": spec}, nil)
}
