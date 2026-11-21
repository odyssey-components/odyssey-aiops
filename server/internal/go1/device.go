package go1

func selectedDevice(values []string) string {
	if len(values) > 0 && values[0] != "" {
		return values[0]
	}
	return "cpu"
}

// DetectTrainingEnvironment runs once per server process, before the API is served.
func (s *Service) DetectTrainingEnvironment() {
	s.environmentOnce.Do(func() {
		modules := []string{
			"aiops_go1_training.runtime.local",
			"aiops_go1_training.runtime.upstream",
			"aiops_go1_training.vision.train",
		}
		environments := make(map[string]map[string]any, len(modules))
		for _, module := range modules {
			if s.c.Workflow {
				environments[module] = map[string]any{"device": "cpu", "ready": true, "devices": []string{"Argo Workflow CPU"}, "errors": []string{}, "source_commit": Commit}
				continue
			}
			environments[module] = preferredEnvironment(func(device string) map[string]any {
				return s.probeModule(module, device)
			})
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		s.trainingEnvironments = environments
		s.environment = environments[modules[0]]
	})
}

func preferredEnvironment(probe func(string) map[string]any) map[string]any {
	var result map[string]any
	for _, device := range []string{"cuda", "rocm", "cpu"} {
		result = probe(device)
		if result["ready"] == true {
			break
		}
	}
	return result
}
