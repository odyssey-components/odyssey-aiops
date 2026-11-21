package go1

import "fmt"

// IDs and values are versioned together; historical task snapshots are untouched.
func fixedSceneDefinitions() []Scene {
	definitions := []Scene{
		{Title: "空旷场地", Kind: "flat", Parameters: map[string]float64{"forward_m_s": .3, "yaw_rate_rad_s": 0}},
		{Title: "拐弯", Kind: "turn", Parameters: map[string]float64{"angle_deg": 90, "road_width_m": .8, "distance_m": 1.2}},
		{Title: "窄路掉头", Kind: "uturn", Parameters: map[string]float64{"road_width_m": .8, "turn_at_m": 1.2}},
		{Title: "越过小坡", Kind: "slope", Parameters: map[string]float64{"angle_deg": 8, "length_m": 1.2}},
		{Title: "俯身穿过横梁", Kind: "duck", Parameters: map[string]float64{"distance_m": 1.2, "height_m": .28}},
		{Title: "跳过低横梁", Kind: "jump", Parameters: map[string]float64{"distance_m": 1.2, "height_m": .20}},
	}
	for i := range definitions {
		scene := &definitions[i]
		scene.ID = fmt.Sprintf("a10f5001-0001-4000-8000-%012d", i+1)
		scene.Format = "go1-scene/v1"
		scene.CreatedAt = "2026-11-19T00:00:00Z"
		scene.Criteria = defaultCriteria(scene.Kind)
	}
	return definitions
}

func (s *Service) fixedScenes() []Scene {
	return fixedSceneDefinitions()
}

func fixedScene(id string) (Scene, bool) {
	for _, scene := range fixedSceneDefinitions() {
		if scene.ID == id {
			return scene, true
		}
	}
	return Scene{}, false
}
