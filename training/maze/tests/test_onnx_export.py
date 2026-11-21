"""ONNX export preserves each lesson's selected JSON policy."""

import hashlib
import tempfile
import tomllib
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast
from conftest import workspace_root

import numpy as np
import onnx
import onnxruntime
import torch
from numpy.typing import NDArray
from aiops_maze_training.lessons.fixed_map_v1.environment import initial_state, load_maze, step
from aiops_maze_training.lessons.fixed_map_v1 import model as fixed
from aiops_maze_training.lessons.local_map_v1 import model as local
from aiops_maze_training.lessons.memory_map_v1 import model as memory
from aiops_maze_training.lessons.memory_map_v1.environment import encode, generate_maze, samples
from aiops_maze_training.lessons.random_map_v1 import model as random_map
from aiops_maze_training.lessons.random_map_v1.environment import generate_maze as random_maze
from aiops_maze_training.common.onnx_export import export_onnx
from aiops_maze_training.common.types import Metadata, Policy

ROOT_TMP = workspace_root() / "tmp"
META: Metadata = {"seed": 7, "epochs": 0, "samples": 0, "accuracy": 0.0, "success": False}


class Session(Protocol):
    def run(self, names: list[str] | None, feeds: dict[str, object]) -> list[NDArray[np.float32]]: ...


def session(path: Path) -> Session:
    return cast(Session, onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"]))


def policies() -> list[tuple[torch.nn.Module, Policy, int]]:
    a, b, c, d = fixed.make_model(18), random_map.make_model(), local.make_model(), memory.make_model()
    return [
        (a, fixed.export_model(a, load_maze(), META), 18),
        (b, random_map.export_model(b, random_maze(7), META), 151),
        (c, local.export_model(c, generate_maze(7), META), 249),
        (d, memory.export_model(d, generate_maze(7), META), 40),
    ]


class OnnxExportTests(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_TMP.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_all_lessons_preserve_logits_dynamic_batch_and_contract(self) -> None:
        for index, (model, policy, size) in enumerate(policies()):
            with self.subTest(lesson=index):
                output = export_onnx(policy, self.root / str(index) / "model.onnx")
                graph = onnx.load_model_from_string(output.read_bytes())
                cast(Callable[..., None], onnx.checker.check_model)(graph)
                self.assertTrue(all(not value.external_data for value in graph.graph.initializer))
                manifest = tomllib.loads(output.with_suffix(".manifest.toml").read_text())
                self.assertEqual(manifest["schema"], "odymaze-onnx/v1")
                self.assertEqual(manifest["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
                self.assertEqual(
                    manifest["inputs"][0],
                    {"name": "observations", "dtype": "float32", "shape": {"batch": "batch", "width": size}},
                )
                inputs = torch.randn(2, size)
                feeds: dict[str, object] = {"observations": inputs.numpy()}
                with torch.no_grad():
                    if isinstance(model, memory.MemoryModel):
                        hidden = torch.randn(2, 32)
                        feeds["hidden"] = hidden.numpy()
                        sequence, next_hidden = model.gru(inputs.unsqueeze(1), hidden.unsqueeze(0))
                        expected = [model.output(sequence[:, 0]), next_hidden[0]]
                    else:
                        expected = [model(inputs)]
                actual = session(output).run(None, feeds)
                for result, reference in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(result, reference.numpy(), rtol=1e-5, atol=1e-6)
                    self.assertTrue(np.isfinite(result).all())

    def test_gru_carries_hidden_through_expert_and_model_rollouts_and_resets(self) -> None:
        model = memory.make_model()
        maze = generate_maze(7)
        runtime = session(export_onnx(memory.export_model(model, maze, META), self.root / "model.onnx"))
        for expert in (True, False):
            state, previous = initial_state(maze), -1
            hidden = torch.zeros(1, 1, 32)
            exported_hidden = np.zeros((1, 32), dtype=np.float32)
            expert_actions = [row["action"] for row in samples(maze)]
            for tick in range(len(expert_actions) if expert else 40):
                observation = torch.tensor(encode(maze, state, previous), dtype=torch.float32)
                with torch.no_grad():
                    logits, hidden = model.step(observation, hidden)
                actual, exported_hidden = runtime.run(
                    None, {"observations": observation.numpy()[None], "hidden": exported_hidden}
                )
                np.testing.assert_allclose(actual, logits.numpy()[None], atol=1e-6, rtol=1e-5)
                np.testing.assert_allclose(exported_hidden, hidden.numpy()[0], atol=1e-6, rtol=1e-5)
                action = expert_actions[tick] if expert else int(logits.argmax())
                state, previous = step(maze, state, action), action

    def test_rejects_invalid_policies_without_creating_outputs(self) -> None:
        _, original, _ = policies()[0]
        changes: list[tuple[str, object]] = [
            ("format", "bad"),
            ("lesson", "bad"),
            ("encoding", "bad"),
            ("actions", []),
            ("metadata", {}),
            ("layers", None),
            ("layers", [None, None]),
        ]
        for field, value in changes:
            policy = cast(Policy, {**original, field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                export_onnx(policy, self.root / "model.onnx")
        malformed_values: list[object] = [None, [], {"format": "unknown"}]
        for malformed in malformed_values:
            with self.assertRaises(ValueError):
                export_onnx(cast(Policy, malformed), self.root / "model.onnx")
        original["layers"][0]["weight"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            export_onnx(original, self.root / "model.onnx")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_rejects_wrong_shapes_unbounded_weights_and_incompatible_maze(self) -> None:
        for invalid in ("shape", "bound", "maze", "gru", "metadata"):
            _, policy, _ = policies()[0]
            if invalid == "shape":
                policy["layers"][0]["weight"][0].pop()
            elif invalid == "bound":
                policy["layers"][0]["bias"][0] = 100.01
            elif invalid == "maze":
                policy["maze"] = {**policy["maze"], "max_steps": 999}
            elif invalid == "gru":
                _, policy, _ = policies()[3]
                assert "gru" in policy
                policy["gru"]["weight_hh"][0].pop()
            else:
                policy["metadata"] = {**META, "test_success_rate": 2.0}
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                export_onnx(policy, self.root / "model.onnx")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_model_manifest_and_symlinks_are_protected(self) -> None:
        _, policy, _ = policies()[0]
        output = self.root / "model.onnx"
        for occupied in (output, output.with_suffix(".manifest.toml")):
            for symbolic in (False, True):
                if symbolic:
                    occupied.symlink_to(self.root / "missing")
                else:
                    occupied.write_text("keep")
                with self.assertRaises(FileExistsError):
                    export_onnx(policy, output)
                self.assertTrue(occupied.is_symlink() if symbolic else occupied.read_text() == "keep")
                occupied.unlink()

    def test_legacy_fixed_policy_without_lesson_is_supported(self) -> None:
        _, policy, _ = policies()[0]
        del policy["lesson"]
        export_onnx(policy, self.root / "legacy.onnx")
