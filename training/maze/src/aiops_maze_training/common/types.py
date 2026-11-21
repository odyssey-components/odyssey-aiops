"""The JSON data exchanged by the lessons, training report, and web model loader."""

from typing import NotRequired, TypedDict


class Position(TypedDict):
    x: int
    y: int
    direction: int


class Maze(TypedDict):
    id: str
    rows: list[str]
    start: Position
    max_steps: int


class Memory(TypedDict):
    known: list[str]
    visited: list[int]


class Sample(TypedDict):
    previous_action: NotRequired[int]
    memory: NotRequired[Memory]
    maze_id: NotRequired[str]
    state: dict[str, int | str]
    observation: list[float]
    action: int


class Evaluation(TypedDict):
    accuracy: float
    success: bool
    steps: int
    actions: list[int]


class Metadata(TypedDict):
    test_success_rate: NotRequired[float]
    seed: int
    epochs: int
    samples: int
    accuracy: float
    success: bool


class Layer(TypedDict):
    activation: str
    weight: list[list[float]]
    bias: list[float]


class GRUParameters(TypedDict):
    hidden_size: int
    weight_ih: list[list[float]]
    weight_hh: list[list[float]]
    bias_ih: list[float]
    bias_hh: list[float]


class Policy(TypedDict):
    gru: NotRequired[GRUParameters]
    lesson: NotRequired[str]
    generator: NotRequired[str]
    format: str
    maze: Maze
    encoding: str
    actions: list[str]
    layers: list[Layer]
    metadata: Metadata


class Epoch(TypedDict):
    success_rate: NotRequired[float]
    epoch: int
    loss: float
    accuracy: float


class Report(Metadata):
    validation: NotRequired[list[dict[str, float]]]
    learning_rate: float
    history: list[Epoch]
    initial: Evaluation
    final: Evaluation
