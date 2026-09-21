"""Typed state, skill and decision records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class RobotState:
    end_effector_xyz: tuple[float, float, float]
    target_xyz: tuple[float, float, float]
    reach_radius_m: float
    payload_kg: float
    payload_limit_kg: float
    human_distance_m: float
    collision_detected: bool
    gripper_has_object: bool

    def __post_init__(self) -> None:
        if len(self.end_effector_xyz) != 3 or len(self.target_xyz) != 3:
            raise ValueError("positions must contain exactly three coordinates")
        if self.reach_radius_m <= 0:
            raise ValueError("reach radius must be positive")
        if self.payload_kg < 0 or self.payload_limit_kg < 0:
            raise ValueError("payload values must be non-negative")
        if self.human_distance_m < 0:
            raise ValueError("human distance must be non-negative")

    def to_text(self) -> str:
        return "; ".join(
            f"{key.replace('_', ' ')}: {value}" for key, value in asdict(self).items()
        )


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    estimated_time_s: float
    estimated_energy_j: float

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError("skill name and description must not be empty")
        if self.estimated_time_s < 0 or self.estimated_energy_j < 0:
            raise ValueError("time and energy estimates must be non-negative")

    def embedding_text(self) -> str:
        return f"{self.name.replace('_', ' ')}: {self.description}"


@dataclass(frozen=True)
class RobotDecision:
    choice: str
    probabilities: Mapping[str, float]
    feasible: Mapping[str, bool]
    reasons: Mapping[str, tuple[str, ...]]
    confidence: float
    entropy: float
    requires_operator: bool
