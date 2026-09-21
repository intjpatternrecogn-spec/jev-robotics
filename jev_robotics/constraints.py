"""Inspectable constraint and cost functions for candidate robot skills."""

from __future__ import annotations

import math

from .types import RobotState, Skill


def target_distance(state: RobotState) -> float:
    return math.dist(state.end_effector_xyz, state.target_xyz)


def evaluate_skill(state: RobotState, skill: Skill) -> tuple[bool, tuple[str, ...]]:
    name = skill.name.lower()
    reasons: list[str] = []
    moving = any(word in name for word in ("move", "approach", "pick", "place", "grasp"))

    if target_distance(state) > state.reach_radius_m and moving:
        reasons.append("target is outside the reachable workspace")
    if state.payload_kg > state.payload_limit_kg and moving:
        reasons.append("payload exceeds the configured limit")
    if state.collision_detected and name not in {"hold", "stop", "retreat"}:
        reasons.append("collision monitor is active")
    if state.human_distance_m < 0.8 and name not in {"hold", "stop", "retreat"}:
        reasons.append("human separation is below 0.8 m")
    if name == "grasp" and target_distance(state) > 0.08:
        reasons.append("gripper is not close enough to grasp")
    if name == "place" and not state.gripper_has_object:
        reasons.append("gripper does not hold an object")

    return not reasons, tuple(reasons)


def task_prior(state: RobotState, skill: Skill, max_time_s: float, max_energy_j: float) -> float:
    """Balance task progress and cost using explicit, inspectable terms."""
    time_cost = skill.estimated_time_s / max(max_time_s, 1e-6)
    energy_cost = skill.estimated_energy_j / max(max_energy_j, 1e-6)
    prior = -0.14 * time_cost - 0.09 * energy_cost
    name = skill.name.lower()
    distance = target_distance(state)
    if any(word in name for word in ("move", "approach")) and distance > 0.08:
        prior += 0.34
    if name == "grasp" and distance <= 0.08 and not state.gripper_has_object:
        prior += 0.30
    if name == "place" and state.gripper_has_object:
        prior += 0.30
    if name == "hold" and not state.collision_detected and state.human_distance_m >= 0.8:
        prior -= 0.12
    return prior
