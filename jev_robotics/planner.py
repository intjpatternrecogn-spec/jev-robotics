"""Constraint-aware orchestration for bounded robot skill selection."""

from __future__ import annotations

import math
from typing import Sequence

from .constraints import evaluate_skill, task_prior
from .model import CandidateDecisionModel
from .types import RobotDecision, RobotState, Skill


class RobotDecisionEngine:
    def __init__(
        self,
        confidence_threshold: float = 0.40,
        normalised_entropy_threshold: float = 0.92,
        model: CandidateDecisionModel | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.normalised_entropy_threshold = normalised_entropy_threshold
        self.model = model or CandidateDecisionModel()

    def decide(self, state: RobotState, question: str, skills: Sequence[Skill]) -> RobotDecision:
        if not skills:
            raise ValueError("at least one skill is required")

        checks = [evaluate_skill(state, skill) for skill in skills]
        feasible = [allowed for allowed, _ in checks]
        reasons = [reason for _, reason in checks]
        names = [skill.name for skill in skills]
        if not any(feasible):
            return RobotDecision(
                choice="request_operator",
                probabilities=dict.fromkeys(names, 0.0),
                feasible=dict(zip(names, feasible)),
                reasons=dict(zip(names, reasons)),
                confidence=0.0,
                entropy=0.0,
                requires_operator=True,
            )

        max_time = max(skill.estimated_time_s for skill in skills)
        max_energy = max(skill.estimated_energy_j for skill in skills)
        output = self.model.score(
            state.to_text(),
            question,
            [skill.embedding_text() for skill in skills],
            [task_prior(state, skill, max_time, max_energy) for skill in skills],
            feasible,
        )
        probabilities = dict(zip(names, output.probabilities))
        choice = max(probabilities, key=probabilities.get)
        confidence = probabilities[choice]
        feasible_count = sum(feasible)
        normalised_entropy = output.entropy / math.log(feasible_count) if feasible_count > 1 else 0.0
        uncertain = (
            confidence < self.confidence_threshold
            or normalised_entropy > self.normalised_entropy_threshold
        )

        return RobotDecision(
            choice="request_operator" if uncertain else choice,
            probabilities=probabilities,
            feasible=dict(zip(names, feasible)),
            reasons=dict(zip(names, reasons)),
            confidence=confidence,
            entropy=output.entropy,
            requires_operator=uncertain,
        )
