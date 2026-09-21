"""Candidate-embedding decision head implemented with the Python standard library."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Sequence


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exponential = [math.exp(value - maximum) for value in values]
    denominator = sum(exponential)
    return [value / denominator for value in exponential]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def unit(vector: Sequence[float]) -> list[float]:
    magnitude = math.sqrt(dot(vector, vector)) or 1.0
    return [value / magnitude for value in vector]


@dataclass(frozen=True)
class ModelOutput:
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    attention: tuple[float, ...]
    entropy: float


class HashTextEncoder:
    """Small deterministic adapter that keeps the example offline and auditable."""

    def __init__(self, dimension: int = 48) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    @staticmethod
    def tokenise(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()) or ["empty"]

    def token_vector(self, token: str) -> list[float]:
        raw: list[float] = []
        index = 0
        while len(raw) < self.dimension:
            block = hashlib.blake2b(f"{token}:{index}".encode(), digest_size=32).digest()
            raw.extend(byte / 127.5 - 1.0 for byte in block)
            index += 1
        return unit(raw[: self.dimension])

    def encode_tokens(self, text: str) -> list[list[float]]:
        return [self.token_vector(token) for token in self.tokenise(text)]

    def encode(self, text: str) -> list[float]:
        vectors = self.encode_tokens(text)
        return unit([sum(column) / len(vectors) for column in zip(*vectors)])


class CandidateDecisionModel:
    """Encode state once and score independently encoded skills in parallel."""

    def __init__(self, dimension: int = 48, temperature: float = 0.24) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.encoder = HashTextEncoder(dimension)
        self.temperature = temperature

    def decision_embedding(self, state: str, question: str) -> tuple[list[float], list[float]]:
        memory = self.encoder.encode_tokens(state)
        query = self.encoder.encode(question)
        attention = softmax([dot(query, token) * 2.0 for token in memory])
        attended = [
            sum(weight * token[index] for weight, token in zip(attention, memory))
            for index in range(self.encoder.dimension)
        ]
        return unit([q + state_value for q, state_value in zip(query, attended)]), attention

    def score(
        self,
        state: str,
        question: str,
        candidates: Sequence[str],
        priors: Sequence[float],
        feasible: Sequence[bool],
    ) -> ModelOutput:
        if not candidates:
            raise ValueError("at least one candidate is required")
        if not (len(candidates) == len(priors) == len(feasible)):
            raise ValueError("candidate metadata lengths must match")
        if not any(feasible):
            raise ValueError("at least one candidate must be feasible")

        decision, attention = self.decision_embedding(state, question)
        options = [self.encoder.encode(candidate) for candidate in candidates]
        logits = [
            (dot(decision, option) + prior) / self.temperature if allowed else -1e9
            for option, prior, allowed in zip(options, priors, feasible)
        ]
        probabilities = softmax(logits)
        entropy = -sum(probability * math.log(probability + 1e-12) for probability in probabilities)
        return ModelOutput(tuple(logits), tuple(probabilities), tuple(attention), entropy)
