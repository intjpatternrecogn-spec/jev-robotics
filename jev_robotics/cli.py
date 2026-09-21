"""Run the JEV-inspired robot decision layer on a synthetic JSON scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .planner import RobotDecisionEngine
from .types import RobotState, Skill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.scenario.read_text(encoding="utf-8"))
    state_payload = payload["state"]
    state_payload["end_effector_xyz"] = tuple(state_payload["end_effector_xyz"])
    state_payload["target_xyz"] = tuple(state_payload["target_xyz"])
    decision = RobotDecisionEngine().decide(
        RobotState(**state_payload),
        payload["question"],
        [Skill(**skill) for skill in payload["skills"]],
    )
    print(json.dumps(decision.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
