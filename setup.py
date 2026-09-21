"""Compatibility installer for offline environments with older setuptools."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


setup(
    name="jev-robotics",
    version="0.1.0",
    description="JEV-inspired dynamic candidate decision layer for robotics",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["jev_robotics", "jev_robotics.*"]),
    python_requires=">=3.10",
    entry_points={"console_scripts": ["jev-robotics=jev_robotics.cli:main"]},
)
