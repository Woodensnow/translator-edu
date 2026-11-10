"""Transcript segment shared by the engine and the streaming session."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
