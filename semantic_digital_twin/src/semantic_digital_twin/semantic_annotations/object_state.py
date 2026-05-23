"""
Dynamic object state representation for the Semantic Digital Twin.

This module provides semantic state annotations used for:
- vision-language model predictions
- physical reasoning
- temporal state tracking
- neuro-symbolic fusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


class FillState(str, Enum):
    EMPTY = "empty"
    FILLED = "filled"
    FULL = "full"


class CutState(str, Enum):
    CUT = "cut"
    UNCUT = "uncut"


@dataclass(eq=False)
class ObjectState(SemanticAnnotation):
    """
    Semantic annotation tracking dynamic object states.
    Supports neuro-symbolic reasoning and VLM integration.
    """

    # Object this state belongs to
    target: SemanticAnnotation = field(kw_only=True)

    # Semantic states
    fill_state: Optional[FillState] = field(default=None, kw_only=True)
    cut_state: Optional[CutState] = field(default=None, kw_only=True)

    # Reasoning metadata
    confidence: float = field(default=1.0, kw_only=True)

    # Source of state estimate
    # Example: "CLIP", "SDT", "FusionEngine"
    source: str = field(default="SDT", kw_only=True)

    # Timestamp of observation/update
    timestamp: Optional[float] = field(default=None, kw_only=True)

    @property
    def is_fill(self) -> bool:
        return self.fill_state == FillState.FULL

    @property
    def is_cut(self) -> bool:
        return self.cut_state == CutState.CUT

    @property
    def has_state(self) -> bool:
        return self.fill_state is not None or self.cut_state is not None
