# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared compass-signal interfaces and models."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field, model_validator

from insight_agent.traces import TraceSnapshot


class Problem(BaseModel):
    """A potential issue surfaced by one Compass Signal."""

    description: str = Field(min_length=1)
    supporting_trace_ids: tuple[str, ...] = Field(
        min_length=1,
        description="Selected examples cited by the signal to support this problem; used for code validation and always included in candidate_trace_ids.",
    )
    candidate_trace_ids: tuple[str, ...] = Field(
        default=(),
        description=(
            "All distinct trace IDs identified by the signal for this problem, including supporting_trace_ids. "
            "When no broader list is supplied, this contains the supporting examples. "
            "These are leads for compilation, not confirmed matches for the final insight or an exhaustive corpus scan."
        ),
    )

    @model_validator(mode="after")
    def include_supporting_candidates(self) -> Problem:
        """Keep every cited example in the candidate set without changing the review sample."""
        self.candidate_trace_ids = tuple(
            dict.fromkeys((*self.candidate_trace_ids, *self.supporting_trace_ids))
        )
        return self


class CompassSignalResult(BaseModel):
    signal_name: str = Field(min_length=1)
    problems: tuple[Problem, ...]
    artifacts: Any = None
    finding_count: int = Field(
        default=0, ge=0, description="Observations before candidate filtering"
    )
    skip_reason: str | None = None
    limitations: tuple[str, ...] = ()


class CompassSignal(Protocol):
    @property
    def name(self) -> str: ...

    def check_prerequisites(self, snapshot: TraceSnapshot) -> str | None:
        """Raise for invalid settings; return a reason for unavailable prerequisites."""

    async def analyze(self, snapshot: TraceSnapshot) -> CompassSignalResult:
        """Analyze after check_prerequisites returns no skip reason."""
        ...


__all__ = ["CompassSignal", "CompassSignalResult", "Problem"]
