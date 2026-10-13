# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and descriptions for Langfuse trace sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trace_ingest.loaders.trace_loaders import TraceDescription

LANGFUSE_DEFAULT_MAX_TRACES = 100


class LangfuseTraceLoadError(RuntimeError):
    """A Langfuse source could not be selected or normalized safely."""


class _LangfuseCorpusDescription(TraceDescription):
    """Corpus facts shared by live and file sources."""

    observation_count: int
    unresolved_parent_count: int


class LangfuseTraceDescription(_LangfuseCorpusDescription):
    """Run metadata for a live Langfuse v3 trace query."""

    base_url: str
    filter: str | None
    from_timestamp: str
    to_timestamp: str
    max_traces: int


@dataclass(frozen=True)
class LangfuseTraceConfig:
    """Configuration for loading complete traces through the Langfuse v3 API."""

    from_timestamp: datetime
    to_timestamp: datetime
    base_url: str | None = None
    filter_string: str | None = None
    max_traces: int = LANGFUSE_DEFAULT_MAX_TRACES

    def __post_init__(self) -> None:
        validate_langfuse_time_window(self.from_timestamp, self.to_timestamp)
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


def validate_langfuse_time_window(from_timestamp: datetime, to_timestamp: datetime) -> None:
    """Require a timezone-aware, nonempty window for CLI and direct API loading."""
    for name, value in (("from_timestamp", from_timestamp), ("to_timestamp", to_timestamp)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must include a timezone")
    if to_timestamp <= from_timestamp:
        raise ValueError("to_timestamp must be after from_timestamp")


class LangfuseFileTraceDescription(_LangfuseCorpusDescription):
    """Run metadata for a local native trace-detail export."""

    export_path: str
    export_trace_count: int
    max_traces: int
    truncated: bool


@dataclass(frozen=True)
class LangfuseFileTraceConfig:
    """A native trace-detail JSON/JSONL file or directory of such files."""

    path: Path
    max_traces: int = LANGFUSE_DEFAULT_MAX_TRACES

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")
