# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load two complete Langfuse CLI responses: observations and scores."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, JsonValue

from trace_ingest.loaders.langfuse.config import (
    LANGFUSE_DEFAULT_MAX_TRACES,
    LangfuseTraceLoadError,
    _describe_corpus,
    _LangfuseCorpusDescription,
)
from trace_ingest.loaders.langfuse.normalization import _normalize_v4_trace, _started_at
from trace_ingest.models import TraceSnapshot
from trace_ingest.source_links import file_source_url

if TYPE_CHECKING:
    from langfuse.api.commons.types.observation_v2 import ObservationV2
    from langfuse.api.scores_v3.types.score_v3 import ScoreV3


class _Meta(BaseModel):
    truncated: bool = Field(strict=True)
    cursor: str | None = None


class _Body(BaseModel):
    data: list[dict[str, JsonValue]]
    meta: _Meta


class _Response(BaseModel):
    status: Literal[200]
    body: _Body


def _read_cli_data(path: Path) -> list[dict[str, JsonValue]]:
    try:
        response = _Response.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise LangfuseTraceLoadError(
            f"Invalid Langfuse CLI export {path}; expected a successful --all --json response: {error}"
        ) from error
    if response.body.meta.truncated or response.body.meta.cursor:
        raise LangfuseTraceLoadError(
            f"Incomplete Langfuse CLI export {path}; increase --max-items or narrow the query and export again"
        )
    return response.body.data


@dataclass(frozen=True)
class LangfuseV4FileTraceConfig:
    """Observation and score CLI response files, with optional platform links."""

    observations_path: Path
    scores_path: Path
    max_traces: int = LANGFUSE_DEFAULT_MAX_TRACES
    base_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations_path", Path(self.observations_path))
        object.__setattr__(self, "scores_path", Path(self.scores_path))
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


class LangfuseV4FileTraceDescription(_LangfuseCorpusDescription):
    """Offline v4 corpus and score coverage."""

    observations_path: str
    scores_path: str
    export_trace_count: int
    max_traces: int
    truncated: bool
    base_url: str | None
    ignored_score_count: int


@dataclass
class LangfuseV4FileTraceLoader:
    """Join observation and score records, then reuse online normalization."""

    config: LangfuseV4FileTraceConfig
    _snapshot: TraceSnapshot = field(
        default_factory=lambda: TraceSnapshot([]), init=False, repr=False
    )
    _export_trace_count: int = field(default=0, init=False)
    _ignored_score_count: int = field(default=0, init=False)

    def load(self) -> TraceSnapshot:
        from langfuse.api.commons.types.observation_v2 import ObservationV2
        from langfuse.api.core.pydantic_utilities import parse_obj_as
        from langfuse.api.scores_v3.types.score_v3 import ScoreV3

        try:
            observations = parse_obj_as(
                list[ObservationV2], _read_cli_data(self.config.observations_path)
            )
            scores = parse_obj_as(list[ScoreV3], _read_cli_data(self.config.scores_path))
        except ValueError as error:
            raise LangfuseTraceLoadError(f"Invalid Langfuse export records: {error}") from error
        groups: dict[str, list[ObservationV2]] = defaultdict(list)
        trace_ids_by_span: dict[str, set[str]] = defaultdict(set)
        for observation in observations:
            if not observation.id or not observation.trace_id or not observation.project_id:
                raise LangfuseTraceLoadError(
                    "observation id, traceId, and projectId must be nonempty"
                )
            groups[observation.trace_id].append(observation)
            trace_ids_by_span[observation.id].add(observation.trace_id)
        projects = {obs.project_id for obs in observations}
        if len(projects) > 1:
            raise LangfuseTraceLoadError("Langfuse export contains multiple projects")
        by_trace: dict[str, list[ScoreV3]] = defaultdict(list)
        score_ids: set[str] = set()
        ignored = 0
        for score in scores:
            if score.id in score_ids:
                raise LangfuseTraceLoadError(f"duplicate score id {score.id!r}")
            score_ids.add(score.id)
            if projects and score.project_id not in projects:
                raise LangfuseTraceLoadError("score projectId does not match observations")
            subject = score.subject
            trace_id = None
            if subject is not None and subject.kind == "trace":
                trace_id = subject.id
            elif subject is not None and subject.kind == "observation":
                trace_id = subject.trace_id
                candidates = trace_ids_by_span.get(subject.id, set())
                if trace_id is None and len(candidates) == 1:
                    trace_id = next(iter(candidates))
            if trace_id in groups:
                by_trace[trace_id].append(score)
            else:
                ignored += 1
        traces = []
        for trace_id, items in groups.items():
            trace = _normalize_v4_trace(trace_id, items, by_trace[trace_id], self.config.base_url)
            if trace.source_url is None:
                trace.source_url = file_source_url(self.config.observations_path)
            trace.attributes["export_path"] = str(self.config.observations_path.resolve())
            traces.append((min(_started_at(obs) for obs in items), trace_id, trace))
        self._snapshot = TraceSnapshot(
            trace for _, _, trace in sorted(traces)[-self.config.max_traces :]
        )
        self._export_trace_count = len(groups)
        self._ignored_score_count = ignored
        return self._snapshot

    def describe(self) -> LangfuseV4FileTraceDescription:
        return {
            **_describe_corpus(
                f"langfuse-v4-export:{self.config.observations_path.resolve()}", self._snapshot
            ),
            "observations_path": str(self.config.observations_path.resolve()),
            "scores_path": str(self.config.scores_path.resolve()),
            "export_trace_count": self._export_trace_count,
            "max_traces": self.config.max_traces,
            "truncated": self._export_trace_count > self._snapshot.trace_count,
            "base_url": self.config.base_url,
            "ignored_score_count": self._ignored_score_count,
        }
