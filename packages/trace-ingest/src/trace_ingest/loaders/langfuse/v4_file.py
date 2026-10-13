# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the documented SDK export: one trace with observations and scores per line."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from trace_ingest.loaders.langfuse.config import (
    LANGFUSE_DEFAULT_MAX_TRACES,
    LangfuseTraceLoadError,
    _describe_corpus,
    _LangfuseCorpusDescription,
)
from trace_ingest.loaders.langfuse.file_io import _records
from trace_ingest.loaders.langfuse.normalization import _normalize_v4_trace, _started_at
from trace_ingest.models import TraceSnapshot
from trace_ingest.source_links import file_source_url


class _ExportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(min_length=1)
    observations: list[dict[str, JsonValue]] = Field(min_length=1)
    scores: list[dict[str, JsonValue]]


@dataclass(frozen=True)
class LangfuseV4FileTraceConfig:
    """A JSONL SDK export with optional platform links."""

    path: Path
    max_traces: int = LANGFUSE_DEFAULT_MAX_TRACES
    base_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


class LangfuseV4FileTraceDescription(_LangfuseCorpusDescription):
    """Offline v4 corpus coverage."""

    export_path: str
    export_trace_count: int
    max_traces: int
    truncated: bool
    base_url: str | None


@dataclass
class LangfuseV4FileTraceLoader:
    """Read SDK records and reuse online normalization without contacting Langfuse."""

    config: LangfuseV4FileTraceConfig
    _snapshot: TraceSnapshot = field(
        default_factory=lambda: TraceSnapshot([]), init=False, repr=False
    )
    _export_trace_count: int = field(default=0, init=False)

    def load(self) -> TraceSnapshot:
        from langfuse.api.commons.types.observation_v2 import ObservationV2
        from langfuse.api.core.pydantic_utilities import parse_obj_as
        from langfuse.api.scores_v3.types.score_v3 import ScoreV3

        traces = []
        trace_ids: set[str] = set()
        project_ids: set[str] = set()
        for line, record in _records(self.config.path):
            try:
                exported = _ExportRecord.model_validate(record)
                if exported.trace_id in trace_ids:
                    raise ValueError(f"duplicate trace id {exported.trace_id!r}")
                trace_ids.add(exported.trace_id)
                observations = parse_obj_as(list[ObservationV2], exported.observations)
                scores = parse_obj_as(list[ScoreV3], exported.scores)
                if any(obs.trace_id != exported.trace_id for obs in observations):
                    raise ValueError("observation traceId does not match its trace")
                project_ids.update(obs.project_id for obs in observations)
                if len(project_ids) != 1 or not next(iter(project_ids)):
                    raise ValueError("export must contain a single nonempty projectId")
                observation_ids = {obs.id for obs in observations}
                score_ids: set[str] = set()
                for score in scores:
                    if score.id in score_ids or score.project_id not in project_ids:
                        raise ValueError("duplicate score id or mismatched score projectId")
                    score_ids.add(score.id)
                    subject = score.subject
                    if subject is None or not (
                        (subject.kind == "trace" and subject.id == exported.trace_id)
                        or (
                            subject.kind == "observation"
                            and (
                                subject.trace_id == exported.trace_id
                                or (subject.trace_id is None and subject.id in observation_ids)
                            )
                        )
                    ):
                        raise ValueError("score subject does not belong to its trace")
                trace = _normalize_v4_trace(
                    exported.trace_id, observations, scores, self.config.base_url
                )
                if trace.source_url is None:
                    trace.source_url = file_source_url(self.config.path)
                trace.attributes["export_path"] = str(self.config.path.resolve())
                traces.append((min(_started_at(obs) for obs in observations), trace.id, trace))
            except (ValueError, LangfuseTraceLoadError) as error:
                raise LangfuseTraceLoadError(
                    f"Invalid Langfuse export {self.config.path}:{line}: {error}"
                ) from error
        self._snapshot = TraceSnapshot(
            trace for _, _, trace in sorted(traces)[-self.config.max_traces :]
        )
        self._export_trace_count = len(traces)
        return self._snapshot

    def describe(self) -> LangfuseV4FileTraceDescription:
        return {
            **_describe_corpus(f"langfuse-v4-export:{self.config.path.resolve()}", self._snapshot),
            "export_path": str(self.config.path.resolve()),
            "export_trace_count": self._export_trace_count,
            "max_traces": self.config.max_traces,
            "truncated": self._export_trace_count > self._snapshot.trace_count,
            "base_url": self.config.base_url,
        }
