# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load CLI exports of Langfuse v4 observations and optional Scores API v3 pages."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

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


@dataclass(frozen=True)
class LangfuseV4FileTraceConfig:
    """CLI observation exports, with optional score exports and platform links."""

    path: Path
    max_traces: int = LANGFUSE_DEFAULT_MAX_TRACES
    scores_path: Path | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")
        if self.scores_path is not None:
            object.__setattr__(self, "scores_path", Path(self.scores_path))


class LangfuseV4FileTraceDescription(_LangfuseCorpusDescription):
    """Offline v4 corpus and score coverage."""

    export_path: str
    export_trace_count: int
    max_traces: int
    truncated: bool
    base_url: str | None
    scores_path: str | None
    ignored_score_count: int


@dataclass
class LangfuseV4FileTraceLoader:
    """Reconstruct traces from CLI response pages without contacting Langfuse."""

    config: LangfuseV4FileTraceConfig
    _snapshot: TraceSnapshot = field(
        default_factory=lambda: TraceSnapshot([]), init=False, repr=False
    )
    _export_trace_count: int = field(default=0, init=False)
    _ignored_score_count: int = field(default=0, init=False)

    def load(self) -> TraceSnapshot:
        from langfuse.api.core.pydantic_utilities import parse_obj_as
        from langfuse.api.observations.types.observations_v2response import ObservationsV2Response
        from langfuse.api.scores_v3.types.score_v3 import ScoreV3

        observations: dict[tuple[str, str], ObservationV2] = {}
        trace_ids_by_span: dict[str, set[str]] = defaultdict(set)
        groups: dict[str, list[ObservationV2]] = defaultdict(list)
        for body, location in _pages(self.config.path):
            try:
                response = parse_obj_as(ObservationsV2Response, body)
                for observation in response.data:
                    if not observation.id or not observation.trace_id or not observation.project_id:
                        raise ValueError("observation id, traceId, and projectId must be nonempty")
                    key = (observation.trace_id, observation.id)
                    if key in observations:
                        raise ValueError(f"duplicate observation id {observation.id!r}")
                    _started_at(observation)
                    observations[key] = observation
                    trace_ids_by_span[observation.id].add(observation.trace_id)
                    groups[observation.trace_id].append(observation)
            except ValueError as error:
                raise LangfuseTraceLoadError(
                    f"Invalid Langfuse observations at {location}: {error}"
                ) from error
        projects = {obs.project_id for obs in observations.values()}
        if len(projects) > 1:
            raise LangfuseTraceLoadError("Langfuse export contains multiple projects")

        scores: dict[str, list[ScoreV3]] = defaultdict(list)
        score_ids: set[str] = set()
        ignored = 0
        if self.config.scores_path is not None:
            for body, location in _pages(self.config.scores_path):
                try:
                    # CLI --all replaces API pagination metadata; parse the typed records only.
                    score_records = parse_obj_as(list[ScoreV3], body["data"])
                    for score in score_records:
                        if score.id in score_ids:
                            raise ValueError(f"duplicate score id {score.id!r}")
                        score_ids.add(score.id)
                        if projects and score.project_id not in projects:
                            raise ValueError("score projectId does not match observations")
                        subject = score.subject
                        trace_id = None
                        if subject is not None and subject.kind == "trace":
                            trace_id = subject.id
                        elif subject is not None and subject.kind == "observation":
                            candidates = trace_ids_by_span.get(subject.id, set())
                            trace_id = subject.trace_id
                            if trace_id is None and len(candidates) == 1:
                                trace_id = next(iter(candidates))
                        if trace_id in groups:
                            scores[trace_id].append(score)
                        else:
                            ignored += 1
                except ValueError as error:
                    raise LangfuseTraceLoadError(
                        f"Invalid Langfuse score at {location}: {error}"
                    ) from error

        # Validate every exported tree before applying the trace limit, as in the v3 file loader.
        traces = []
        for trace_id, items in groups.items():
            trace = _normalize_v4_trace(trace_id, items, scores[trace_id], self.config.base_url)
            if trace.source_url is None:
                trace.source_url = file_source_url(self.config.path)
            trace.attributes["export_path"] = str(self.config.path.resolve())
            traces.append((min(_started_at(obs) for obs in items), trace_id, trace))
        self._snapshot = TraceSnapshot(
            trace for _, _, trace in sorted(traces)[-self.config.max_traces :]
        )
        self._export_trace_count = len(groups)
        self._ignored_score_count = ignored
        return self._snapshot

    def describe(self) -> LangfuseV4FileTraceDescription:
        return {
            **_describe_corpus(f"langfuse-v4-export:{self.config.path.resolve()}", self._snapshot),
            "export_path": str(self.config.path.resolve()),
            "export_trace_count": self._export_trace_count,
            "max_traces": self.config.max_traces,
            "truncated": self._export_trace_count > self._snapshot.trace_count,
            "base_url": self.config.base_url,
            "scores_path": str(self.config.scores_path.resolve())
            if self.config.scores_path
            else None,
            "ignored_score_count": self._ignored_score_count,
        }


def _pages(path: Path) -> Iterator[tuple[dict[str, object], str]]:
    """Read CLI envelopes, one JSON document or one envelope per JSONL line."""
    files = (
        sorted(p for p in path.iterdir() if p.is_file() and p.suffix in {".json", ".jsonl"})
        if path.is_dir()
        else [path]
    )
    if not files:
        raise LangfuseTraceLoadError(f"Langfuse export {path} contains no JSON/JSONL files")
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
            pages = (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if file.suffix == ".jsonl"
                else [json.loads(text)]
            )
            if not pages:
                raise ValueError("empty file; export a successful CLI response even for no results")
            for page_index, page in enumerate(pages, 1):
                if not isinstance(page, dict) or page.get("status") != 200:
                    raise ValueError(
                        "expected a successful CLI --json response envelope (status 200)"
                    )
                body = page.get("body")
                if not isinstance(body, dict) or not isinstance(body.get("data"), list):
                    raise ValueError("expected an observations or scores response with body.data")
                meta = body.get("meta")
                if not isinstance(meta, dict):
                    raise ValueError("expected response metadata")
                if meta.get("truncated"):
                    raise ValueError(
                        "CLI export is truncated; shorten the window or increase --max-items"
                    )
                if meta.get("cursor"):
                    raise ValueError("CLI export has more pages; export with --all")
                yield body, f"{file}, response {page_index}"
        except (ValueError, UnicodeError) as error:
            raise LangfuseTraceLoadError(f"Invalid Langfuse CLI export {file}: {error}") from error
