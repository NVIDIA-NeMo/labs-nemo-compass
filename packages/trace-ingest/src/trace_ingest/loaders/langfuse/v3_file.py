# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join native Langfuse v3 CLI trace, observation, and score exports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from trace_ingest.loaders.langfuse.config import (
    LangfuseTraceLoadError,
    LangfuseV3FileTraceConfig,
    LangfuseV3FileTraceDescription,
    _describe_corpus,
)
from trace_ingest.loaders.langfuse.file_io import _read_cli_data
from trace_ingest.loaders.langfuse.normalization import (
    _normalize_trace,
    _trace_id,
    _trace_timestamp,
)
from trace_ingest.models import TraceSnapshot
from trace_ingest.source_links import file_source_url


@dataclass
class LangfuseV3FileTraceLoader:
    """Read three complete CLI responses without contacting Langfuse."""

    config: LangfuseV3FileTraceConfig
    _snapshot: TraceSnapshot = field(
        default_factory=lambda: TraceSnapshot([]), init=False, repr=False
    )
    _export_trace_count: int = field(default=0, init=False, repr=False)

    def load(self) -> TraceSnapshot:
        from langfuse.api.commons.types.observations_view import ObservationsView
        from langfuse.api.commons.types.trace_with_details import TraceWithDetails
        from langfuse.api.commons.types.trace_with_full_details import TraceWithFullDetails
        from langfuse.api.core.pydantic_utilities import parse_obj_as
        from langfuse.api.scores.types.get_scores_response_data import GetScoresResponseData

        try:
            traces = parse_obj_as(list[TraceWithDetails], _read_cli_data(self.config.traces_path))
            observations = parse_obj_as(
                list[ObservationsView], _read_cli_data(self.config.observations_path)
            )
            scores = parse_obj_as(
                list[GetScoresResponseData], _read_cli_data(self.config.scores_path)
            )
            trace_ids = [_trace_id(trace) for trace in traces]
            if len(trace_ids) != len(set(trace_ids)):
                raise LangfuseTraceLoadError("duplicate trace id")
            ordered = sorted(traces, key=lambda trace: (_trace_timestamp(trace), _trace_id(trace)))
            exported_ids = set(trace_ids)
            by_trace = defaultdict(dict)
            for observation in observations:
                if observation.trace_id not in exported_ids:
                    continue
                group = by_trace[observation.trace_id]
                if observation.id in group:
                    raise LangfuseTraceLoadError(f"duplicate observation id {observation.id!r}")
                group[observation.id] = observation
            scores_by_trace = defaultdict(dict)
            seen_scores = set()
            for score in scores:
                if score.trace_id not in exported_ids:
                    continue
                if score.id in seen_scores:
                    raise LangfuseTraceLoadError(f"duplicate score id {score.id!r}")
                seen_scores.add(score.id)
                if score.data_type == "CORRECTION":
                    raise LangfuseTraceLoadError(
                        "CORRECTION scores on exported traces are not supported by v3 trace normalization"
                    )
                scores_by_trace[score.trace_id][score.id] = score
            normalized = []
            for trace in ordered:
                observation_ids, score_ids = trace.observations, trace.scores
                if observation_ids is None or score_ids is None:
                    raise LangfuseTraceLoadError(
                        "Trace exports require observation and score references; omit --fields when exporting traces"
                    )
                trace_observations = by_trace[trace.id]
                trace_scores = scores_by_trace[trace.id]
                missing_observations = set(observation_ids) - trace_observations.keys()
                missing_scores = set(score_ids) - trace_scores.keys()
                if missing_observations or missing_scores:
                    raise LangfuseTraceLoadError(
                        f"Trace {trace.id!r} is missing referenced observations {sorted(missing_observations)} "
                        f"or scores {sorted(missing_scores)}; export a wider observation window or include all scores"
                    )
                project = getattr(trace, "projectId", None)
                for record in [*trace_observations.values(), *trace_scores.values()]:
                    record_project = getattr(record, "project_id", None) or getattr(
                        record, "projectId", None
                    )
                    if project and record_project and project != record_project:
                        raise LangfuseTraceLoadError(f"Trace {trace.id!r} has mismatched projectId")
                for score in trace_scores.values():
                    if score.observation_id and score.observation_id not in trace_observations:
                        raise LangfuseTraceLoadError(
                            f"Score {score.id!r} references a missing observation"
                        )
                payload = trace.dict(exclude_unset=True)
                payload["observations"] = [
                    item.dict(exclude_unset=True) for item in trace_observations.values()
                ]
                payload["scores"] = [
                    item.dict(exclude_unset=True) for item in trace_scores.values()
                ]
                full = parse_obj_as(TraceWithFullDetails, payload)
                result = _normalize_trace(
                    full,
                    require_complete_tree=True,
                    source_pointer={
                        "provider": "langfuse",
                        "export_path": str(self.config.traces_path.resolve()),
                        "trace_id": trace.id,
                    },
                )
                result.source_url = file_source_url(self.config.traces_path)
                normalized.append(result)
        except ValueError as error:
            raise LangfuseTraceLoadError(f"Invalid Langfuse v3 export records: {error}") from error
        self._export_trace_count = len(traces)
        self._snapshot = TraceSnapshot(normalized[-self.config.max_traces :])
        return self._snapshot

    def describe(self) -> LangfuseV3FileTraceDescription:
        return {
            **_describe_corpus(
                f"langfuse-export:{self.config.traces_path.resolve()}", self._snapshot
            ),
            "traces_path": str(self.config.traces_path.resolve()),
            "observations_path": str(self.config.observations_path.resolve()),
            "scores_path": str(self.config.scores_path.resolve()),
            "export_trace_count": self._export_trace_count,
            "max_traces": self.config.max_traces,
            "truncated": self._export_trace_count > self._snapshot.trace_count,
        }
