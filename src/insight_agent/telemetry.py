# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide tracing for one Compass CLI invocation; enabled by dedicated credentials."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from functools import partial
from pathlib import Path
from typing import Literal

import nooa
from nooa.tracing import enable_tracing, shutdown_traces
from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation._spans import OpenInferenceSpan
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pydantic import BaseModel, ConfigDict, Field

from insight_agent.config import RunConfig
from insight_agent.config import TraceConfig as SourceConfig
from insight_agent.telemetry_export import CompassSpanExporter
from insight_agent.version import __version__
from trace_ingest.loaders.trace_loaders import TraceDescription, TraceLoader
from trace_ingest.models import TraceSnapshot

tracer = OITracer(trace.get_tracer("compass", __version__), config=TraceConfig())


@contextmanager
def span(name: str) -> Iterator[OpenInferenceSpan]:
    """Record one Compass operation, including cancellation as interrupted work."""
    with tracer.start_as_current_span(name, openinference_span_kind="chain") as record:
        try:
            yield record
        except asyncio.CancelledError as error:
            # OTel intentionally ignores BaseException subclasses. Cancellation is
            # not a completed Compass operation, so retain the actual exception.
            record.record_exception(error)
            record.set_status(trace.StatusCode.ERROR)
            raise


class InputMetadata(BaseModel):
    """Corpus identity and resolved source coordinates, never source trace contents.

    Query filters and time bounds are already in the root span's typed input config.
    The fingerprint identifies the analyzed content but cannot retrieve or archive it.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    source_type: Literal[
        "filesystem",
        "gym",
        "atif",
        "mlflow_experiment",
        "mlflow_export",
        "langfuse",
        "langfuse_trace_export_file",
        "langsmith",
        "langsmith_trace_export_file",
        "braintrust",
        "intake",
    ]
    trace_count: int = Field(ge=0)
    normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_saved: Literal[False] = False
    max_traces: int | None = Field(default=None, ge=1)
    truncated: bool | None = None
    export_trace_count: int | None = Field(default=None, ge=0)
    ignored_score_count: int | None = Field(default=None, ge=0)
    endpoint: str | None = None
    project_id: str | None = None
    experiment_id: str | None = None
    workspace: str | None = None
    source_path: str | None = None
    scores_path: str | None = None
    api_version: str | None = None


class RunInput(RunConfig):
    """Compass run configuration and complete resolved input provenance."""

    input_source: InputMetadata | None = None


class InputMetadataSummary(InputMetadata):
    """Searchable scalar projection; complete coordinates remain in the root input."""

    details_location: Literal["inputs.input_source"] = "inputs.input_source"


class ExecutionMetadata(BaseModel):
    """Software identity of the Compass process that generated this trace."""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    compass_version: str = Field(alias="compass.version")
    nooa_version: str = Field(alias="nooa.version")
    python_version: str = Field(alias="python.version")
    git_sha: str | None = Field(default=None, alias="compass.git_sha")
    git_dirty: bool | None = Field(default=None, alias="compass.git_dirty")


def _input_metadata(
    config: SourceConfig, description: TraceDescription, snapshot: TraceSnapshot
) -> InputMetadata:
    """Identify the loaded corpus without per-trace lists or duplicate query settings."""
    source_type, source = next(
        (name, value) for name, value in config if name != "max_traces" and value is not None
    )
    configured = source.model_dump()
    resolved = dict(description)
    digest = hashlib.sha256()
    for item in snapshot:
        digest.update(item.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    coordinates = {
        key: resolved.get(key)
        for key in (
            "project_id",
            "experiment_id",
            "workspace",
            "api_version",
            "max_traces",
            "truncated",
            "export_trace_count",
            "ignored_score_count",
        )
    }
    if source_type == "intake":
        query = resolved.get("query")
        if isinstance(query, dict):
            coordinates["max_traces"] = query.get("max_traces")
    coordinates["api_version"] = coordinates["api_version"] or configured.get("api_version")
    coordinates["endpoint"] = (
        resolved.get("base_url")
        or resolved.get("api_url")
        or resolved.get("tracking_uri")
        or configured.get("base_url")
    )
    for input_name, output_name in (("path", "source_path"), ("scores_path", "scores_path")):
        path = configured.get(input_name)
        if isinstance(path, Path):
            coordinates[output_name] = str(path.resolve())
    return InputMetadata.model_validate(
        {
            "source_type": source_type,
            "trace_count": len(snapshot),
            "normalized_sha256": digest.hexdigest(),
            **coordinates,
        }
    )


def _input_metadata_fields(metadata: InputMetadata) -> dict[str, str | int | bool]:
    """Export named scalar fields without splitting values into arbitrary fragments."""
    return {
        f"compass.input.{key}": value
        for key, value in InputMetadataSummary.model_validate(metadata.model_dump())
        .model_dump(exclude_none=True)
        .items()
        if not isinstance(value, str) or len(value) < 500
    }


def record_input_metadata(config: RunConfig, loader: TraceLoader, snapshot: TraceSnapshot) -> None:
    """Attach provenance to the current Compass root after its ingestion span closes."""
    root = trace.get_current_span()
    if root.is_recording():
        input_source = _input_metadata(config.trace, loader.describe(), snapshot)
        root_input = RunInput.model_validate({**config.model_dump(), "input_source": input_source})
        root.set_attribute("input.value", root_input.model_dump_json(exclude={"api_base"}))
        root.set_attribute("input.mime_type", "application/json")
        root.set_attribute(
            "metadata",
            json.dumps(
                {
                    **_execution_metadata(),
                    **_input_metadata_fields(input_source),
                }
            ),
        )


def _execution_metadata() -> dict[str, str | bool]:
    metadata = ExecutionMetadata(
        compass_version=__version__,
        nooa_version=nooa.__version__,
        python_version=platform.python_version(),
    )
    # Inspect Compass's checkout, not the caller's or analyzed agent's repository.
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / ".git").exists():
        git = partial(
            subprocess.check_output,
            cwd=checkout,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        try:
            sha = git(["git", "rev-parse", "HEAD"]).strip()
            dirty = bool(git(["git", "status", "--porcelain", "--untracked-files=normal"]))
            metadata.git_sha = sha
            metadata.git_dirty = dirty
        except (OSError, subprocess.SubprocessError):
            pass  # Git is optional; installed wheels need only their package version.
    return metadata.model_dump(by_alias=True, exclude_none=True)


@contextmanager
def telemetry_session(config: RunConfig) -> Iterator[OpenInferenceSpan]:
    api_key = os.getenv("COMPASS_TELEMETRY_LANGSMITH_API_KEY", "").strip()
    project = os.getenv("COMPASS_TELEMETRY_LANGSMITH_PROJECT", "").strip()
    if api_key and not project:
        print("Compass telemetry skipped: set COMPASS_TELEMETRY_LANGSMITH_PROJECT", file=sys.stderr)
        api_key = ""
    endpoint = (
        os.getenv("COMPASS_TELEMETRY_LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
    ).rstrip("/")
    source = config.trace.langsmith
    if api_key and source and source.project == project:
        from langsmith.utils import get_api_url

        if get_api_url(source.api_url) == endpoint:
            print(
                "Compass telemetry skipped: analyzing the telemetry project itself", file=sys.stderr
            )
            api_key = ""
    if api_key:
        headers = {"x-api-key": api_key, "Langsmith-Project": project}
        if workspace := os.getenv("COMPASS_TELEMETRY_LANGSMITH_WORKSPACE_ID"):
            headers["X-Tenant-Id"] = workspace
        # Nooa prints initialization messages; stdout belongs to the CLI's YAML output.
        with redirect_stdout(sys.stderr):
            enable_tracing(
                exporters=[
                    CompassSpanExporter(
                        OTLPSpanExporter(
                            endpoint=f"{endpoint}/otel/v1/traces",
                            headers=headers,
                            compression=Compression.Gzip,
                            timeout=float(
                                os.getenv("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT")
                                or os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT")
                                or "60"
                            ),
                        )
                    )
                ],
            )
    try:
        with span("compass") as root:
            root.set_input(
                RunInput.model_validate(config.model_dump()).model_dump(
                    mode="json", exclude={"api_base"}
                )
            )
            if api_key:
                # Use OpenInference span metadata for LangSmith compatibility.
                root.set_attribute("metadata", json.dumps(_execution_metadata()))
                print(
                    f"Compass telemetry: execution content and source provenance to {project}; OTLP trace ID {root.get_span_context().trace_id:032x}",
                    file=sys.stderr,
                )
            yield root
    finally:
        if api_key:
            shutdown_traces()
