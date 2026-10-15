# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide tracing for one Compass CLI invocation; enabled by dedicated credentials."""

from __future__ import annotations

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
from typing import Literal, NotRequired, TypedDict

import nooa
from nooa.tracing import enable_tracing, shutdown_traces
from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation._spans import OpenInferenceSpan
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pydantic import JsonValue

from insight_agent.config import RunConfig
from insight_agent.config import TraceConfig as SourceConfig
from insight_agent.telemetry_export import ExceptionEventExporter
from insight_agent.version import __version__
from trace_ingest.loaders.trace_loaders import TraceDescription, TraceLoader
from trace_ingest.models import TraceSnapshot

tracer = OITracer(trace.get_tracer("compass", __version__), config=TraceConfig())
span = partial(tracer.start_as_current_span, openinference_span_kind="chain")


class FileReference(TypedDict):
    path: str
    sha256: str | None
    error: NotRequired[str]


class TraceReference(TypedDict):
    id: str
    source_url: str | None


class InputMetadata(TypedDict):
    source_type: str
    selection: dict[str, JsonValue]
    description: TraceDescription
    trace_count: int
    traces: list[TraceReference]
    files: list[FileReference]
    normalized_sha256: str
    snapshot_saved: Literal[False]


def _file_metadata(path: Path) -> FileReference:
    """Fingerprint a source file without exporting its contents."""
    path = path.resolve()
    try:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"path": str(path), "sha256": digest}
    except OSError:
        # A source may disappear after ingestion; telemetry must not prevent analysis.
        return {"path": str(path), "sha256": None, "error": "file unavailable after ingestion"}


def _input_metadata(
    config: SourceConfig, description: TraceDescription, snapshot: TraceSnapshot
) -> InputMetadata:
    """Identify the loaded corpus, keeping trace bodies out of the run's metadata."""
    source_type, source = next(
        (name, value) for name, value in config if name != "max_traces" and value is not None
    )
    files: list[FileReference] = []
    for value in source.model_dump().values():
        if not isinstance(value, Path):
            continue
        if value.is_dir():
            # These are the two loaders accepting directories of export files.
            suffixes = (
                {".jsonl"} if source_type == "langsmith_trace_export_file" else {".json", ".jsonl"}
            )
            files.extend(
                _file_metadata(path)
                for path in sorted(value.iterdir())
                if path.is_file()
                and (
                    path.suffix.lower()
                    if source_type == "langsmith_trace_export_file"
                    else path.suffix
                )
                in suffixes
            )
        else:
            files.append(_file_metadata(value))

    digest = hashlib.sha256()
    references: list[TraceReference] = []
    for item in snapshot:
        references.append({"id": item.id, "source_url": item.source_url})
        digest.update(item.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return {
        "source_type": source_type,
        "selection": config.model_dump(mode="json", exclude_none=True),
        "description": description,
        "trace_count": len(snapshot),
        "traces": references,
        "files": files,
        "normalized_sha256": digest.hexdigest(),
        "snapshot_saved": False,
    }


def _input_metadata_fields(metadata: InputMetadata) -> dict[str, str | int | bool]:
    """Keep provenance within LangSmith's scalar metadata value limits.

    The deployed OTLP mapper drops strings of 500 or more characters. Numbered
    parts preserve long JSON values without truncating references or file details.
    """
    fields: dict[str, str | int | bool] = {}
    for key, value in metadata.items():
        name = f"compass.input.{key}"
        encoded = value if isinstance(value, (str, int, bool)) else json.dumps(value)
        if isinstance(encoded, str) and len(encoded) >= 500:
            parts = [encoded[index : index + 499] for index in range(0, len(encoded), 499)]
            fields[f"{name}.parts"] = len(parts)
            fields.update({f"{name}.{index}": part for index, part in enumerate(parts)})
        else:
            fields[name] = encoded
    return fields


def record_input_metadata(
    config: SourceConfig, loader: TraceLoader, snapshot: TraceSnapshot
) -> None:
    """Attach provenance to the current Compass root after its ingestion span closes."""
    root = trace.get_current_span()
    if root.is_recording():
        root.set_attribute(
            "metadata",
            json.dumps(
                {
                    **_execution_metadata(),
                    **_input_metadata_fields(_input_metadata(config, loader.describe(), snapshot)),
                }
            ),
        )


def _execution_metadata() -> dict[str, str | bool]:
    metadata: dict[str, str | bool] = {
        "compass.version": __version__,
        "nooa.version": nooa.__version__,
        "python.version": platform.python_version(),
    }
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
            metadata.update({"compass.git_sha": sha, "compass.git_dirty": dirty})
        except (OSError, subprocess.SubprocessError):
            pass  # Git is optional; installed wheels need only their package version.
    return metadata


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
                    ExceptionEventExporter(
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
            root.set_input(config.model_dump(mode="json", exclude={"api_base"}))
            if api_key:
                # Use OpenInference span metadata for LangSmith compatibility.
                root.set_attribute("metadata", json.dumps(_execution_metadata()))
                print(
                    f"Compass telemetry: execution content and input references to {project}; OTLP trace ID {root.get_span_context().trace_id:032x}",
                    file=sys.stderr,
                )
            yield root
    finally:
        if api_key:
            shutdown_traces()
