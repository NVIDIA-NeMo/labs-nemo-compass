# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide tracing for one Compass CLI invocation; enabled by LangSmith credentials."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from functools import partial

from nooa.tracing import enable_tracing, exporters, shutdown_traces
from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation._spans import OpenInferenceSpan
from opentelemetry import trace

from insight_agent.config import RunConfig
from insight_agent.version import __version__

PROJECT = "compass-dogfood"

tracer = OITracer(trace.get_tracer("compass", __version__), config=TraceConfig())
span = partial(tracer.start_as_current_span, openinference_span_kind="chain")


@contextmanager
def telemetry_session(config: RunConfig) -> Iterator[OpenInferenceSpan]:
    api_key = (
        os.getenv("LANGSMITH_API_KEY", "").strip() or os.getenv("LANGCHAIN_API_KEY", "").strip()
    )
    endpoint = (os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")
    source = config.trace.langsmith
    if (
        api_key
        and source
        and source.project == PROJECT
        and (source.api_url or endpoint).rstrip("/") == endpoint
    ):
        print("Compass telemetry skipped: analyzing the telemetry project itself", file=sys.stderr)
        api_key = ""
    if api_key:
        headers = {"x-api-key": api_key, "Langsmith-Project": PROJECT}
        if workspace := os.getenv("LANGSMITH_WORKSPACE_ID"):
            headers["X-Tenant-Id"] = workspace
        # Nooa prints initialization messages; stdout belongs to the CLI's YAML output.
        with redirect_stdout(sys.stderr):
            enable_tracing(
                exporters=[exporters.otlp(f"{endpoint}/otel/v1/traces", headers)],
                extra_resource_attrs={"compass.version": __version__},
            )
    try:
        with span("compass") as root:
            root.set_input(config.model_dump(mode="json", exclude={"api_base"}))
            if api_key:
                print(
                    f"Compass telemetry: full content to {PROJECT}; OTLP trace ID {root.get_span_context().trace_id:032x}",
                    file=sys.stderr,
                )
            yield root
    finally:
        if api_key:
            shutdown_traces()
