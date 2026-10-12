# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Compass spans; Nooa/OpenInference and OpenTelemetry own instrumentation/export."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from pydantic_core import to_json

from insight_agent.config import RunConfig
from insight_agent.version import __version__

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

_TRACER: ContextVar[Tracer | None] = ContextVar("compass_tracer", default=None)


def capture(record: Span | None, field: str, value: object) -> None:
    if record is not None and record.is_recording():
        record.set_attribute(
            f"{field}.value",
            to_json(value, fallback=lambda obj: {"omitted_type": type(obj).__name__}).decode(),
        )
        record.set_attribute(f"{field}.mime_type", "application/json")


@contextmanager
def span(name: str, inputs: object = None) -> Iterator[Span | None]:
    tracer = _TRACER.get()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as record:
        record.set_attribute("openinference.span.kind", "CHAIN")
        capture(record, "input", inputs)
        yield record


@contextmanager
def telemetry_session(config: RunConfig, report: Callable[[str], None]) -> Iterator[Span | None]:
    settings = config.telemetry
    if settings is None:
        yield None
        return
    try:
        from nooa.runtime.hooks import get_hooks, set_hooks
        from nooa.tracing import NemoOOAgentsInstrumentor
        from nooa.tracing._secret_scrubber import SecretScrubSpanProcessor
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import SpanLimits, TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise ValueError("Telemetry requires: uv sync --extra langsmith") from exc

    endpoint = (
        settings.api_url or os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
    ).rstrip("/")
    source = config.trace.langsmith
    if (
        source
        and source.project == settings.project
        and (
            source.api_url or os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
        ).rstrip("/")
        == endpoint
    ):
        raise ValueError("Use a separate telemetry project from the LangSmith input project")
    headers = {
        "x-api-key": os.getenv("LANGSMITH_API_KEY") or os.environ["LANGCHAIN_API_KEY"],
        "Langsmith-Project": settings.project,
    }
    if workspace := os.getenv("LANGSMITH_WORKSPACE_ID"):
        headers["X-Tenant-Id"] = workspace
    llm = LiteLLMInstrumentor()
    if llm.is_instrumented_by_opentelemetry:
        raise ValueError(
            "LiteLLM is already instrumented; run Compass telemetry in a separate process"
        )
    provider = TracerProvider(span_limits=SpanLimits(max_span_attributes=SpanLimits.UNSET))
    provider.add_span_processor(
        SecretScrubSpanProcessor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=f"{endpoint}/otel/v1/traces", headers=headers, timeout=5)
            )
        )
    )
    previous_hooks = get_hooks()
    agent = NemoOOAgentsInstrumentor()
    token = _TRACER.set(provider.get_tracer("compass", __version__))
    try:
        agent.instrument(tracer_provider=provider)
        llm.instrument(tracer_provider=provider)
        with span("compass", config.model_dump(mode="json", exclude={"api_base"})) as root:
            assert root is not None
            root.set_attribute("langsmith.metadata.compass_version", __version__)
            report(
                f"Compass telemetry: full content to {settings.project}; OTLP trace ID {root.get_span_context().trace_id:032x}"
            )
            yield root
    finally:
        llm.uninstrument()
        agent.uninstrument()
        set_hooks(previous_hooks)
        _TRACER.reset(token)
        provider.force_flush(timeout_millis=10000)
        provider.shutdown()
