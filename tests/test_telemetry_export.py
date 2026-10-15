# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pytest
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link, Status, StatusCode

from insight_agent.telemetry_export import ExceptionEventExporter


def finished_span(status=StatusCode.ERROR, existing_exception=False, fallback=False):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource({"service.name": "compass-test"}),
        span_limits=SpanLimits(max_events=1, max_links=1, max_attributes=6),
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-source", "1.2.3")
    with tracer.start_as_current_span("parent") as parent:
        with tracer.start_as_current_span(
            "code_execution", links=[Link(parent.get_span_context())] * 2
        ) as span:
            for index in range(8):
                span.set_attribute(f"other-{index}", index)
            prefix = "error" if fallback else "exception"
            span.set_attribute(f"{prefix}.type", "ValueError")
            span.set_attribute(f"{prefix}.message", "Recovered error")
            span.set_attribute("exception.stacktrace", "original stack")
            span.add_event("dropped-event")
            span.add_event("exception" if existing_exception else "diagnostic", {"a": 1})
            span.set_status(
                Status(status, "Recovered error" if status == StatusCode.ERROR else None)
            )
    result = exporter.get_finished_spans()[0]
    provider.shutdown()
    return result


@pytest.mark.parametrize("fallback", [False, True])
def test_error_event_preserves_encoded_span_and_original(fallback):
    original = finished_span(fallback=fallback)
    before = encode_spans([original])
    delegate = Mock(spec=SpanExporter)
    delegate.export.return_value = SpanExportResult.SUCCESS
    exporter = ExceptionEventExporter(delegate)
    assert exporter.export([original]) == SpanExportResult.SUCCESS
    forwarded = delegate.export.call_args.args[0][0]
    assert forwarded is not original
    assert json.loads(forwarded.to_json())["events"][-1]["name"] == "exception"
    after = encode_spans([forwarded])
    encoded = after.resource_spans[0].scope_spans[0].spans[0]
    event = encoded.events.pop()
    assert event.name == "exception"
    assert event.time_unix_nano == original.end_time
    assert {item.key: item.value.string_value for item in event.attributes} == {
        "exception.type": "ValueError",
        "exception.message": "Recovered error",
        "exception.stacktrace": "original stack",
    }
    # Every other encoded field, including identity, hierarchy, limits and scope, survives.
    assert after == before
    assert encode_spans([original]) == before
    assert len(original.events) == 1


@pytest.mark.parametrize(
    "status,existing_exception",
    [(StatusCode.OK, False), (StatusCode.UNSET, False), (StatusCode.ERROR, True)],
)
def test_preserves_success_and_existing_exception(status, existing_exception):
    original = finished_span(status, existing_exception)
    delegate = Mock(spec=SpanExporter)
    ExceptionEventExporter(delegate).export([original])
    assert delegate.export.call_args.args[0][0] is original


def test_forwards_failure_and_exporter_lifecycle():
    delegate = Mock(spec=SpanExporter)
    delegate.export.return_value = SpanExportResult.FAILURE
    delegate.force_flush.return_value = False
    exporter = ExceptionEventExporter(delegate)
    assert exporter.export([]) == SpanExportResult.FAILURE
    assert exporter.force_flush(1234) is False
    delegate.force_flush.assert_called_once_with(1234)
    exporter.shutdown()
    delegate.shutdown.assert_called_once_with()
