# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt error attributes to the exception events understood by LangSmith OTLP."""

import json
from collections.abc import Sequence
from operator import attrgetter

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.util import ns_to_iso_str
from opentelemetry.trace import SpanContext, StatusCode


class _ExceptionEventSpan(ReadableSpan):
    """Read-only export view; the ended SDK span remains untouched.

    SDK spans cannot accept events after ending. Delegate the public snapshot
    fields instead of constructing a new SDK snapshot or mutating its internals.
    """

    def __init__(self, original: ReadableSpan, event: Event) -> None:
        self._original = original
        self._export_events = (*original.events, event)

    name = property(attrgetter("_original.name"))
    context = property(attrgetter("_original.context"))
    parent = property(attrgetter("_original.parent"))
    resource = property(attrgetter("_original.resource"))
    attributes = property(attrgetter("_original.attributes"))
    events = property(attrgetter("_export_events"))
    links = property(attrgetter("_original.links"))
    kind = property(attrgetter("_original.kind"))
    instrumentation_info = property(attrgetter("_original.instrumentation_info"))
    instrumentation_scope = property(attrgetter("_original.instrumentation_scope"))
    status = property(attrgetter("_original.status"))
    start_time = property(attrgetter("_original.start_time"))
    end_time = property(attrgetter("_original.end_time"))
    dropped_attributes = property(attrgetter("_original.dropped_attributes"))
    dropped_events = property(attrgetter("_original.dropped_events"))
    dropped_links = property(attrgetter("_original.dropped_links"))

    def to_json(self, indent: int | None = 4) -> str:
        document = json.loads(self._original.to_json(indent=indent))
        event = self._export_events[-1]
        document["events"].append(
            {
                "name": event.name,
                "timestamp": ns_to_iso_str(event.timestamp),
                "attributes": dict(event.attributes or {}),
            }
        )
        return json.dumps(document, indent=indent)

    def get_span_context(self) -> SpanContext | None:
        return self._original.get_span_context()


def _with_exception_event(span: ReadableSpan) -> ReadableSpan:
    if span.status.status_code != StatusCode.ERROR or any(
        event.name == "exception" for event in span.events
    ):
        return span
    attributes = span.attributes or {}
    exception = {
        key: attributes[key]
        for key in ("exception.type", "exception.message", "exception.stacktrace")
        if key in attributes
    }
    if "exception.type" not in exception and "error.type" in attributes:
        exception["exception.type"] = attributes["error.type"]
    if "exception.message" not in exception:
        message = attributes.get("error.message")
        if message:
            exception["exception.message"] = message
    if not exception:
        return span
    return _ExceptionEventSpan(span, Event("exception", exception, timestamp=span.end_time))


class ExceptionEventExporter(SpanExporter):
    """Forward spans with explicit error events without changing execution identity."""

    def __init__(self, delegate: SpanExporter) -> None:
        self._delegate = delegate

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._delegate.export(tuple(_with_exception_event(span) for span in spans))

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)
