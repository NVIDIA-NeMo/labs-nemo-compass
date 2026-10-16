# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from nooa.runtime.harness_metrics import HarnessMetrics
from nooa.tracing._hooks_impl import OpenInferenceHooks
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import TypeAdapter

from insight_agent.telemetry_framework import (
    FRAMEWORK_FIELD_MODELS,
    SPAN_CONTRACTS,
    MethodInput,
    PythonInput,
    PythonOutput,
    TimingFields,
    attribute_definition,
)


def test_framework_contracts_have_field_meanings():
    for model in (*FRAMEWORK_FIELD_MODELS, MethodInput, PythonInput, PythonOutput, TimingFields):
        assert all(field.description for field in model.model_fields.values())
        assert model.model_json_schema()
    for contract in SPAN_CONTRACTS.values():
        assert contract.description
        for payload in (contract.input, contract.output):
            if payload is not None:
                assert payload.json_schema()


def test_real_nooa_code_execution_fields_match_contract():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    hooks = OpenInferenceHooks(provider.get_tracer("contract-test"))
    agent = SimpleNamespace()
    source = "print('hello')\nreturn 42"
    context = hooks.before_code_execution(agent, source, "execution-1")
    hooks.after_code_execution(
        agent,
        source,
        SimpleNamespace(stdout="hello\n", stderr="", returned_value=42),
        None,
        context,
        "execution-1",
    )
    (recorded,) = exporter.get_finished_spans()
    attributes = recorded.attributes
    assert attributes is not None
    assert all(attribute_definition(key) is not None for key in attributes)
    assert isinstance(attributes["input.value"], str)
    assert isinstance(attributes["output.value"], str)
    assert PythonInput.model_validate_json(attributes["input.value"]).code == source
    assert PythonOutput.model_validate_json(attributes["output.value"]).returned_value == "42"
    provider.shutdown()


def test_harness_contract_resolves_real_exported_values():
    metrics = HarnessMetrics(
        exec_python_total=2, exec_python_success=1, implicit_return_transformed=1
    )
    metrics.time_llm_call.record(0.1)
    metrics.time_llm_call.record(0.2)
    fields = metrics.to_span_attributes()
    assert fields["harness.time.llm_call.count"] == 2
    for key in fields:
        definition = attribute_definition(key)
        assert definition is not None
        assert definition.description
        TypeAdapter(definition.python_type).validate_python(fields[key], strict=True)
    assert (
        attribute_definition("llm.input_messages.3.message.tool_calls.0.tool_call.function.name")
        is not None
    )
    assert attribute_definition("unrecognized.field") is None


def test_real_litellm_instrumentation_payload_projections():
    from litellm import ModelResponse
    from nooa.tracing._litellm_patch import _patched_set_output_message_value
    from openinference.instrumentation.litellm import (
        _instrument_func_type_completion,
        _instrument_func_type_responses,
    )
    from openinference.instrumentation.litellm._responses_attributes import (
        _get_attributes_from_response_output,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("model-projection-test")
    with tracer.start_as_current_span("llm.chat_completion") as span:
        _instrument_func_type_completion(
            span,
            {
                "model": "test",
                "messages": [
                    {"role": "user", "content": "hello", "cache_control": {"type": "ephemeral"}}
                ],
            },
        )
        _patched_set_output_message_value(
            span, ModelResponse(choices=[{"message": {"role": "assistant", "content": "answer"}}])
        )
    with tracer.start_as_current_span("llm.responses") as span:
        _instrument_func_type_responses(
            span, {"model": "test", "input": [{"role": "user", "content": "hello"}]}
        )
        span.set_attributes(
            _get_attributes_from_response_output(
                {
                    "model": "test",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "answer"}],
                        }
                    ],
                }
            )
        )
    for span in exporter.get_finished_spans():
        contract = SPAN_CONTRACTS[span.name]
        assert span.attributes is not None
        for side, adapter in (("input", contract.input), ("output", contract.output)):
            assert adapter is not None
            value = span.attributes[f"{side}.value"]
            assert isinstance(value, str)
            if span.attributes.get(f"{side}.mime_type") == "application/json":
                adapter.validate_json(value)
            else:
                adapter.validate_python(value)
    provider.shutdown()
