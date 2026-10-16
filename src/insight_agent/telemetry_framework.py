# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Field inventory for the framework spans visible in Compass telemetry.

Compass payload contracts are enforced at their recording sites. Framework
contracts describe the installed Nooa/OpenInference serializers; they do not
rewrite or reject third-party telemetry. In particular, a Python repr is a
human-readable diagnostic, not a lossless structured return value.

Use ``SPAN_CONTRACTS`` for input/output models and ``attribute_definition`` for
flat OTLP fields. Standard fields are owned by OpenInference semantic conventions;
provider request/response types are owned by the OpenAI SDK. Nooa's full harness
metric inventory remains in ``nooa.runtime.harness_metrics.HarnessMetrics`` and
its ``_SPAN_SCHEMA`` mapping, rather than a second copy maintained by Compass.
"""

from dataclasses import dataclass
from typing import Annotated

from nooa.runtime.harness_metrics import _SPAN_SCHEMA, HarnessMetrics
from openai.types.chat import ChatCompletionMessageToolCallParam
from openai.types.responses import ResponseInputParam
from openinference.instrumentation import Message
from openinference.semconv import trace as semantics
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from insight_agent import telemetry_models as phase
from insight_agent.config import TraceConfig
from insight_agent.telemetry import ExecutionMetadata, InputMetadataSummary, RunInput


class FrameworkFields(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AgentFields(FrameworkFields):
    name: str = Field(
        alias="agent.name", description="Python agent class performing this method or generation."
    )
    method: str = Field(
        alias="agent.method", description="Python method requested from this agent."
    )
    call_id: str = Field(
        alias="agent.call_id",
        description="Nooa method invocation identifier, distinct from the OTLP span ID.",
    )
    parent_call_id: str = Field(
        alias="agent.parent_call_id",
        description="Calling Nooa method invocation ID, if this method was called by another agent method.",
    )
    signature: str = Field(
        alias="agent.method_signature", description="Inspected Python method signature."
    )
    docstring: str = Field(
        alias="agent.docstring",
        description="First line of the Python method's docstring, when available.",
    )
    file_path: str = Field(
        alias="agent.file_path",
        description="Source file containing this agent method, relative to the framework's working directory when possible.",
    )
    source_code: str = Field(
        alias="agent.source_code",
        description="Python source of an implemented method; not generated model output.",
    )
    strategy: str = Field(
        alias="agent.strategy.name", description="Nooa generation strategy selected for the method."
    )


class GenerationFields(FrameworkFields):
    strategy: str = Field(
        alias="generation.strategy",
        description="Nooa strategy executing this generation session, which may involve several model calls and tool executions.",
    )
    id: str = Field(
        alias="generation.id",
        description="Nooa generation session identifier; code execution spans use it to identify their generation.",
    )
    parent_id: str = Field(
        alias="generation.parent_id",
        description="Parent generation session identifier for nested generation.",
    )
    model_name: str = Field(
        alias="generation.llm.model_name",
        description="Model selected by Nooa for this generation session.",
    )
    selection_source: str = Field(
        alias="generation.llm.selection_source",
        description="Nooa selection rule that supplied this generation's model, such as method override or agent default.",
    )
    result_type: str = Field(
        alias="result.type",
        description="Python class name of the returned value; Nooa writes 'None' for any falsy result, not only None.",
    )


class ExecutionFields(FrameworkFields):
    code_length: int = Field(
        alias="code.length",
        description="Character count of the Python source submitted for execution.",
    )
    execution_id: str = Field(
        alias="execution.id",
        description="Nooa execution identifier for this Python/tool invocation.",
    )
    tool_call_id: str = Field(
        alias="tool_call_id",
        description="Nooa's duplicate model tool-call identifier; standard tool_call.id is the canonical OpenInference counterpart.",
    )
    method_name: str = Field(
        alias="method.name", description="Generated Python method invoked by the execution runtime."
    )
    invocation_id: str = Field(
        alias="invocation.id", description="Nooa generated-method invocation identifier."
    )


class ContextFields(FrameworkFields):
    system_message: str = Field(
        alias="nooa.system_message",
        description="Rendered system context, or its unified diff from the previous context in the same method call; also used as this span's text input.",
    )
    is_diff: bool = Field(
        alias="nooa.system_message.is_diff",
        description="False for the first complete context snapshot; true for a unified diff against the preceding snapshot.",
    )
    turn_index: int = Field(
        alias="nooa.system_message.turn_index",
        description="Zero-based context-render attempt in this method call; unchanged contexts emit no span, so indices may have gaps.",
    )


class DiagnosticFields(FrameworkFields):
    viewer_plugin: str = Field(
        alias="nooa.viewer.plugin",
        description="Nooa trace viewer renderer hint; does not identify an additional execution step.",
    )
    error_type: str = Field(
        alias="error.type",
        description="Nooa error class or cancellation category; exception.type carries the exception-event equivalent.",
    )
    error_message: str = Field(
        alias="error.message",
        description="Nooa failure or cancellation description; a later successful parent may have recovered from this failure.",
    )
    exception_type: str = Field(
        alias="exception.type",
        description="Exception class recorded by Nooa and forwarded to an OpenTelemetry exception event.",
    )
    exception_message: str = Field(
        alias="exception.message",
        description="Exception message after the framework's secret scrubbing.",
    )
    exception_stacktrace: str = Field(
        alias="exception.stacktrace",
        description="Formatted exception traceback after secret scrubbing.",
    )
    redacted_count: int = Field(
        alias="secrets.redacted_count",
        description="Count of secret replacements made by Nooa's scrub processor in this span.",
    )


class ModelExtensionFields(FrameworkFields):
    finish_reason: str = Field(
        alias="llm.finish_reason",
        description="Provider's reason the model response ended, e.g. stop or tool_calls.",
    )
    reasoning_content: str = Field(
        alias="llm.reasoning_content",
        description="Reasoning text returned by the provider and captured by Nooa's LiteLLM instrumentation, when present.",
    )
    original_cost: float = Field(
        alias="llm.cost.total.original",
        description="Original response cost in USD from the gateway's x-litellm-response-cost-original header.",
    )
    discount: float = Field(
        alias="llm.cost.discount_amount",
        description="Response discount in USD reported by the LiteLLM gateway.",
    )
    margin: float = Field(
        alias="llm.cost.margin_amount",
        description="Response margin amount in USD reported by the LiteLLM gateway.",
    )
    margin_percent: float = Field(
        alias="llm.cost.margin_percent",
        description="Response margin percentage reported by the LiteLLM gateway; not a token count.",
    )


class HarnessFields(FrameworkFields):
    error_count: int = Field(
        alias="harness.exec_error.count",
        description="Recorded Python execution errors in this generation session; zero is omitted.",
    )
    error_types: list[str] = Field(
        alias="harness.exec_error.types",
        description="Exception class names for recorded execution errors; bounded diagnostic history, not necessarily unique types.",
    )
    execution_success: int = Field(
        alias="harness.exec_python.success",
        description="Python executions completed successfully in this generation.",
    )
    execution_total: int = Field(
        alias="harness.exec_python.total",
        description="Total Python execution attempts in this generation.",
    )
    implicit_return: int = Field(
        alias="harness.implicit_return_transformed",
        description="Count of implicit Python returns transformed into explicit returns during this generation.",
    )
    prefill_type: str = Field(
        alias="harness.prefill_type",
        description="Nooa prefill strategy used to initialize this generation's execution context.",
    )
    turn_count: int = Field(
        alias="harness.turn_count",
        description="Number of generation loop turns, not number of input traces or successful model calls.",
    )


class TimingFields(FrameworkFields):
    total_s: float = Field(
        description="Sum of measured durations in seconds for this operation during one generation session; may overlap child-operation timings."
    )
    min_s: float = Field(
        description="Smallest measured duration in seconds; zero/default fields are omitted."
    )
    max_s: float = Field(description="Largest measured duration in seconds.")
    avg_s: float = Field(description="Total duration divided by measurement count, in seconds.")
    count: int = Field(
        description="Number of measurements contributing to this operation's timing statistics."
    )
    samples: list[float] = Field(
        description="First at most 20 measured durations in seconds, rounded to six decimal places; not the complete sample set when count exceeds 20."
    )


TIMING_OPERATIONS = {
    "session_init": "Initialize the Python execution session",
    "prefill": "Execute initial context setup",
    "prepare_context": "Prepare the agent context for a model call",
    "render_context": "Render context into model messages",
    "llm_call": "Wait for a model call",
    "code_validation": "Check generated Python before execution",
    "code_execution": "Execute generated Python",
    "tracing_overhead": "Record framework telemetry",
}


class StandardFields(FrameworkFields):
    """Observed OpenInference scalar projections; aliases use upstream semantic names."""

    session_id: str = Field(
        alias="session.id",
        description="Session identifier assigned by the Nooa session processor; groups executions and is not an input trace ID.",
    )
    input_value: str = Field(
        alias="input.value",
        description="Serialized input payload; input.mime_type determines JSON versus text, and the span contract defines its decoded type.",
    )
    input_mime_type: str = Field(
        alias="input.mime_type",
        description="Input serialization media type, typically application/json or text/plain.",
    )
    output_value: str = Field(
        alias="output.value",
        description="Serialized output payload; output.mime_type and the span contract determine its decoded type.",
    )
    output_mime_type: str = Field(
        alias="output.mime_type",
        description="Output serialization media type; framework text outputs may omit this attribute.",
    )
    openinference_span_kind: str = Field(
        alias="openinference.span.kind",
        description="OpenInference execution category, such as CHAIN, AGENT, TOOL, or LLM.",
    )
    metadata: str = Field(
        alias="metadata",
        description="JSON object of custom root metadata; schemas are ExecutionMetadata and InputMetadataSummary.",
    )
    llm_model_name: str = Field(
        alias="llm.model_name", description="Model identifier recorded for this model call."
    )
    llm_provider: str = Field(
        alias="llm.provider",
        description="Model provider inferred by the instrumentation from the LiteLLM model name.",
    )
    llm_invocation_parameters: str = Field(
        alias="llm.invocation_parameters",
        description="JSON request options after instrumentation removes sensitive/excluded keys; dynamic provider-specific options are arbitrary JSON, not a complete request.",
    )
    llm_token_count_prompt: int = Field(
        alias="llm.token_count.prompt",
        description="Input tokens reported for this model call, including cached tokens where the provider includes them.",
    )
    llm_token_count_completion: int = Field(
        alias="llm.token_count.completion",
        description="Output tokens reported by the provider for this call.",
    )
    llm_token_count_total: int = Field(
        alias="llm.token_count.total",
        description="Total provider-reported tokens for this call; do not sum adapter and nested provider spans for the same request.",
    )
    llm_token_count_completion_details_reasoning: int = Field(
        alias="llm.token_count.completion_details.reasoning",
        description="Reasoning output tokens reported by the provider; a component of completion tokens, not additional usage.",
    )
    llm_token_count_prompt_details_cache_read: int = Field(
        alias="llm.token_count.prompt_details.cache_read",
        description="Prompt tokens read from provider cache; a component of prompt tokens.",
    )
    llm_cost_total: float = Field(
        alias="llm.cost.total",
        description="Model call cost in USD; adapter/provider layers can describe the same billable request.",
    )
    tool_name: str = Field(alias="tool.name", description="Name of the invoked execution tool.")
    tool_id: str = Field(
        alias="tool.id",
        description="Identifier of this tool invocation, used to correlate its model-requested tool call.",
    )
    tool_call_id: str = Field(
        alias="tool_call.id", description="Identifier supplied for the model tool call."
    )
    tool_call_function_name: str = Field(
        alias="tool_call.function.name", description="Function requested by the model tool call."
    )
    tool_call_function_arguments: str = Field(
        alias="tool_call.function.arguments",
        description="JSON-encoded tool arguments; tool-specific input contracts define the decoded shape.",
    )
    tool_json_schema: str = Field(
        alias="tool.json_schema",
        description="JSON tool declaration exposed to the model, including name, description and argument schema.",
    )
    message_role: str = Field(
        alias="message.role",
        description="Role of this indexed model message, e.g. system, user, assistant, or tool.",
    )
    message_content: str | list[dict[str, JsonValue]] = Field(
        alias="message.content",
        description="Text or structured tool-result content blocks of this indexed model message.",
    )
    message_tool_call_id: str = Field(
        alias="message.tool_call_id",
        description="Tool-call identifier to which this message refers.",
    )
    message_content_type: str = Field(
        alias="message_content.type",
        description="Type of this content block, e.g. text or reasoning.",
    )
    message_content_text: str = Field(
        alias="message_content.text",
        description="Text contained in this indexed message content block.",
    )
    message_content_encrypted_content: str = Field(
        alias="message_content.encrypted_content",
        description="Opaque provider-encrypted reasoning content retained for subsequent requests; not readable reasoning text.",
    )

    message_content_id: str = Field(
        alias="message_content.id",
        description="Provider identifier of this message content item, used to correlate reasoning/content across requests.",
    )


class LangSmithFields(FrameworkFields):
    span_id: str = Field(
        alias="OTEL_SPAN_ID",
        description="Original hexadecimal OTLP span ID retained by LangSmith's ingestion mapper.",
    )
    trace_id: str = Field(
        alias="OTEL_TRACE_ID",
        description="Original hexadecimal OTLP trace ID; may differ from LangSmith's UUID trace ID.",
    )
    run_depth: int = Field(
        alias="ls_run_depth",
        description="LangSmith-computed depth in the parent run tree, with root depth zero.",
    )
    model_name: str = Field(
        alias="ls_model_name",
        description="Model name normalized by LangSmith from the model span's OpenInference attributes.",
    )


class LangSmithUsageMetadata(FrameworkFields):
    """Token usage LangSmith adds to model outputs after OTLP ingestion."""

    input_tokens: int = Field(
        ge=0,
        description="Provider-reported input tokens for this model span, including cached input when reported.",
    )
    output_tokens: int = Field(
        ge=0,
        description="Provider-reported output tokens for this model span, including reasoning when reported.",
    )
    total_tokens: int = Field(
        ge=0,
        description="Total tokens for this model span; adapter and nested provider spans can describe the same request and must not be added together.",
    )


class LangSmithTextOutput(FrameworkFields):
    """LangSmith's text wrapper, distinct from the original OTLP output contract."""

    text: str = Field(
        description="Original output text, including serialized message arrays when LangSmith wraps them instead of parsing them."
    )
    usage_metadata: LangSmithUsageMetadata | None = Field(
        default=None,
        description="Token accounting added by LangSmith, not part of the model's answer.",
    )


class LangSmithParsedOutput(BaseModel):
    """Parsed model JSON with LangSmith-added token accounting.

    Other keys belong to the original model response and retain their JSON types.
    Their application-specific meaning comes from that response's schema; Compass
    does not invent a fixed schema for generated answers such as ``value``.
    """

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    usage_metadata: LangSmithUsageMetadata | None = Field(
        default=None,
        description="Token accounting added by LangSmith alongside the parsed response fields.",
    )


# These describe the platform-visible projection, not the OTLP payloads below.
# Empty model content can appear as null outputs even when usage is on the run.
LANGSMITH_MODEL_OUTPUT = TypeAdapter(LangSmithTextOutput | LangSmithParsedOutput | None)


class MethodInput(FrameworkFields):
    args: list[JsonValue] | str = Field(
        description="Positional method arguments; unsupported Python objects become diagnostic strings and oversized lists may become a bounded repr string."
    )
    kwargs: dict[str, JsonValue] | str = Field(
        description="Named method arguments with JSON values or diagnostic strings; may become a bounded repr string when oversized."
    )


class PythonInput(FrameworkFields):
    code: str = Field(
        description="First at most 10,000 characters of Python source submitted to the execution tool; code.length records the full character count."
    )


class PythonOutput(FrameworkFields):
    stdout: str = Field(
        description="Bounded diagnostic formatting of captured standard output; not guaranteed to equal the original byte stream."
    )
    stderr: str = Field(description="Bounded diagnostic formatting of captured standard error.")
    returned_value: str | None = Field(
        description="Bounded Python repr of the returned object, or null for no return/None; not a typed application result."
    )


class ChatMessage(BaseModel):
    """LiteLLM messages support provider extensions beyond the OpenAI API schema.

    Unknown provider extension keys retain JSON values; they are not interpreted
    as Compass fields. Known extensions below occur in actual Compass requests.
    """

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    role: str = Field(
        description="Message role supplied to the model, usually system, user, assistant or tool."
    )
    content: str | list[JsonValue] | None = Field(
        default=None,
        description="Message text or provider-specific multimodal content blocks; null for tool-only assistant turns.",
    )
    tool_calls: list[ChatCompletionMessageToolCallParam] | None = Field(
        default=None, description="Assistant tool calls in the OpenAI chat schema."
    )
    tool_call_id: str | None = Field(
        default=None,
        description="Identifier of the assistant tool call answered by this tool message.",
    )
    name: str | None = Field(
        default=None, description="Optional message participant/function name."
    )
    cache_control: dict[str, JsonValue] | None = Field(
        default=None,
        description="Provider-specific prompt caching directive inserted by Nooa, e.g. ephemeral caching.",
    )
    reasoning_items: list[JsonValue] | None = Field(
        default=None,
        description="Provider reasoning items carried forward by LiteLLM, including opaque encrypted content; not readable explanations.",
    )


class ChatInput(FrameworkFields):
    messages: list[ChatMessage | str] = Field(
        description="Chat messages passed to LiteLLM; message objects may be serialized JSON strings. Model and invocation options are separate OTLP attributes, not this input object."
    )


DiagnosticText = Annotated[
    str,
    Field(
        description="Nooa's bounded human-readable Python representation, potentially abbreviated; not lossless JSON."
    ),
]
ContextText = Annotated[
    str,
    Field(
        description="Full system context or unified diff; nooa.system_message.is_diff distinguishes them."
    ),
]


@dataclass(frozen=True)
class SpanContract:
    """Discoverable payload schemas; None means this span has no payload on that side."""

    description: str
    input: TypeAdapter | None
    output: TypeAdapter | None


def _contract(
    description: str, input_type: object = None, output_type: object = None
) -> SpanContract:
    return SpanContract(
        description,
        TypeAdapter(input_type) if input_type is not None else None,
        TypeAdapter(output_type) if output_type is not None else None,
    )


SPAN_CONTRACTS = {
    "compass": _contract("One CLI analysis run.", RunInput, phase.RunOutput),
    "ingestion": _contract("Load the configured input source.", TraceConfig, phase.IngestionOutput),
    **{
        f"signal.{name}": _contract("One configured Compass signal.", model, phase.SignalOutput)
        for name, model in phase.SIGNAL_INPUT_TYPES.items()
    },
    "validation": _contract(
        "Check candidates against code.", phase.ValidationInput, phase.ValidationOutput
    ),
    "validation.problem": _contract(
        "Check one candidate against code.",
        phase.ProblemValidationInput,
        phase.ProblemValidationOutput,
    ),
    "compilation": _contract(
        "Turn signal candidates into actionable insights.",
        phase.CompilationInput,
        phase.InsightsOutput,
    ),
    "confidence": _contract(
        "Rate compiled insights against code.", phase.ConfidenceInput, phase.ConfidenceOutput
    ),
    "confidence.insight": _contract(
        "Rate one insight against code.",
        phase.InsightConfidenceInput,
        phase.InsightConfidenceResult,
    ),
    "evidence_scan": _contract(
        "Evaluate an insight predicate over input traces.",
        phase.EvidenceScanInput,
        phase.EvidenceScanResult | phase.EvidenceScanFailure,
    ),
    "method.*": _contract(
        "Nooa agent method invocation; payloads are framework diagnostic representations.",
        MethodInput | DiagnosticText,
        DiagnosticText,
    ),
    "agent.execute": _contract(
        "Nooa generation session; model calls and tools are its children, not separate agent runs.",
        None,
        DiagnosticText,
    ),
    "agent.context": _contract("Snapshot or diff of rendered agent system context.", ContextText),
    "code_execution": _contract(
        "Execute model-authored Python.",
        PythonInput | DiagnosticText,
        PythonOutput | DiagnosticText,
    ),
    "tool_execution.return_result": _contract(
        "Return the generation result to the caller.",
        Annotated[
            dict[str, JsonValue] | str,
            Field(
                description="Arguments supplied to return_result; keys follow the generated return schema for this method, with bounded diagnostic strings for unsupported Python values."
            ),
        ],
        DiagnosticText,
    ),
    "llm.chat_completion": _contract(
        "LiteLLM chat-completion adapter call; may contain a provider Responses child for the same request.",
        ChatInput,
        str,
    ),
    "llm.responses": _contract(
        "Provider Responses API call beneath the chat-completion adapter; do not add both layers' usage for totals.",
        str | ResponseInputParam,
        list[Message],
    ),
}

SPAN_CONTRACTS["agent.execute.*"] = SPAN_CONTRACTS["agent.execute"]

SPAN_CONTRACTS["agent.context_snapshot"] = _contract(
    "Complete rendered system context for the agent, not an incremental update.", ContextText
)
SPAN_CONTRACTS["agent.context_diff"] = _contract(
    "Unified diff against the preceding rendered system context in the same agent call.",
    ContextText,
)

HARNESS_METRICS_MODEL = HarnessMetrics

FRAMEWORK_FIELD_MODELS = (
    AgentFields,
    GenerationFields,
    ExecutionFields,
    ContextFields,
    DiagnosticFields,
    ModelExtensionFields,
    HarnessFields,
    LangSmithFields,
    StandardFields,
)


@dataclass(frozen=True)
class AttributeDefinition:
    """Type owner and meaning of one exported flat attribute."""

    owner: str
    description: str
    python_type: object


def attribute_definition(key: str) -> AttributeDefinition | None:
    """Resolve Compass, Nooa, LangSmith, and indexed OpenInference field names."""
    models = (*FRAMEWORK_FIELD_MODELS, ExecutionMetadata)
    lookup = key
    if key.startswith("compass.input."):
        models = (InputMetadataSummary,)
        lookup = key.removeprefix("compass.input.")
    for model in models:
        for name, field in model.model_fields.items():
            if lookup == (field.alias or name):
                return AttributeDefinition(
                    model.__name__, field.description or "", field.annotation
                )
    if key.startswith("harness.time."):
        operation, _, member = key.removeprefix("harness.time.").partition(".")
        if operation in TIMING_OPERATIONS and member in TimingFields.model_fields:
            field = TimingFields.model_fields[member]
            return AttributeDefinition(
                "TimingFields",
                f"{TIMING_OPERATIONS[operation]}. {field.description}",
                field.annotation,
            )
    for entry in _SPAN_SCHEMA:
        if key == entry.key:
            return AttributeDefinition(
                "nooa.runtime.harness_metrics.HarnessMetrics",
                f"{entry.category}: {entry.label}. Per generation session; zero/default values are omitted. Timing values ending _s are seconds; samples are capped at 20. See upstream schema for the exact projection.",
                JsonValue,
            )
    # Indexed message and tool keys compose upstream constants with array indices.
    normalized = ".".join(part for part in key.split(".") if not part.isdecimal())
    for field in StandardFields.model_fields.values():
        if normalized == field.alias or normalized.endswith("." + str(field.alias)):
            return AttributeDefinition("StandardFields", field.description or "", field.annotation)
    for cls in (
        semantics.SpanAttributes,
        semantics.MessageAttributes,
        semantics.MessageContentAttributes,
        semantics.ToolCallAttributes,
        semantics.ToolAttributes,
    ):
        for name, value in vars(cls).items():
            if (
                name.isupper()
                and isinstance(value, str)
                and (normalized == value or normalized.endswith("." + value))
            ):
                return AttributeDefinition(
                    f"openinference.semconv.trace.{cls.__name__}.{name}",
                    "OpenInference standard field; its canonical meaning is defined by this semantic-convention constant and provider SDK payload types.",
                    JsonValue,
                )
    return None
