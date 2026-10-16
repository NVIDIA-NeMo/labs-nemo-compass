# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed inputs and outputs for Compass-owned spans."""

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from insight_agent.compass_signals.compass_signals import CompassSignalResult, Problem
from insight_agent.config import (
    AnomalyAndPatternsSignalConfig,
    EthosDivergenceConfig,
    GraderFailurePatternsConfig,
    ToolIssueSignalConfig,
    UserSentimentConfig,
)
from insight_agent.insight import Insight, Rating


class TelemetryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestionOutput(TelemetryModel):
    trace_count: int = Field(ge=0)


SignalConfig = TypeVar("SignalConfig", bound=BaseModel)


class SignalInput(TelemetryModel, Generic[SignalConfig]):
    trace_count: int = Field(ge=0)
    config: SignalConfig


class EthosSignalInput(SignalInput[EthosDivergenceConfig]):
    ethos: str | None = None


SIGNAL_INPUT_TYPES = {
    "anomaly_and_patterns": SignalInput[AnomalyAndPatternsSignalConfig],
    "tool_issues": SignalInput[ToolIssueSignalConfig],
    "ethos_divergence": EthosSignalInput,
    "grader_failure_patterns": SignalInput[GraderFailurePatternsConfig],
    "user_sentiment": SignalInput[UserSentimentConfig],
}


class SignalOutput(TelemetryModel):
    status: Literal["completed", "skipped"]
    problems: tuple[Problem, ...]
    skip_reason: str | None = None
    limitations: tuple[str, ...]

    @classmethod
    def from_result(cls, result: CompassSignalResult) -> "SignalOutput":
        return cls(
            status="skipped" if result.skip_reason else "completed",
            problems=result.problems,
            skip_reason=result.skip_reason,
            limitations=result.limitations,
        )


class ProblemSummary(TelemetryModel):
    signal: str
    description: str


class SourceCodeValidationInput(TelemetryModel):
    problems: list[ProblemSummary]
    code_base: str


class SourceCodeValidationOutput(TelemetryModel):
    supported: list[ProblemSummary] = Field(default_factory=list)
    unsupported: list[ProblemSummary] = Field(default_factory=list)
    unresolved: list[ProblemSummary] = Field(default_factory=list)


class ProblemValidationInput(TelemetryModel):
    problem: Problem
    code_base: str


class ProblemValidationOutput(TelemetryModel):
    supported: bool | None


class SignalCandidates(TelemetryModel):
    signal: str
    problems: tuple[Problem, ...]
    limitations: tuple[str, ...]


class CompilationInput(TelemetryModel):
    signals: list[SignalCandidates]
    existing_insights: list[Insight]


class InsightsOutput(TelemetryModel):
    insights: list[Insight]


class ConfidenceInput(TelemetryModel):
    insight_count: int = Field(ge=0)
    code_base: str


class InsightConfidenceInput(TelemetryModel):
    insight_name: str
    description: str
    code_base: str


class InsightConfidenceResult(TelemetryModel):
    insight_name: str
    confidence: Rating | None


class ConfidenceOutput(TelemetryModel):
    ratings: list[InsightConfidenceResult]


class RunOutput(TelemetryModel):
    limitations: list[str] = Field(default_factory=list)
    trace_count: int = Field(ge=0)
    insights: list[Insight]
    rejected_by_code: int = Field(ge=0)


class EvidenceScanInput(TelemetryModel):
    insight_name: str
    trace_count: int = Field(ge=0)


class EvidenceScanResult(TelemetryModel):
    status: Literal["complete", "incomplete"]
    match_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)


class EvidenceScanFailure(TelemetryModel):
    status: Literal["skipped", "failed"]
    reason: str
    error_type: str | None
