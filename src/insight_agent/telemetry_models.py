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
    config: SignalConfig = Field(
        description="Effective settings for this signal, including defaults."
    )


class EthosSignalInput(SignalInput[EthosDivergenceConfig]):
    ethos: str | None = Field(
        default=None,
        description="Exact ethos text supplied to this analysis; null when analysis has not started or was skipped.",
    )


SIGNAL_INPUT_TYPES = {
    "anomaly_and_patterns": SignalInput[AnomalyAndPatternsSignalConfig],
    "tool_issues": SignalInput[ToolIssueSignalConfig],
    "ethos_divergence": EthosSignalInput,
    "grader_failure_patterns": SignalInput[GraderFailurePatternsConfig],
    "user_sentiment": SignalInput[UserSentimentConfig],
}


class SignalOutput(TelemetryModel):
    status: Literal["completed", "skipped"]
    problems: tuple[Problem, ...] = Field(
        description="Candidate issues found by this signal, before code validation and insight compilation."
    )
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
    signal: str = Field(description="Configuration key of the signal that reported this problem.")
    description: str


class SourceCodeValidationInput(TelemetryModel):
    problems: list[ProblemSummary]
    code_base: str


class SourceCodeValidationOutput(TelemetryModel):
    supported: list[ProblemSummary] = Field(
        default_factory=list,
        description="Problems supported by inspected code and retained for compilation.",
    )
    unsupported: list[ProblemSummary] = Field(
        default_factory=list,
        description="Problems contradicted by inspected code and excluded from compilation.",
    )
    unresolved: list[ProblemSummary] = Field(
        default_factory=list,
        description="Problems whose support could not be determined; retained for compilation.",
    )


class ProblemValidationInput(TelemetryModel):
    problem: Problem
    code_base: str


class ProblemValidationOutput(TelemetryModel):
    supported: bool | None = Field(
        description="True means code supports the candidate; false means code contradicts it; null means unresolved and retained."
    )


class SignalCandidates(TelemetryModel):
    signal: str
    problems: tuple[Problem, ...] = Field(
        description="Candidate problems available to compilation after any code validation."
    )
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
    confidence: Rating | None = Field(
        description="Code support: high means behavior, root cause, scope, and remediation hold up; med means behavior is supported but another part is wrong or unconfirmed; low means the central claim is contradicted. Null means no rating was obtained."
    )


class ConfidenceOutput(TelemetryModel):
    ratings: list[InsightConfidenceResult]


class RunOutput(TelemetryModel):
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limits of this run: skipped signals, missing input capabilities, unresolved code checks, and incomplete evidence scans.",
    )
    trace_count: int = Field(ge=0)
    insights: list[Insight]
    rejected_by_code: int = Field(ge=0)


class EvidenceScanInput(TelemetryModel):
    insight_name: str
    trace_count: int = Field(ge=0)


class EvidenceScanResult(TelemetryModel):
    status: Literal["complete", "incomplete"] = Field(
        description="Complete means every trace received a match/no-match decision; incomplete means some decisions were unknown."
    )
    match_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)


class EvidenceScanFailure(TelemetryModel):
    status: Literal["skipped", "failed"] = Field(
        description="Skipped means no predicate was supplied with an explanation; failed means the predicate or its checks raised an error. Existing evidence is retained."
    )
    reason: str = Field(
        description="Explanation for skipping, or the exception message for failure (up to 300 characters)."
    )
    error_type: str | None
