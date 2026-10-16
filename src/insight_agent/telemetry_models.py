# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical contracts for Compass-owned span inputs and outputs.

Field descriptions define the exported JSON keys. These contracts describe work
performed by Compass; source records are not synthetic execution spans.
"""

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
    trace_count: int = Field(ge=0, description="Number of input traces loaded for this run.")


SignalConfig = TypeVar("SignalConfig", bound=BaseModel)


class SignalInput(TelemetryModel, Generic[SignalConfig]):
    trace_count: int = Field(ge=0, description="Number of input traces available to this signal.")
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
    status: Literal["completed", "skipped"] = Field(
        description="Whether this signal analyzed the inputs or skipped due to unavailable prerequisites."
    )
    problems: tuple[Problem, ...] = Field(
        description="Candidate issues found by this signal, before code validation and insight compilation."
    )
    skip_reason: str | None = Field(
        default=None, description="Why prerequisites prevented analysis; null when analysis ran."
    )
    limitations: tuple[str, ...] = Field(
        description="Known gaps in the input or analysis that limit interpretation of these candidates."
    )

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
    description: str = Field(
        description="Original candidate problem description, without source trace contents."
    )


class SourceCodeValidationInput(TelemetryModel):
    problems: list[ProblemSummary] = Field(
        description="Candidate problems to check against the codebase, with their originating signals."
    )
    code_base: str = Field(description="Local codebase directory inspected during validation.")


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
    problem: Problem = Field(
        description="Candidate problem being checked, with its source trace references."
    )
    code_base: str = Field(description="Local codebase directory inspected for this candidate.")


class ProblemValidationOutput(TelemetryModel):
    supported: bool | None = Field(
        description="True means code supports the candidate; false means code contradicts it; null means unresolved and retained."
    )


class SignalCandidates(TelemetryModel):
    signal: str = Field(description="Signal configuration key that produced these candidates.")
    problems: tuple[Problem, ...] = Field(
        description="Candidate problems available to compilation after any code validation."
    )
    limitations: tuple[str, ...] = Field(description="Known limitations of this signal's analysis.")


class CompilationInput(TelemetryModel):
    signals: list[SignalCandidates] = Field(
        description="Candidates grouped by originating signal, without diagnostic artifacts."
    )
    existing_insights: list[Insight] = Field(
        description="Previously saved insights that the compiler can preserve or extend."
    )


class InsightsOutput(TelemetryModel):
    insights: list[Insight] = Field(description="Actionable insights produced by this phase.")


class ConfidenceInput(TelemetryModel):
    insight_count: int = Field(
        ge=0, description="Number of compiled insights being checked against the codebase."
    )
    code_base: str = Field(description="Local codebase directory inspected when rating confidence.")


class InsightConfidenceInput(TelemetryModel):
    insight_name: str = Field(description="Name of the insight being checked against code.")
    description: str = Field(description="Claim whose support in the codebase is being assessed.")
    code_base: str = Field(
        description="Local codebase directory used for the confidence assessment."
    )


class InsightConfidenceResult(TelemetryModel):
    insight_name: str = Field(description="Name of the compiled insight being rated.")
    confidence: Rating | None = Field(
        description="Code support: high means behavior, root cause, scope, and remediation hold up; med means behavior is supported but another part is wrong or unconfirmed; low means the central claim is contradicted. Null means no rating was obtained."
    )


class ConfidenceOutput(TelemetryModel):
    ratings: list[InsightConfidenceResult] = Field(
        description="Code-based confidence assigned to each final insight."
    )


class RunOutput(TelemetryModel):
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limits of this run: skipped signals, missing input capabilities, unresolved code checks, and incomplete evidence scans.",
    )
    trace_count: int = Field(ge=0, description="Number of source traces loaded for this run.")
    insights: list[Insight] = Field(
        description="Final insights written to the configured output destination."
    )
    rejected_by_code: int = Field(
        ge=0,
        description="Number of candidate problems excluded because code validation returned unsupported.",
    )


class EvidenceScanInput(TelemetryModel):
    insight_name: str = Field(
        description="Insight whose evidence predicate is evaluated over the input corpus."
    )
    trace_count: int = Field(
        ge=0, description="Number of source traces available to the evidence scan."
    )


class EvidenceScanResult(TelemetryModel):
    status: Literal["complete", "incomplete"] = Field(
        description="Complete means every trace received a match/no-match decision; incomplete means some decisions were unknown."
    )
    match_count: int = Field(
        ge=0, description="Number of source traces matching the insight's evidence predicate."
    )
    unknown_count: int = Field(
        ge=0,
        description="Number of source traces for which the predicate could not determine a match.",
    )


class EvidenceScanFailure(TelemetryModel):
    status: Literal["skipped", "failed"] = Field(
        description="Skipped means no predicate was supplied with an explanation; failed means the predicate or its checks raised an error. Existing evidence is retained."
    )
    reason: str = Field(
        description="Explanation for skipping, or the exception message for failure (up to 300 characters)."
    )
    error_type: str | None = Field(
        description="Exception class for a failed scan; null for an explicitly skipped scan."
    )
