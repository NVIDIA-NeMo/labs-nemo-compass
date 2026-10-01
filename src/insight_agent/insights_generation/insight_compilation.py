# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Literal

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy
from nooa.strategy_validation import InvariantError
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from insight_agent.evidence_streams._trace import walk_spans
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult
from insight_agent.insight import Insight
from insight_agent.traces import Trace, TraceSnapshot

Decision = Literal["match", "no_match", "unknown"]


class EvidenceDecision(BaseModel):
    """A predicate's decision with witnesses in the canonical trace representation."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    status: Decision
    witness_span_ids: tuple[str, ...] = ()


class EvidenceCompletion(BaseModel):
    """Transient compiler output; predicates and checks are never persisted as insights."""

    insight: Insight
    # Executable callables cross the agent's Python boundary, not its JSON tool schema.
    predicate: SkipJsonSchema[Callable[[Trace], EvidenceDecision | dict[str, object]] | None] = (
        Field(default=None, exclude=True, repr=False)
    )
    checks: list[tuple[Trace, Decision]] = Field(default_factory=list, exclude=True, repr=False)
    unresolved_reason: str | None = None

    def _evaluate(self, trace: Trace) -> EvidenceDecision:
        assert self.predicate is not None
        span_ids = {visit.span.id for visit in walk_spans(trace)}
        decision = EvidenceDecision.model_validate(self.predicate(trace))
        if decision.status == "match" and not decision.witness_span_ids:
            raise ValueError(f"Match in {trace.id} has no source witnesses")
        if set(decision.witness_span_ids) - span_ids:
            raise ValueError(f"Witness does not exist in trace {trace.id}")
        return decision

    def check(self) -> None:
        """Validate the completion outcome and execute its acceptance checks."""
        if self.predicate is None:
            if not self.unresolved_reason or not self.unresolved_reason.strip() or self.checks:
                raise ValueError("Abstention requires a nonblank unresolved_reason and no checks")
            return
        if not callable(self.predicate) or self.unresolved_reason is not None:
            raise ValueError("Completion requires a live callable and no unresolved_reason")
        if {expected for _, expected in self.checks} != {"match", "no_match", "unknown"}:
            raise ValueError("Check a positive, a close negative, and missing evidence first")
        for trace, expected in self.checks:
            actual = self._evaluate(trace).status
            if actual != expected:
                raise ValueError(f"Check {trace.id}: expected {expected}, got {actual}")

    def scan(self, snapshot: TraceSnapshot) -> tuple[list[str], int]:
        """Run acceptance checks, then the whole snapshot; any failure rejects additions."""
        self.check()
        if self.predicate is None:
            raise ValueError(self.unresolved_reason)
        matches = []
        unknown = 0
        for trace in snapshot:
            trace_id = trace.id
            decision = self._evaluate(trace)
            if decision.status == "match":
                matches.append(trace_id)
            unknown += decision.status == "unknown"
        if not matches:
            raise ValueError(
                "No snapshot match reproduced the positive check; diagnose the method or seed"
            )
        return matches, unknown

    def apply(self, snapshot: TraceSnapshot) -> Insight:
        """Attach only a successful scan's additions; report unresolved completion."""
        try:
            matches, unknown = self.scan(snapshot)
        except Exception as error:
            warnings.warn(
                f"Evidence completion unresolved for {self.insight.name!r}: "
                f"{type(error).__name__}: {str(error)[:300]}. Initial citations retained; "
                "no completion additions attached.",
                stacklevel=2,
            )
            return self.insight
        if unknown:
            warnings.warn(
                f"Evidence completion unresolved for {self.insight.name!r}: "
                f"{unknown} of {len(snapshot)} traces lack sufficient interpretable evidence. "
                "Only verified matches attached; semantic completeness is not established.",
                stacklevel=2,
            )
        return self.insight.model_copy(
            update={"trace_refs": list(dict.fromkeys([*self.insight.trace_refs, *matches]))}
        )


def _validate_completions(
    _agent: Agent, completions: list[EvidenceCompletion], _call: object
) -> None:
    for completion in completions:
        try:
            completion.check()
        except Exception as error:
            raise InvariantError(
                f"{completion.insight.name}: {error}. "
                "Call return_result with the existing live completion objects inside Python, "
                "rather than rebuilding them as JSON. To abstain, omit the predicate and "
                "checks and supply a nonblank unresolved_reason."
            ) from error


class InsightCompilation(Agent):
    async def compile_insights(
        self,
        evidence_streams: list[EvidenceStreamResult],
        trace_snapshot: TraceSnapshot,
        existing_insights: list[Insight],
    ) -> list[Insight]:
        completions = await self._compile_insights(
            evidence_streams, trace_snapshot, existing_insights
        )
        return [completion.apply(trace_snapshot) for completion in completions]

    @strategy(CodeActStrategy(config=CodeActConfig(postconditions=[_validate_completions])))
    async def _compile_insights(
        self,
        evidence_streams: list[EvidenceStreamResult],
        trace_snapshot: TraceSnapshot,
        existing_insights: list[Insight],
    ) -> list[EvidenceCompletion]:  # ty: ignore[empty-body] -- Nooa generates the ellipsis method at runtime.
        """
        Your job is to be the last step of the insight creation process. An
        insight is a bug report or problem that is identified by looking at the
        runtime traces of an AI agent, either in production or an offline
        evaluation context.

        The goal of an insight is to show a developer a problem with their agent
        that they can fix by making a change to their code. Often times this
        could be updating the prompt, changing time outs, fixing code logic,
        updating their agent to use a different model, etc.

        The evidence stream is a set of potential problems that have been
        surfaced by earlier stages. These may be real problems, and they may not
        be fixable.

        Insights should be ranked holistically based on the following factors:
        1. How fixable is this issue by the agent developer?

        2. How high is the  impact of this issue? For example if the formatting
        is slightly incorrect or if there is a contradiction that's not the end
        of the world. If the agent is failing to even produce a response for the
        user 30% of the time though because of an error, that's a huge deal!

        Your job is to validate that these insights are all impactful and
        fixable. If any proposed problem from an Evidence Stream doesn't meet
        this bar, it should be discarded. Some things are not ideal, but are
        recovered by an agent -- for example sometimes an agent will call a
        coding tool that fails, but then go on to recover. There's not anything
        we can do about this issue! And it's OK. Experts do this too. You can
        validate by examining the referenced traces. Every new insight we create
        must have more than one trace that supports it. We want to identify
        problems that are broader in scope than a one-off.

        Inspect supporting traces in bounded excerpts with
        trace_snapshot.get_trace_by_id(trace_id); use Python for snapshot scans.

        After validating, you must merge the new insights with the existing
        insights, and across evidence streams.

        Here's an example of two insights that are semantic duplicates and
        should be merged:

        1. When users request their airline ticket to be canceled, the system
        prompt allows it but there isn't a tool for it, so the agent tells the
        user that the ticket can't be canceled

        2. Ticket cancelation tool is missing from tool schema

        Here's an example of two insights that should remain independent:

        1. Ticket cancellation calls time out for users that are already checked
        in

        2. Ticket cancellation causes an error when the user has a pending
        refund

        Here's an example of an insight that is overly broad. This kind of
        insight should never be created by you, but if there is an existing one
        in the existing_insights, you should keep it and not modify it.

        1. Bash tool call is returning errors

        Preserve each existing insight's id exactly. Never invent an id or
        assign an existing id to a new insight; new insights have id=None.
        Existing insights should never be removed or modified, but you can
        update the trace_refs on existing insights to match new traces you
        identified. Preserve historical refs, including those outside this snapshot.
        When you merge insights, newly combined refs must support the final merged
        claim; do not blindly union refs or candidates.

        Leave trace_links empty; the application resolves source links after compilation.

        After validation, narrowing, and merging, complete evidence for each final
        insight. Keep initial trace_refs at validated examples and historical refs;
        the application adds completion matches. Use the following workflow for every
        snapshot size. Example inspection aids predicate development; neither direct
        review nor similarity retrieval replaces it.

        1. Fix the claim. State its observable requirements in working analysis:
           operation, trigger, behavior, joins, order, recovery, duration, impact,
           exclusions, and missing evidence. An error occurrence alone cannot prove
           causality or task failure. Never weaken the claim to gain citations. If
           material qualifications cannot be faithfully checked, preserve the insight
           and return predicate=None, no checks, and a nonblank unresolved_reason.

        2. Develop a pure predicate(trace) against the actual Trace schema. Use
           supporting_trace_ids, complete candidate_trace_ids, and stream artifacts
           for examples and discrepancy investigation. Fetch with
           trace_snapshot.get_trace_by_id(id); walk_spans(trace) yields visits with
           .span. Inspect bounded excerpts, keeping full data in Python. Preserve
           source identities and event order; do not mutate source traces. Make the
           callable self-contained, with needed imports inside it. Never hardcode
           trace IDs, expected counts, or answer lists.

           Return EvidenceDecision(status=..., witness_span_ids=(...)):
           - match: recorded events establish every material condition; cite their
             canonical span IDs. Membership, keywords, and copied history alone are
             insufficient. Anchor copied records to their original events.
           - no_match: sufficient evidence rules out the claim.
           - unknown: required evidence is unavailable or uninterpretable, unless
             another observed condition already rules out the claim.

        3. Build compact checks=[(Trace, expected_status), ...]: an established
           positive, close negatives, and missing required evidence. For counterfactuals,
           use model_copy(deep=True), change one material condition, and leave others
           satisfied. Remove individual required fields from matching examples rather
           than only testing empty traces. Diagnose seed mismatches: the predicate,
           extraction, or original citation may be wrong. Do not relax the claim to
           make seeds pass.

        4. Create EvidenceCompletion(insight=initial_insight, predicate=predicate,
           checks=checks) and run .scan(trace_snapshot). It executes the checks,
           validates result shape and source witnesses, and scans the entire selected
           snapshot, including traces absent from upstream candidates. On execution
           failure or failed checks, make at most one predicate repair and rerun.
           If still invalid, return predicate=None with unresolved_reason; discard
           all proposed additions.
           A broken method's zero matches do not establish absence.

        Return live completions through return_result inside Python, without manually
        adding results to trace_refs. Return validation can request a correction while
        the session is live; restoring a callable does not consume its predicate repair.
        Finalization reruns checks and scans, attaches deduplicated matches, retains
        initial citations on failure, and warns about unresolved completion. Report
        inability to complete through unresolved_reason, without a predicate or checks.
        With a usable predicate, unknown decisions report partial coverage while matches
        remain attachable; do not also set unresolved_reason. Valid references do not
        prove semantic support; scanning establishes coverage under the checked predicate,
        not exhaustive truth. Keep interpretation and coverage limits in working analysis.

        Return one EvidenceCompletion per ranked, validated and merged insight.
        Include existing insights.
        """
        ...


__all__ = ["Insight", "InsightCompilation"]
