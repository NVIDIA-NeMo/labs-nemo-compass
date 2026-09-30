# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Literal

from nooa import Agent
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

    def scan(self, snapshot: TraceSnapshot) -> tuple[list[str], int]:
        """Run checks, then the whole snapshot. Any failure invalidates all additions.

        Call during development for execution feedback; finalization reruns this
        before attaching refs. Checks include a source positive, a close negative,
        and missing evidence, using Trace copies for counterfactuals when needed.
        """
        if self.unresolved_reason or self.predicate is None:
            raise ValueError(self.unresolved_reason or "No checked predicate supplied")
        if {expected for _, expected in self.checks} != {"match", "no_match", "unknown"}:
            raise ValueError("Check a positive, a close negative, and missing evidence first")
        for trace, expected in self.checks:
            actual = self._evaluate(trace).status
            if actual != expected:
                raise ValueError(f"Check {trace.id}: expected {expected}, got {actual}")
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

        Inspect bounded trace excerpts with trace_snapshot.get_trace_by_id(trace_id).
        Use the available Python execution capability to iterate trace_snapshot
        for programmatic searches and verification without printing entire traces
        or datasets into context. Inspect compact witnesses and ambiguous cases.

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
        identified. Preserve historical refs already attached to existing insights,
        including those outside this snapshot; their absence here is not a reason
        to remove them. Verify newly combined or attached refs against the final
        merged claim rather than blindly unioning refs or candidates.

        After validation, narrowing, and merging, develop, execute, and check a
        matching method for each final insight. Use this same workflow regardless
        of snapshot or candidate-set size; do not substitute direct review or
        similarity/top-k retrieval as the completion strategy.

        1. State the fixed claim's observable requirements in your working analysis:
           operation, trigger, behavior, joins, order, recovery, duration, impact,
           exclusions, and missing-evidence conditions. Do not weaken material
           qualifications to gain citations. An error occurrence does not prove
           causality or task-level failure. If intent, causality, or conversational
           judgment cannot be faithfully checked, return an EvidenceCompletion
           with unresolved_reason and no predicate. Keep the insight unchanged.
        2. Inspect bounded examples from supporting_trace_ids and complete
           candidate_trace_ids, using artifacts for provenance. Develop a pure
           Python predicate(trace) against the actual canonical Trace representation;
           research event views are not supplied. Make the callable self-contained:
           import needed standard-library modules inside it rather than relying on
           temporary cell variables. Preserve required fields, source
           identities, joins, and order. walk_spans(trace) yields visits with .span.
           Do not hardcode trace IDs, expected counts, or dataset answer lists.
        3. Return EvidenceDecision(status="match"|"no_match"|"unknown",
           witness_span_ids=(...)) from the predicate. A match needs actual canonical
           span IDs witnessing every material condition. Group membership, keywords,
           and copied history alone do not suffice; anchor copied records to the
           original event. no_match requires sufficient evidence ruling the claim
           out. Unavailable or uninterpretable required evidence is unknown, unless
           another observed condition already rules out the conjunction.
        4. Build a compact checks list of (Trace, expected_status) pairs: an
           established source positive, close counterexamples violating material
           conditions, and missing required evidence. Use model_copy(deep=True)
           when making counterfactuals. Keep other conditions satisfied when
           testing a boundary; remove individual required fields from otherwise
           matching examples rather than only testing an empty trace. Choose
           discriminating checks, not a large generated suite: e.g. a different
           operation, a successful intervening call, different artifact, or missing
           duration as relevant to the claim.
           A seed mismatch requires diagnosis of the rule, extraction, or citation;
           never relax the claim just to match every supplied seed.
        5. Create EvidenceCompletion(insight=initial_insight, predicate=predicate,
           checks=checks) and execute its scan(trace_snapshot). This checks output
           shape and source witnesses and runs the full selected snapshot, not just
           upstream candidates. Use failures as feedback for at most one repair of
           the method, then rerun the checks and scan. Syntax/runtime failures or
           failed checks invalidate the attempt and all its proposed additions;
           broken code returning zero is not evidence of absence. If still invalid,
           set predicate=None and explain unresolved_reason. Do not return an
           invalid method's plausible-looking subset as verified evidence.

        Keep initial_insight.trace_refs at the validated initial examples (and
        preserved historical refs), without manually adding completion results.
        Return EvidenceCompletion objects: the application reruns their checks and
        scans before attaching deduplicated matches, and warns on failure or unknowns.
        Never mutate source traces. If a method cannot be developed or processing
        is partial, report unresolved_reason rather than silently claiming completion.
        Source-reference validity is not semantic verification. A successful scan
        establishes coverage under the checked predicate within this snapshot,
        not exhaustive semantic truth. Keep interpretation and coverage limits explicit
        in your working analysis; do not introduce stored claim contracts.

        Leave trace_links empty; the application resolves source links after compilation.

        Return one EvidenceCompletion per ranked, validated and merged insight.
        Include existing insights, preserving their identity, prose, and historical refs.
        """
        ...


__all__ = ["Insight", "InsightCompilation"]
