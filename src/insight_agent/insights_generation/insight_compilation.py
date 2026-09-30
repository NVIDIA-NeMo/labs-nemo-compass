# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nooa import Agent

from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult
from insight_agent.insight import Insight
from insight_agent.traces import TraceSnapshot


class InsightCompilation(Agent):
    async def compile_insights(
        self,
        evidence_streams: list[EvidenceStreamResult],
        trace_snapshot: TraceSnapshot,
        existing_insights: list[Insight],
    ) -> list[Insight]:  # ty: ignore[empty-body] -- Nooa generates the ellipsis method at runtime.
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

        After validating, narrowing, and merging, complete the evidence for each
        final insight within the selected snapshot:
        1. Fix the insight's meaning. Establish the exact conditions recorded
           events must satisfy, including any temporal sequence, context, recovery,
           duration, or impact asserted by the claim. Do not broaden the claim
           during evidence expansion to accommodate more traces.
        2. Examine all known upstream membership: supporting_trace_ids and
           candidate_trace_ids on relevant Problems, using stream artifacts for
           provenance. Candidates are leads, not verified support; an empty
           candidate collection does not imply no other matches exist.
        3. Search the rest of trace_snapshot for additional occurrences satisfying
           the same conditions, even when no stream cited them. Where reliable,
           use programmatic checks grounded in recorded events, joining evidence
           across steps/spans by recorded identity when required. Group membership,
           shared keywords, and copied history alone are insufficient witnesses.
           Do not assume every semantic claim has a reliable mechanical predicate;
           inspect context when a scan cannot establish the claim.
        4. Return every verified matching trace ID in trace_refs, deduplicated.
           Missing required telemetry is unresolved, not a match or proof of
           absence. Keep compact verification counts and unresolved/coverage limits
           in your working analysis; a completed scan alone does not establish
           exhaustive semantic coverage. Never attach unresolved candidates.

        Leave trace_links empty; the application resolves source links after compilation.

        Return the list of ranked, validated and merged insights. Include
        existing insights.
        """
        ...


__all__ = ["Insight", "InsightCompilation"]
