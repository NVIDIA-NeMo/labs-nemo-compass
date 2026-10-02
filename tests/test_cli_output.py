# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from insight_agent.cli.output import RunOutput, RunResult
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult, Problem
from insight_agent.insight import Insight


def make_insight(name="Agent calls tools absent from the active catalog"):
    return Insight.model_validate(
        {
            "name": name,
            "description": "A long explanation that belongs in the saved report.",
            "evidence": [
                {"trace_id": f"trace-{i}", "spans": [{"span_id": f"span-{i}"}]} for i in range(3)
            ],
        }
    )


@pytest.mark.parametrize("width", [40, 80])
def test_default_report_prioritizes_findings_and_keeps_coverage_visible(width):
    stream = StringIO()
    result = RunResult(
        200,
        [
            *[
                EvidenceStreamResult(
                    stream_name=name,
                    problems=tuple(
                        Problem(description=f"Candidate {i}", supporting_trace_ids=("trace-0",))
                        for i in range(candidates)
                    ),
                    limitations=("Missing schemas",) if name == "tool-issues" else (),
                )
                for name, candidates in (("anomaly-and-patterns", 5), ("tool-issues", 3))
            ],
            *[
                EvidenceStreamResult(stream_name=name, problems=(), skip_reason=reason)
                for name, reason in (
                    ("ethos-divergence", "No ethos document"),
                    ("eval-failure-patterns", "No evaluator results"),
                    ("user-sentiment", "No embedding backend configured"),
                )
            ],
        ],
        [make_insight()],
    )
    RunOutput(Console(file=stream, width=width)).report(result, Path("insights.yml"))
    rendered = stream.getvalue()
    assert "1 insight from 200 traces" in rendered
    assert "Evidence: 3 supporting traces" in rendered
    assert rendered.index("Agent calls") < rendered.index("Coverage") < rendered.index("Saved:")
    normalized = " ".join(rendered.split())
    assert "Ran: Patterns (5 candidates), tool issues (3 candidates; Missing schemas)" in normalized
    assert (
        "Skipped: Ethos - no document Evaluation - no results Sentiment - no embedding backend"
        in normalized
    )
    assert (
        "To run skipped checks: https://github.com/NVIDIA-NeMo/labs-trace-intel/blob/main/docs/evidence-streams.md"
        in rendered
    )
    for hidden in ("trace-0", "span-0", "long explanation", "--details", "\x1b"):
        assert hidden not in rendered
    # Keep the setup URL intact for copying; other content wraps to the terminal width.
    assert all(len(line) <= width for line in rendered.splitlines() if "https://" not in line)


def test_preview_is_bounded_and_new_insights_come_first_without_changing_result():
    old = [make_insight(f"Existing issue {i}") for i in range(8)]
    new = make_insight("New issue")
    result = RunResult(
        200,
        [EvidenceStreamResult(stream_name="tool-issues", problems=(), finding_count=2)],
        [*old, new],
        existing_insights=old,
    )
    stream = StringIO()
    output = RunOutput(Console(file=stream, width=120))
    output.report(result, Path("insights.yml"))
    rendered = stream.getvalue()
    assert rendered.index("New issue") < rendered.index("Existing issue 0")
    assert "Existing issue 3" in rendered
    assert "Existing issue 4" not in rendered
    assert "4 more insights in YAML output." in rendered
    assert "Tool issues (2 findings; no candidates)" in rendered
    assert "Skipped" not in rendered and "To run skipped checks" not in rendered
    assert result.insights == [*old, new]

    stream.seek(0)
    stream.truncate()
    output.report(result, Path("-"), details=True)
    rendered = stream.getvalue()
    assert "Existing issue 7" in rendered
    assert "long explanation" in rendered and "Span span-0" in rendered
    assert "more insights" not in rendered and "Saved:" not in rendered


def test_no_traces_reports_missing_coverage_and_no_output_file():
    stream = StringIO()
    result = RunResult(
        0,
        [EvidenceStreamResult(stream_name="tool-issues", problems=(), skip_reason="No tool calls")],
        [],
    )
    RunOutput(Console(file=stream, width=120)).report(result, Path("insights.yml"))
    rendered = stream.getvalue()
    assert "Ran:      None" in rendered
    assert "Skipped:  Tool issues - No tool calls" in rendered
    assert "No output file written." in rendered
    assert "Saved:" not in rendered
    assert "No traces loaded. Check your source, filters, and time window." in rendered
