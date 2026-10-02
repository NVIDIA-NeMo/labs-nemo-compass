# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from insight_agent.cli.output import RunOutput, RunResult
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult
from insight_agent.insight import Insight


def test_report_separates_completed_and_skipped_analyses():
    stream = StringIO()
    output = RunOutput(Console(file=stream))
    output.report(
        RunResult(
            10,
            [
                EvidenceStreamResult(
                    stream_name="tool-issues", problems=(), limitations=("Missing schemas",)
                ),
                EvidenceStreamResult(
                    stream_name="ethos-divergence", problems=(), skip_reason="No ethos document"
                ),
            ],
            [],
        ),
        Path("insights.yml"),
        details=True,
    )

    completed, skipped = stream.getvalue().split("Skipped")
    assert "Completed" in completed and "Tool issues" in completed
    assert "Missing schemas" in completed and "Ethos divergence" not in completed
    assert "Ethos divergence" in skipped and "No ethos document" in skipped
    assert "evidence_streams to false" in skipped and "docs/evidence-streams.md" in skipped
    assert "Saved:" not in skipped


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
            EvidenceStreamResult(
                stream_name="tool-issues", problems=(), limitations=("Missing schemas",)
            ),
            EvidenceStreamResult(stream_name="anomaly-and-patterns", problems=()),
            *[
                EvidenceStreamResult(stream_name=name, problems=(), skip_reason="Missing input")
                for name in ("ethos-divergence", "eval-failure-patterns", "user-sentiment")
            ],
        ],
        [make_insight()],
    )
    RunOutput(Console(file=stream, width=width)).report(result, Path("insights.yml"))
    rendered = stream.getvalue()
    assert "Produced 1 insight from 200 traces." in rendered
    assert "3 supporting traces" in rendered
    assert rendered.index("1. Agent") < rendered.index("Checks:") < rendered.index("Saved:")
    normalized = " ".join(rendered.split())
    assert "Checks: 2 completed, 3 skipped; 1 with limited coverage." in normalized
    assert "--details" in rendered
    for hidden in (
        "Missing schemas",
        "Missing input",
        "trace-0",
        "span-0",
        "long explanation",
        "https://",
    ):
        assert hidden not in rendered
    assert "\x1b" not in rendered
    assert all(len(line) <= width for line in rendered.splitlines())
    assert len(rendered.splitlines()) <= (16 if width == 40 else 11)


def test_preview_is_bounded_and_new_insights_come_first_without_changing_result():
    old = [make_insight(f"Existing issue {i}") for i in range(8)]
    new = make_insight("New issue")
    result = RunResult(200, [], [*old, new], existing_insights=old)
    stream = StringIO()
    output = RunOutput(Console(file=stream, width=120))
    output.report(result, Path("insights.yml"))
    rendered = stream.getvalue()
    assert "1. New issue" in rendered
    assert "5. Existing issue 3" in rendered
    assert "Existing issue 4" not in rendered
    assert "4 more insights in YAML output." in rendered
    assert result.insights == [*old, new]

    stream.seek(0)
    stream.truncate()
    output.report(result, Path("-"), details=True)
    rendered = stream.getvalue()
    assert "9. Existing issue 7" in rendered
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
    assert "Checks: 0 completed, 1 skipped." in rendered
    assert "No output file written." in rendered
    assert "Saved:" not in rendered
    assert "No traces loaded. Check your source, filters, and time window." in rendered
