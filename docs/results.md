<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Read your results

Start with the saved insights. Each highlights your agent’s behavior and links it to
supporting traces you can inspect in your source platform or export.

An illustrative entry in `insights.yml`:

```yaml
- name: Retries repeat an invalid order lookup
  description: >-
    After lookup_order reports an unknown order ID, the agent repeats
    the same request without asking the customer to correct the ID.
  evidence:
    - trace_id: run-12
      url: https://observability.example.com/traces/run-12
      spans:
        - span_id: lookup-3
          url: https://observability.example.com/spans/lookup-3
    - trace_id: run-38
      url: https://observability.example.com/traces/run-38
```

Open a trace link to review the supporting behavior before deciding what to change.
Each `evidence` entry identifies a supporting trace and optionally its relevant spans.
URLs are optional and come from the loader, not the inference model.

Live LangSmith and Langfuse traces use their provider-returned UI locations; live MLflow
traces use the HTTP tracking server’s UI. Braintrust links require your
[organization name](sources/braintrust.md#links-to-braintrust). Intake traces currently
retain trace IDs without viewer links.

Local inputs link to the original file on the machine that ran the analysis. Canonical
JSONL, ATIF, Gym JSONL, and LangSmith exports include a `#L<number>` fragment for the
physical source line. Opening the file at that line depends on your viewer’s support.
Other native exports link to the containing file. Canonical file links always point
to the input file; user-supplied URLs are not retained.

When a provider does not supply a usable URL, or its location cannot be determined
(for example, an MLflow `databricks` or local tracking URI), the ID remains available.
Previously saved links survive reconciliation for traces absent from the current run;
links for traces loaded in this run are resolved from the current source.

Each trace entry can list relevant `spans` at any nesting depth, with a `span_id` and
optional `url`. Braintrust and live LangSmith supply span links; other sources retain
span IDs. An entry without spans refers to the whole trace.

With `--details`, trace and span IDs are clickable labels; plain-text logs show the
URLs. Spans appear beneath their trace, with the trace link as a fallback when a span
has no link. Displayed IDs are shortened but stay distinguishable within their group;
saved YAML retains complete IDs and links.

## Terminal report

The default report previews up to five insights, with new insights first. The YAML
output contains every insight and its complete evidence.

```text
1 insight from 200 traces

Agent calls tools absent from the active catalog
  Evidence: 3 supporting traces

Coverage
  Ran:      Patterns (5 candidates), tool issues (3 candidates)
  Skipped:  Ethos - no document
            Evaluation - no results
            Sentiment - no embedding backend
  To run skipped checks: https://github.com/NVIDIA-NeMo/labs-trace-intel/blob/main/docs/evidence-streams.md

Saved: out/try-next-step/insights.yml
```

Coverage lists the checks that ran, their candidate counts and limitations, and why
other checks were skipped. Disabled checks are omitted. Supporting trace counts do
not represent a failure rate.

Open the saved YAML to read descriptions and inspect evidence without running
analysis again. For a detailed terminal report during a run, add `--details`:

```bash
insight-agent --config config.yaml --details
```

This displays every insight description and its trace/span links. Set `details: true`
in YAML to make this the default; `--no-details` overrides it.

A **candidate issue** is a problem found during analysis. Further review may merge or
discard it, so candidate counts can exceed the number of saved insights.

**No findings** means an evidence stream ran and found nothing to report in the available data.
**Skipped** means it could not run. A limitation beside a completed evidence stream describes
missing coverage; it does not mean the whole evidence stream was skipped.

## An evidence stream was skipped

Open [Evidence streams](evidence-streams.md) for the prerequisite and a setup example.
[Disable evidence streams](evidence-streams.md#disable-an-evidence-stream) you don’t need in your configuration.
Disabled evidence streams are omitted from the report.

If no traces were loaded, check your source, filters, and time window first.

## No insights were produced

The run may have found no actionable patterns, or its candidates may not have passed
review. This does not establish that your agent is free of problems. Read the coverage
section for skipped checks and limitations.

## Saved output

The output is a YAML list with `name`, `description`, `evidence`, and optional
`updated_date` for each insight. Each insight has at least two supporting traces.
Empty spans and absent URLs are omitted.

`updated_date` is a UTC timestamp set when an insight is created or gains a trace it
didn't already have. It is omitted until that first happens, and an insight that gains
no new trace in a run keeps its prior `updated_date` unchanged. See
[repeat a run](configuration.md#repeat-a-run) for reconciling across runs.

By default, a non-empty collection is written to `insights.yml`.
**An empty result leaves any existing output file untouched.** Use a fresh output path
per run, or capture stdout to receive the current result, including `[]`:

```bash
insight-agent --config config.yaml --output-path - > run-insights.yml
```

Progress and the report go to stderr. Piped stdout contains YAML.
The default output file is still written unless `--output-path -` is selected.
A successful run, including one with skipped evidence streams or no insights, exits with code 0.
Missing required environment settings exit with code 2; other errors exit nonzero.

To carry findings forward, see [repeat a run](configuration.md#repeat-a-run).
