<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configuration reference

For your first run, choose a [source guide](../README.md#start-here).
Use this page when you want to adjust an existing setup.

## Settings

Save reusable settings in YAML. Keep [credentials](model-access.md#credentials)
in the environment or `.env`.

| Setting | Purpose |
| --- | --- |
| `trace` | Exactly one source; see the [source guides](../README.md#start-here). |
| `trace.max_traces` | Limit complete traces from a provider or native export. Unsupported for canonical JSONL, ATIF, and Gym rollouts. |
| `output_path` | YAML output file; defaults to `insights.yml`. Use `-` for stdout. |
| `model`, `api_base`, `max_tokens` | [Inference settings](model-access.md#choose-a-model). |
| `evidence_streams` | [Evidence streams and their prerequisites](evidence-streams.md). All five are enabled by default. |
| `code_base` | [Local agent source](evidence-streams.md#check-findings-against-code) to consult during validation. |
| `code_validation_concurrency` | Maximum simultaneous candidate code validations across all streams; positive integer, default 4. |
| `existing_insights` | Previous JSON or YAML insight collection to reconcile with this run. |

Relative paths resolve from the directory where you run the command.

## One-run overrides

CLI options override individual YAML values. YAML uses underscores; CLI options use hyphens:

```bash
insight-agent --config config.yaml \
  --trace.max-traces 25 \
  --output-path investigation.yml
```

You can also run without YAML:

```bash
insight-agent --trace.filesystem.path traces.jsonl --max-tokens 16384
```

Use `insight-agent --help-all` for every option and its default.
Explicit CLI and YAML model settings take priority over [environment defaults](model-access.md#credentials).

## Code validation concurrency

When `code_base` is configured, `code_validation_concurrency` limits candidate checks across
all evidence streams in the run. Each candidate can make several model calls while inspecting code.
The default of 4 is a conservative starting point, not a measured optimum for your endpoint.
Choose a positive integer based on your provider limits and observed latency.
If a candidate check fails, the run cancels the remaining checks and waits for their cleanup
before propagating the original failure.

For example, limit a run to two simultaneous candidate validations:

```bash
insight-agent --config config.yaml --code-validation-concurrency 2
```

## Repeat a run

Keep the configuration and a complete trace export to analyze the same input again.
Model-generated findings can vary between runs. Give each run its own `output_path`
when you want to compare results.

To update a previous collection, add these settings to your configuration:

```yaml
existing_insights: previous-insights.yml
output_path: updated-insights.yml
```

Trace Analyst reconciles old and new findings into a complete collection.
See [output behavior](results.md#saved-output) before using the files in a pipeline.
