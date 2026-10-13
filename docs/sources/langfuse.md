<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze Langfuse traces

Compass supports Langfuse v3 and v4, automatically detecting the version for live
connections. Online loaders include available trace and observation scores; none are required.
Scores support `grader_failure_patterns`, which skips with "No evaluator results"
when none are available. Other analyses can still run.

Langfuse v4 moved from a trace-centric to an observation-centric model: its
[Observations API](https://langfuse.com/docs/api-and-data-platform/features/public-api#observations-api-v2)
returns observations. Compass groups them by `traceId` to reconstruct traces for analysis.

## Setup

Install with the `langfuse` extra, then [configure your inference model and key](../model-access.md#choose-a-model).
Until the PyPI release is available, install with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install 'insight-agent[langfuse] @ git+https://github.com/NVIDIA-NeMo/labs-nemo-compass.git@main'
```

Put your deployment URL and project API keys in `.env`:

```dotenv
LANGFUSE_BASE_URL=https://langfuse.example.com
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
```

## Analyze live traces

Save as `config.yaml`, adjusting the dates and optional filter:

```yaml
output_path: insights.yml
trace:
  max_traces: 100
  langfuse:
    from_timestamp: 2026-09-01T00:00:00Z
    to_timestamp: 2026-09-02T00:00:00Z
    filter: '[{"type":"string","column":"environment","operator":"=","value":"production"}]'
```

```bash
insight-agent --config config.yaml
```

Results are written to `insights.yml` when insights are found. See [reading results](../results.md).
Paths resolve from your working directory. API keys select the project; optional
`trace.langfuse.base_url` overrides the environment URL. The default limit is 100 traces.

Both timestamps are required, with a timezone: start inclusive, end exclusive.
Selection differs by version:

- **V3:** filters select traces; Compass fetches their full details.
- **V4:** filters select observations; Compass fetches all observations for their trace IDs
  **within the same window**. Observations outside it are excluded; costs and latency
  reflect only loaded observations. Widen the window when you need complete traces.

Use your version's [v3 trace](https://api.reference.langfuse.com/#tag/trace/GET/api/public/traces)
or [v4 observation](https://api.reference.langfuse.com/#tag/observationsv2/GET/api/public/v2/observations)
filter syntax. For example, `name` means trace name in v3 and observation name in v4;
use `traceName` for v4 trace names. Compass fetches v4 observations for up to four trace IDs
concurrently and batches score requests; no tuning is needed.

For tool-schema checks, record definitions in `input.tools` on generation observations.
See [Langfuse compatibility](https://langfuse.com/docs/compatibility) for v4 data-availability delays.

## Use an export

With the [Langfuse CLI](https://langfuse.com/docs/api-and-data-platform/features/cli)
installed, use the example for your server version below. The v3 example also needs `jq`.

Coding agents can use the [export skill](../../.agents/skills/export-langfuse-traces/SKILL.md).

### Langfuse v3

```bash
langfuse --env .env --api-version 3 api traces list \
  --from-timestamp 2026-09-01T00:00:00Z \
  --to-timestamp 2026-09-02T00:00:00Z \
  --all --json |
jq -r '.body.data[].id' |
while read -r trace_id; do
  langfuse --env .env --api-version 3 api traces get "$trace_id" --json |
    jq -c '.body'
done > langfuse-v3.jsonl
```

Replace your config's `trace` section with:

```yaml
trace:
  langfuse_trace_export_file:
    api_version: v3
    path: langfuse-v3.jsonl
```

### Langfuse v4

Export observations and optionally project scores if you plan to use the
[`grader_failure_patterns`](../compass-signals.md#grader-failures) signal:

```bash
langfuse --env .env --api-version 4 api observations list \
  --from-start-time 2026-09-01T00:00:00Z \
  --to-start-time 2026-09-02T00:00:00Z \
  --fields core,basic,io,metadata,model,usage,prompt,trace_context \
  --all --json > observations.json

# Optional
langfuse --env .env --api-version 4 api scores list \
  --fields details,subject,annotation --all --json > scores.json
```

Keep exported responses unchanged. Compass rejects truncated exports: `--all` defaults
to 1,000 items and at most 100 requests. Increase `--max-items` / `--limit` or narrow
the query if needed. Observation filters and dates can exclude observations
belonging to the same trace.

Replace your config's `trace` section with:

```yaml
trace:
  langfuse_trace_export_file:
    api_version: v4
    observations_path: observations.json
    scores_path: scores.json  # Optional
    base_url: https://langfuse.example.com
```

`base_url` is optional and enables Langfuse links in Compass' results.

Run `insight-agent --config config.yaml` as above.
