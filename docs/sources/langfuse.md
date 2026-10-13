<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze Langfuse traces

Choose a time window containing the conversations you want to understand.
Recorded scores, including scores attached to observations, are included automatically.

The live adapter supports Langfuse v3 and v4 (including Langfuse Cloud).
It reads the server version from `GET /api/public/health` and automatically uses
the v3 trace API for server releases 3.x or the observations-based path for 4.x.

## Connect and run

Install Compass with the `langfuse` extra. Once the PyPI release is available:

```bash
python -m pip install 'insight-agent[langfuse]'
```

Until then, install directly from GitHub with [uv and Git](../../README.md#start-here).
This does not require a local checkout:

```bash
uv tool install \
  'insight-agent[langfuse] @ git+https://github.com/NVIDIA-NeMo/labs-nemo-compass.git@main'
```

[Configure your inference model and key](../model-access.md#choose-a-model).
Add your project’s Langfuse keys to `.env`:

```dotenv
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
```

Save this as `config.yaml`. Replace the URL and dates with your deployment and trace window:

```yaml
max_tokens: 16384
trace:
  max_traces: 100
  langfuse:
    base_url: https://langfuse.example.com
    from_timestamp: 2026-09-01T00:00:00Z
    to_timestamp: 2026-09-02T00:00:00Z
```

```bash
insight-agent --config config.yaml
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped Compass Signals.

## Select the right traces

Your API keys select the project. Both timestamps are required and must include a timezone;
the start is inclusive and the end is exclusive. The default limit is 100 distinct traces.
`base_url` overrides `LANGFUSE_BASE_URL` in the environment.

To filter by environment, add this under `trace.langfuse`:

```yaml
filter: '[{"type":"string","column":"environment","operator":"=","value":"production"}]'
```

### How selection differs between v3 and v4

- **v3:** The window and filters select trace records, newest trace timestamp first.
- **v4:** The window and filters select observations (individual steps), newest observation
  start time first. Compass takes distinct trace IDs, then loads all observations for
  each selected trace within that same time window, without reapplying the selection
  filter. Both selection and detail reads include time bounds to avoid expensive
  all-history queries on Langfuse v4.

V4 traces can be partial at the window boundaries: steps starting before the start
or at/after the end are excluded, even if they belong to a selected trace. Choose a
window covering the entire execution when you need all its steps. Each loaded v4
trace records this scope in `attributes.observation_window`; reported costs and
latency describe the loaded steps. Missing parents are reported, but a missing child
or sibling cannot always be detected. Narrowing the window reduces query work but
also reduces the evidence available to Compass.

Write `filter` for the Langfuse version you are connecting to. You are responsible
for using that version's query syntax and filter semantics:

- **Langfuse v3:** [Traces API filters](https://api.reference.langfuse.com/#tag/trace/GET/api/public/traces).
- **Langfuse v4:** [Observations API v2 filters](https://api.reference.langfuse.com/#tag/observationsv2/GET/api/public/v2/observations).

Compass checks that the filter is a JSON array of objects, but leaves column,
operator, and value validation to Langfuse. It does not translate filters between
versions. For example, `name` selects trace names on v3 and observation names on
v4; use `traceName` for trace names on v4.

V4 no longer has separate trace-level inputs and outputs. Compass uses the sole physical
root observation for those values and preserves each step's inputs, outputs, and context.
Costs and latency are computed from the loaded observations. Trace and observation scores
are fetched separately, including scores recorded after the selection window, and retain
their typed values and subject associations. Missing parents are reported in the source
summary; Compass cannot recover observations that were never recorded or were deleted.

V4 observation metadata may be truncated by Langfuse. Older instrumentation can also take
up to 15 minutes to appear in v4 queries; see [Langfuse compatibility](https://langfuse.com/docs/compatibility).

For tool-schema checks, record tool definitions in `input.tools` on generation observations.
Missing or conflicting catalogs limit those checks.

## Use an export

These examples work from any directory with the [Langfuse CLI](https://langfuse.com/docs/api-and-data-platform/features/cli)
installed (`npm install --global @langfuse/cli@1.2.4`). The v3 example also uses `jq`.

Create a `.env` file in that directory:

```dotenv
LANGFUSE_BASE_URL=https://langfuse.example.com
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
```

Choose the example for your server version and adjust its dates and output filename.
If you have a checkout, your coding agent can also use the optional
[export skill](../../.agents/skills/export-langfuse-traces/SKILL.md).

### Langfuse v3

List the matching trace IDs, then fetch their full details including observations
and scores. The `jq` check stops the export if the trace list hits the CLI's limit.

```bash
set -euo pipefail

langfuse --env .env --api-version 3 api traces list \
  --from-timestamp 2026-09-01T00:00:00Z \
  --to-timestamp 2026-09-02T00:00:00Z \
  --all --json |
jq -r 'if .body.meta.truncated then error("Trace list truncated; increase --max-items") else .body.data[].id end' |
while IFS= read -r trace_id; do
  langfuse --env .env --api-version 3 api traces get "$trace_id" --json |
    jq -c '.body'
done > langfuse-v3.jsonl
```

Choose an unused output filename; shell redirection overwrites existing files.
If the export fails, discard the incomplete file before retrying. Each JSONL line
contains one full trace-detail record extracted from the CLI response.

### Langfuse v4

Export observations in your chosen window, then the project's scores. Scores are
not time-filtered, so later evaluations are included. Run these in a new directory
or choose unused output filenames; shell redirection overwrites existing files.

```bash
langfuse --env .env --api-version 4 api observations list \
  --from-start-time 2026-09-01T00:00:00Z \
  --to-start-time 2026-09-02T00:00:00Z \
  --fields core,basic,io,metadata,model,usage,prompt,trace_context \
  --all --json > observations.json

langfuse --env .env --api-version 4 api scores list \
  --fields details,subject,annotation --all --json > scores.json
```

Keep both CLI responses unchanged. Compass checks completion and rejects failed,
incomplete, or truncated responses. `--all` defaults to 1,000 items and stops
after at most 100 requests; increase `--max-items` and `--limit` or narrow the
query if an export is truncated. Compass attaches scores to the exported traces
and reports how many unrelated or ambiguous scores it skips.

V4 filters select individual observations, not complete traces. Filtering by step
name or type can omit other steps, and the time window can cut across executions.
Choose a window and filters covering the evidence you want analyzed.

### Analyze the export

For v3, replace your config's `trace` section with:

```yaml
trace:
  max_traces: 100
  langfuse_trace_export_file:
    api_version: v3
    path: langfuse-v3.jsonl
```

For v4, provide both CLI output files:

```yaml
trace:
  max_traces: 100
  langfuse_trace_export_file:
    api_version: v4
    observations_path: observations.json
    scores_path: scores.json
    base_url: https://langfuse.example.com
```

V4's `base_url` is optional and enables links back to Langfuse.
Run `insight-agent --config config.yaml` from the directory containing your config
and exports. Compass must be installed with the `langfuse` extra, including for
offline loading. Analysis still needs your [inference credentials](../model-access.md#choose-a-model).

Only the documented v3 trace-detail JSONL and v4 CLI JSON responses are supported, not UI
exports or directories. A v4 score response with no scores is valid, but the file
is still required.
