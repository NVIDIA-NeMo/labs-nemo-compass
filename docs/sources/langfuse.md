<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze Langfuse traces

Choose a time window containing the conversations you want to understand.
Recorded scores, including scores attached to observations, are included automatically.

The live adapter supports Langfuse v3 and v4 (including Langfuse Cloud).
It reads the server version from `GET /api/public/health` and automatically uses
the v3 trace API for server releases 3.x or the observations-based path for 4.x.

## Connect and run

With [uv and Git installed](../../README.md#start-here), install the CLI with Langfuse support:

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

Compass supports command-line exports from Langfuse v3 and v4. The examples below
use the [Langfuse CLI](https://langfuse.com/docs/api-and-data-platform/features/cli)
with Node.js, `uv`, and `jq` installed. Add `LANGFUSE_BASE_URL` to `.env` alongside
your project keys, and adjust the dates. Each example creates a new export directory and checks that the query returned data.
Your coding agent can run these same commands for you.

### Langfuse v3

Export the matching trace IDs, then fetch each complete trace with its observations
and scores. The explicit API version selects the CLI's v3 commands for self-hosted
v3 deployments.

```bash
set -euo pipefail
export_dir=$(mktemp -d ./langfuse-v3.XXXXXX)

uv run --no-project --env-file .env -- npx --yes @langfuse/cli@1.2.4 \
  --api-version 3 api traces list \
  --from-timestamp 2026-09-01T00:00:00Z \
  --to-timestamp 2026-09-02T00:00:00Z \
  --limit 100 --all --max-items 10000 --json > "$export_dir/selection.json"
jq -e '.status == 200 and .body.meta.truncated == false and (.body.data | length) > 0' \
  "$export_dir/selection.json" >/dev/null

jq -r '.body.data[].id' "$export_dir/selection.json" |
while IFS= read -r trace_id; do
  uv run --no-project --env-file .env -- npx --yes @langfuse/cli@1.2.4 \
    --api-version 3 api traces get "$trace_id" --json
done > "$export_dir/traces.jsonl"
echo "$export_dir/traces.jsonl"
```

Replace your config's `trace` section with the printed path:

```yaml
trace:
  max_traces: 100
  langfuse_export:
    path: ./langfuse-v3.EXAMPLE/traces.jsonl
```

V3 is the default export format. Complete raw API trace-detail records and CLI
response envelopes are accepted; trace-list summaries are not. Point at
`traces.jsonl`, not the directory containing `selection.json`.

### Langfuse v4

Export observations with the fields needed for analysis. The CLI combines all pages
into one response envelope; keep that envelope so Compass can detect truncation.

```bash
set -euo pipefail
export_dir=$(mktemp -d ./langfuse-v4.XXXXXX)

uv run --no-project --env-file .env -- npx --yes @langfuse/cli@1.2.4 \
  --api-version 4 api observations list \
  --from-start-time 2026-09-01T00:00:00Z \
  --to-start-time 2026-09-02T00:00:00Z \
  --fields core,basic,io,metadata,model,usage,prompt,trace_context \
  --limit 1000 --all --max-items 100000 --json > "$export_dir/observations.json"
jq -e '.status == 200 and .body.meta.truncated == false and (.body.data | length) > 0' \
  "$export_dir/observations.json" >/dev/null

# Optional: include scores for the exported traces, including later evaluations.
jq -r '.body.data[].traceId' "$export_dir/observations.json" | sort -u |
while IFS= read -r trace_id; do
  uv run --no-project --env-file .env -- npx --yes @langfuse/cli@1.2.4 \
    --api-version 4 api scores list --trace-id "$trace_id" \
    --fields details,subject,annotation \
    --limit 100 --all --max-items 10000 --json
done > "$export_dir/scores.jsonl"
jq -e -s 'all(.[]; .status == 200 and .body.meta.truncated == false)' \
  "$export_dir/scores.jsonl" >/dev/null
echo "$export_dir"
```

Use the printed directory in your config:

```yaml
trace:
  max_traces: 100
  langfuse_export:
    api_version: v4
    path: ./langfuse-v4.EXAMPLE/observations.json
    scores_path: ./langfuse-v4.EXAMPLE/scores.jsonl
    base_url: https://langfuse.example.com
```

`scores_path` is optional. `base_url` is optional and enables links back to Langfuse.
V4 accepts successful CLI `--json` response envelopes: one per JSON file or one per
line in JSONL. Use `--all`; unfinished cursor pages and truncated exports are rejected.

The export contains only the observations matching your query. Unlike the live
loader, it does not expand selected steps into their other trace steps. Filters on
observation name, type, or root status can exclude useful evidence, and time windows
can cut across executions. Choose the window and filters accordingly.

### Export limits

The CLI's `--all` defaults to 1,000 items and stops after at most 100 requests.
The examples set explicit item limits and check for nonempty, untruncated results.
If no records match, choose a different window. If the export is truncated, use
smaller time windows in separate files. Preserve whole CLI responses
rather than extracting their `data` arrays for v4.

Both loaders also accept directories of immediate `.json` and `.jsonl` files in
their supported format. Duplicate records are rejected, so avoid overlapping
exports. UI downloads, CSV, and blob-storage exports are outside this workflow.
Rerun `insight-agent --config config.yaml` to analyze the local files; loading them
requires no Langfuse connection, only your inference credentials for analysis.
