<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze Langfuse traces

Choose a time window containing the conversations you want to understand.
Recorded scores, including scores attached to observations, are included automatically.

The live adapter supports Langfuse v3 and v4 (including Langfuse Cloud).
It automatically probes the observations endpoint and uses the v3 trace API only
when that endpoint returns 404 or 405. Authentication, rate-limit, and server errors
are reported rather than triggering a fallback. To choose explicitly, set
`trace.langfuse.api_version` to `v3` or `v4` (default: `auto`). The selected API is
recorded in the run's source description.

The optional dependency is Python SDK `>=4.17,<5`, locked to 4.17.0. The SDK's
legacy trace methods keep v3 deployments and existing v3 exports usable.

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

In v4, `filter` uses the [Observations API v2 columns and operators](https://langfuse.com/docs/api-and-data-platform/features/public-api#query-parameters-and-filters).
For example, `name` means an observation's name; use `traceName` for the trace name,
and `startTime` instead of the v3 trace `timestamp` column. Environment filters above
work with both paths. Existing trace filters are not silently translated when a
server upgrades; set `api_version` explicitly when a query depends on one schema.

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

Replace the `trace` section with a file containing complete v3 trace details:

```yaml
trace:
  max_traces: 100
  langfuse_export:
    path: langfuse-traces.jsonl
```

Rerun the same analysis command. Local loading needs inference credentials only.
Use complete trace-detail records with observations; trace-list summaries and UI CSV files
do not contain enough information. Raw API responses, CLI response envelopes, and directories
of immediate `.json` or `.jsonl` files are accepted. The loader validates all records, rejects
duplicate IDs, and selects the newest traces up to the limit. V4 observation-page exports
are not accepted by this file loader; use the live v4 adapter.

<details>
<summary>Create a complete export</summary>

Set your Langfuse credentials, including `LANGFUSE_BASE_URL`, in `.env` and adjust the dates.
This writes a new file and refuses to overwrite an existing one.

```bash
uv run --isolated --no-project --env-file .env --with 'langfuse>=4.17,<5' python - <<'PY'
from datetime import datetime, timezone
from itertools import count
from langfuse import Langfuse

client = Langfuse(tracing_enabled=False)
with open("langfuse-traces.jsonl", "x", encoding="utf-8") as output:
    for page in count(1):
        result = client.api.trace.list(
            from_timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
            to_timestamp=datetime(2026, 9, 2, tzinfo=timezone.utc),
            page=page, limit=100, order_by="timestamp.asc",
        )
        for trace in result.data:
            output.write(client.api.trace.get(trace.id).json(by_alias=True) + "\n")
        if page >= result.meta.total_pages:
            break
PY
```

</details>
