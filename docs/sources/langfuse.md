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

From a repository checkout, use these Python SDK examples to create the JSONL
format Compass reads. The repository pins Langfuse SDK 4.17.0, which supports both
server versions. Set `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, and
`LANGFUSE_SECRET_KEY` in `.env`, and adjust the dates and output filename below.
Your coding agent can use the [export skill](../../.agents/skills/export-langfuse-traces/SKILL.md)
to do this for you.

### Langfuse v3

Fetch every matching trace, then its full details including observations and scores:

```bash
uv run --extra langfuse --env-file .env python - <<'PYTHON'
from datetime import datetime
from itertools import count
from langfuse import Langfuse

client = Langfuse(tracing_enabled=False)
start = datetime.fromisoformat("2026-09-01T00:00:00+00:00")
end = datetime.fromisoformat("2026-09-02T00:00:00+00:00")
with open("langfuse-v3.jsonl", "x", encoding="utf-8") as output:
    for page in count(1):
        result = client.api.trace.list(
            from_timestamp=start, to_timestamp=end, page=page, limit=100,
        )
        for trace in result.data:
            detail = client.api.trace.get(trace.id)
            output.write(detail.json(exclude_unset=True) + "\n")
        if page >= result.meta.total_pages:
            break
PYTHON
```

### Langfuse v4

Group the observations in the window by trace and fetch each trace's scores.
Scores are not time-filtered, so later evaluations are included.

```bash
uv run --extra langfuse --env-file .env python - <<'PYTHON'
import json
from collections import defaultdict
from datetime import datetime
from langfuse import Langfuse

client = Langfuse(tracing_enabled=False)
start = datetime.fromisoformat("2026-09-01T00:00:00+00:00")
end = datetime.fromisoformat("2026-09-02T00:00:00+00:00")

def records(fetch, **kwargs):
    cursor = None
    while True:
        result = fetch(limit=100, cursor=cursor, **kwargs)
        yield from result.data
        cursor = result.meta.cursor
        if not cursor:
            break

def serialize(item):
    return json.loads(item.json(exclude_unset=True))

with open("langfuse-v4.jsonl", "x", encoding="utf-8") as output:
    traces = defaultdict(list)
    for observation in records(
        client.api.observations.get_many,
        from_start_time=start, to_start_time=end,
        fields="core,basic,io,metadata,model,usage,prompt,trace_context",
    ):
        traces[observation.trace_id].append(serialize(observation))
    for trace_id, observations in traces.items():
        scores = records(
            client.api.scores_v3.get_many_v3, trace_id=trace_id,
            fields="details,subject,annotation",
        )
        output.write(json.dumps({
            "trace_id": trace_id,
            "observations": observations,
            "scores": [serialize(score) for score in scores],
        }) + "\n")
PYTHON
```

You can add SDK filters to the listing calls. V3 filters select traces; v4 filters
select individual observations and do not automatically include other steps of
those traces. The v4 window can also cut across executions. Choose a window and
filters covering the steps you want analyzed; absent steps cannot be recovered
from an export.

Both examples refuse to overwrite an existing file. If an export fails, discard
the incomplete file before retrying. They follow pagination to completion; v4
holds the selected observations in memory while grouping them.

### Analyze the file

Replace your config's `trace` section, choosing the version and file you exported:

```yaml
trace:
  max_traces: 100
  langfuse_export:
    api_version: v4
    path: langfuse-v4.jsonl
    base_url: https://langfuse.example.com
```

For v3, use `api_version: v3`, `path: langfuse-v3.jsonl`, and omit `base_url`.
For v4, `base_url` is optional and enables links back to Langfuse.
Rerun `insight-agent --config config.yaml`. Offline loading does not contact
Langfuse; analysis still needs your inference credentials.

Each v3 line is a full SDK trace-detail record. Each v4 line contains `trace_id`,
`observations`, and `scores` (an empty list is valid). Preserve
`exclude_unset=True` when serializing to distinguish missing values from explicit
nulls. Only these JSONL formats are supported, not CLI response envelopes, UI
exports, directories, or separate score files.
