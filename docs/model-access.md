<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data and model access

NeMo Compass needs an inference model to investigate traces and write insights.
Its API key is separate from your Braintrust, LangSmith, Langfuse, MLflow, or Intake credentials.

## Choose a model

NeMo Compass uses the LiteLLM library to connect to supported inference providers,
including OpenAI and Anthropic. Choose a model that supports tool calling
and structured output, and save its settings in `.env`.

For OpenAI:

```dotenv
INSIGHT_AGENT_MODEL=openai/gpt-5.2
INSIGHT_AGENT_API_KEY=your-openai-api-key
```

For Anthropic:

```dotenv
INSIGHT_AGENT_MODEL=anthropic/claude-sonnet-4-6
INSIGHT_AGENT_API_KEY=your-anthropic-api-key
```

These are full LiteLLM model strings: the `openai/` or `anthropic/` prefix selects
the provider. Direct access uses the provider's default endpoint; remove any old
`INSIGHT_AGENT_API_BASE` or YAML/CLI `api_base` override when switching providers.
For OpenAI directly, also unset inherited `OPENAI_API_BASE` and `OPENAI_BASE_URL`
gateway overrides.

You can also use an OpenAI-compatible gateway by setting its model name and API base:

```dotenv
INSIGHT_AGENT_MODEL=openai/your-model-name
INSIGHT_AGENT_API_KEY=your-gateway-key
INSIGHT_AGENT_API_BASE=https://gateway.example.com/v1
```

Replace the example values with your provider’s settings. Use the documented API base,
without appending a resource path such as `/chat/completions`.
The first `openai/` selects the protocol; the rest is the gateway’s model name.
If no model is set, the package defaults to `openai/azure/openai/gpt-5.6-luna`,
which requires the corresponding gateway configuration.

The source guides set `max_tokens: 16384` to limit each model response.
Adjust that YAML setting to a value your endpoint supports.
You can also set `model` and `api_base` in YAML or override them on the CLI.

## Recommended models

We recommend the following models for NeMo Compass:

- **NVIDIA Nemotron 3.5 Super VL**
- **Claude Sonnet 5.5**
- **GPT-6 Luna**

## Credentials

Keep `.env` in the folder where you run the command to load it automatically:

```bash
insight-agent --config config.yaml
```

Or select an environment file explicitly with uv:

```bash
uv tool run --env-file /path/to/.env insight-agent --config config.yaml
```

Exported variables take priority over values in either file.
Keep credentials out of configuration YAML and source control.

`OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are accepted as fallback inference keys,
in that order. Set `INSIGHT_AGENT_API_KEY` explicitly when credentials for multiple
providers are present. For OpenRouter, put its key in `INSIGHT_AGENT_API_KEY`;
`OPENROUTER_API_KEY` alone does not satisfy the CLI's credential check.

`OPENAI_API_BASE` and `OPENAI_BASE_URL` are fallback endpoint settings, except for
models using the `anthropic/` or `openrouter/` prefixes. Explicit
`INSIGHT_AGENT_API_BASE` and YAML/CLI `api_base` overrides still apply to those models.
Choose the matching model explicitly. The CLI requires an inference key before loading traces.

Each source guide names the credentials needed for live access.
Loading a saved export does not require credentials for that trace platform.

Return to your [source guide](../README.md#start-here) to finish setup.

## Where data goes

| When you use… | Data access |
| --- | --- |
| A live trace source | NeMo Compass reads traces and recorded evaluation signals from that service. |
| Model-based analysis | Trace content, evaluation signals, and candidate findings may be sent to your configured inference provider. |
| An ethos document | Its contents are included in model-based analysis. |
| Code validation | Selected source excerpts are sent to the inference provider. The validator has read-only file tools. |
| Remote sentiment embeddings | Extracted user messages are sent to the configured embedding endpoint. |
| Local sentiment embeddings | The embedding model runs on your machine; model and tokenizer files may download on first use. |

Remote sentiment also loads a tokenizer, which may need an initial download.
Using local trace files still requires inference access. There is no automatic redaction
of trace content before analysis; choose inputs and endpoints appropriate for your data.

The CLI writes insights to a local YAML file or stdout. With telemetry disabled it
does not publish findings back to the trace platform. [Output details](results.md#saved-output).

### Finding the inputs of a Compass run

When Compass telemetry is enabled, open the root `compass` span in LangSmith and
look for the `compass.input.*` fields in its metadata. Structured values such as
trace references and file fingerprints are JSON strings because LangSmith’s OTLP
mapper preserves scalar metadata values. These describe the input corpus without
copying input traces into telemetry spans:

- `source_type` and `selection` identify the configured loader, query, filters,
  time window, and limit where applicable.
- `description` contains the loader's resolved source details and corpus counts,
  including environment-resolved endpoints and project IDs where available.
- `trace_count` and `traces` record the loaded trace count and each input trace's
  ID and source URL. HTTP(S) links open the original platform; file links
  identify local files and, where supported, line numbers. A null URL means the
  source supplies no link; use the source coordinates and trace ID instead.
- `files` records absolute paths and SHA-256 hashes for local inputs, including
  files in export directories and auxiliary Langfuse score files. Hashes are read
  after ingestion; an unavailable file is marked explicitly.
- `normalized_sha256` fingerprints the exact normalized corpus consumed by Compass:
  each trace's UTF-8 `model_dump_json()` followed by a newline, in snapshot order.
  This is distinct from the hashes of the original file bytes.

This works for all live platform loaders and file loaders (canonical JSONL, Gym,
ATIF, and native platform exports). A hash identifies content but cannot retrieve
it. `snapshot_saved: false` makes explicit that Compass does not archive or upload
an input snapshot. Preserve the original files yourself; local file URLs are not
accessible from other machines, and live records may change or expire.

The `ingestion` span still measures loading, and subsequent spans describe
Compass's analysis. Model and tool spans can contain input excerpts used during
that analysis; this change does not redact their execution content.

Telemetry requests use gzip compression to reduce transfer size. The OTLP exporter’s
64 MiB limit still applies to the uncompressed request. Requests have a 60-second
timeout by default; `OTEL_EXPORTER_OTLP_TRACES_TIMEOUT` (or the shared
`OTEL_EXPORTER_OTLP_TIMEOUT`) can override it in seconds.

Validation, evidence scanning, and confidence rating have explicit spans. Evidence
scan outputs distinguish complete, incomplete, skipped, and failed scans; a recovered
scan failure does not fail the overall run. Tool errors include exception events so
LangSmith can show failed attempts even when the agent recovers.
