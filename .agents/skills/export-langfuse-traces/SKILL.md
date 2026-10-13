---
name: export-langfuse-traces
description: Export Langfuse v3 or v4 traces in the formats Compass accepts for offline analysis. Use when downloading traces for Compass; not for changing instrumentation or implementing loaders.
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Export Langfuse traces for Compass

Read the authoritative examples in
[the Langfuse export guide](../../../docs/sources/langfuse.md#use-an-export).
Use the user's environment file and adapt the window, filters, and output paths.
The commands work without a repository checkout. Local validation requires
Compass installed with the `langfuse` extra.

- Choose the deployment's server version, not the installed SDK or CLI version.
  Project API keys select the project. Keep credentials out of output and reports.
- V3 uses three installed-CLI list commands with `--api-version 3 --all --json`:
  `traces`, `observations`, and `scores-v2`. Preserve their native envelopes.
  Filter traces to the requested window; observation exports must contain every
  ID referenced by those traces. Export observations unbounded, or use a wider
  window that still includes every referenced step. Scores remain unbounded
  to include later evaluations.
- V4 uses two installed-CLI commands: observations and scores, both with
  `--all --json`. Keep their native response envelopes unchanged. Scores are
  project-wide so later evaluations remain available; Compass attaches relevant
  scores and reports skipped unrelated or ambiguous subjects.
- Retain time bounds on v4 observations. Filters select steps, not whole traces;
  explain when the query or window can omit other steps. Do not silently broaden
  the requested selection.
- Use unused output paths. Failed or truncated exports are not usable. CLI
  `--all` defaults to 1,000 items and has a 100-request ceiling; increase its item
  and page limits or narrow the query when needed.
- Validate with the matching `LangfuseV3FileTraceLoader` or
  `LangfuseV4FileTraceLoader` from `trace_ingest.loaders.langfuse`. Report trace/span
  counts, missing-parent diagnostics, and skipped scores. No inference is needed.

Provide a `trace.langfuse_trace_export_file` config with explicit `api_version`.
Both versions require `observations_path` and `scores_path`; v3 also requires
`traces_path`. V4 optionally accepts `base_url` for platform links. An empty score
response is valid, but its file is still required. Do not substitute UI exports,
directories, trace-detail JSONL, or custom SDK bundles for the native CLI responses.
