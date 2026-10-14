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
- V3 uses the CLI and `jq`: list matching trace IDs, fetch each full trace,
  and write its response body as one JSONL line. Keep omitted fields and explicit
  nulls unchanged. Check that trace selection completed before using the export.
- V4 uses installed-CLI commands for observations and optional scores, with
  `--all --json`. Keep their native response envelopes unchanged. Scores are
  project-wide so later evaluations remain available; Compass attaches relevant
  scores and reports skipped unrelated or ambiguous subjects.
- Retain time bounds on v4 observations. Filters select observations, not whole traces;
  explain when the query or window can omit other observations. Do not silently broaden
  the requested selection.
- Use unused output paths. Failed or truncated exports are not usable. CLI
  `--all` defaults to 1,000 items and has a 100-request ceiling; increase its item
  and page limits or narrow the query when needed.
- Validate with the matching `LangfuseV3FileTraceLoader` or
  `LangfuseV4FileTraceLoader` from `trace_ingest.loaders.langfuse`. Report trace/span
  counts, missing-parent diagnostics, and skipped scores. No inference is needed.

Provide a `trace.langfuse_trace_export_file` config with explicit `api_version`.
V3 takes `path`; v4 requires `observations_path`, with optional
`scores_path` for evaluation evidence and `base_url` for platform links. Omit
`scores_path` when no scores are supplied; an empty score response is also valid. Do not substitute UI exports, directories, or custom v4 SDK
bundles for the documented formats.
