---
name: export-langfuse-traces
description: Export Langfuse v3 or v4 traces to the Python SDK JSONL format Compass accepts for offline analysis. Use when downloading traces for Compass; not for changing instrumentation or implementing loaders.
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Export Langfuse traces for Compass

Read the authoritative Python examples in
[the Langfuse export guide](../../../docs/sources/langfuse.md#use-an-export).
The commands supply their own pinned SDK and work from any directory. Use the
user's environment file and adapt the date window, filters, and output path.
For local validation, use a Compass installation with the `langfuse` extra.

- Use the deployment's server version, not the installed SDK version. If unknown,
  read `client.api.health.health().version`; the project API keys select the project.
- V3 exports one full trace-detail record per JSONL line. V4 exports one record
  containing `trace_id`, `observations`, and `scores` per line. An empty scores
  list is valid. Keep `exclude_unset=True` when serializing SDK records.
- Follow every page, retaining time bounds on v4 observation reads. V4 filters
  select steps, not whole traces: explain when the requested query or window can
  omit other steps. Do not silently broaden the user's selection.
- Use exclusive file creation. A failed export is incomplete; do not present it
  as usable or append a retry to it. Keep credentials out of output and reports.
- Validate the completed file with `LangfuseV3FileTraceLoader` or
  `LangfuseV4FileTraceLoader` from `trace_ingest.loaders.langfuse`, matching the
  server version. Report trace/span counts and any missing-parent diagnostics.
  This check does not require inference.

Provide the exported file path and a `trace.langfuse_export` config with explicit
`api_version` and `path`. For v4, include the deployment `base_url` when platform
links are desired. Do not substitute CLI envelopes, UI downloads, directories,
or separate score files for the documented format.
