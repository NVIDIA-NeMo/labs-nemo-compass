# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the JSONL export format documented for Langfuse."""

import json
from collections.abc import Iterator
from pathlib import Path

from trace_ingest.loaders.langfuse.config import LangfuseTraceLoadError


def _records(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    count = 0
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("expected one trace object per line")
                except ValueError as error:
                    raise LangfuseTraceLoadError(
                        f"Invalid Langfuse export {path}:{line_number}: {error}"
                    ) from error
                count += 1
                yield line_number, record
    except (OSError, UnicodeError) as error:
        raise LangfuseTraceLoadError(f"Cannot read Langfuse export {path}: {error}") from error
    if not count:
        raise LangfuseTraceLoadError(f"Langfuse export {path} contains no traces")
