# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read complete native Langfuse CLI response files."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from trace_ingest.loaders.langfuse.config import LangfuseTraceLoadError


class _Meta(BaseModel):
    truncated: bool = Field(strict=True)
    cursor: str | None = None


class _Body(BaseModel):
    data: list[dict[str, JsonValue]]
    meta: _Meta


class _Response(BaseModel):
    status: Literal[200]
    body: _Body


def _read_cli_data(path: Path) -> list[dict[str, JsonValue]]:
    try:
        response = _Response.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise LangfuseTraceLoadError(
            f"Invalid Langfuse CLI export {path}; expected a successful --all --json response: {error}"
        ) from error
    if response.body.meta.truncated or response.body.meta.cursor:
        raise LangfuseTraceLoadError(
            f"Incomplete Langfuse CLI export {path}; increase --max-items or narrow the query and export again"
        )
    return response.body.data
