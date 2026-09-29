# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a trusted external exporter and ingest its canonical JSONL output."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trace_ingest.loaders.fs import FSDataLoader
from trace_ingest.loaders.trace_loaders import TraceDescription
from trace_ingest.models import TraceSnapshot


class CustomTraceLoadError(ValueError):
    """The configured exporter could not produce valid canonical traces."""


class CustomTraceConfig(BaseModel):
    """A trusted executable, literal arguments, and the published JSONL location."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)
    args: list[str]
    output_path: Path
    timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)

    @field_validator("command")
    @classmethod
    def valid_command(cls, value: str) -> str:
        if not value.strip() or "\0" in value:
            raise ValueError("command must name an executable, not be empty or contain NUL")
        return value

    @field_validator("args")
    @classmethod
    def valid_args(cls, value: list[str]) -> list[str]:
        if any("\0" in arg for arg in value):
            raise ValueError("args must not contain NUL")
        if not any("{output_path}" in arg for arg in value):
            raise ValueError("args must include {output_path} so the utility can write its JSONL")
        return value

    @model_validator(mode="after")
    def valid_output(self) -> CustomTraceConfig:
        if not self.output_path.name:
            raise ValueError("output_path must name a JSONL file")
        return self


@dataclass
class CustomTraceLoader:
    """Execute once per loader instance; only publish successfully validated output."""

    config: CustomTraceConfig
    _loader: FSDataLoader | None = field(default=None, init=False, repr=False)
    _snapshot: TraceSnapshot | None = field(default=None, init=False, repr=False)

    def load(self) -> TraceSnapshot:
        if self._snapshot is not None:
            return self._snapshot
        executable = shutil.which(self.config.command)
        if executable is None:
            raise CustomTraceLoadError(
                f"Custom loader executable {self.config.command!r} was not found or is not executable. "
                "Install it on PATH or configure its executable path."
            )
        output = self.config.output_path.resolve()
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".trace-export-", dir=output.parent) as folder:
                staging = Path(folder) / "traces.jsonl"
                args = [arg.replace("{output_path}", str(staging)) for arg in self.config.args]
                # Logs cannot corrupt the analyst's YAML stdout. No shell expansion is performed.
                result = subprocess.run(
                    [executable, *args],
                    stdin=subprocess.DEVNULL,
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
                if result.returncode:
                    raise CustomTraceLoadError(
                        f"Custom loader exited with status {result.returncode}; "
                        "check its diagnostics and configured arguments."
                    )
                if not staging.is_file() or staging.is_symlink():
                    raise CustomTraceLoadError(
                        "Custom loader did not write a regular JSONL file to {output_path}. "
                        "Check that its output argument uses that placeholder."
                    )
                # Never replace a previous export with malformed, empty, or partial output.
                FSDataLoader(staging).load()
                staging.replace(output)
            loader = FSDataLoader(output)
            snapshot = loader.load()
        except subprocess.TimeoutExpired as error:
            raise CustomTraceLoadError(
                f"Custom loader exceeded timeout_seconds={self.config.timeout_seconds:g}. "
                "Check the utility or increase its timeout."
            ) from error
        except (OSError, ValueError) as error:
            if isinstance(error, CustomTraceLoadError):
                raise
            raise CustomTraceLoadError(
                f"Custom loader output could not be loaded: {error}"
            ) from error
        self._loader, self._snapshot = loader, snapshot
        return snapshot

    def describe(self) -> TraceDescription:
        self.load()
        assert self._loader is not None
        description = self._loader.describe().copy()
        description["source"] = f"custom:{self.config.output_path.resolve()}"
        return description
