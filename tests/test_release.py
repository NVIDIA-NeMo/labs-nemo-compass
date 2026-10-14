# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the release CLI contract; distribution contents are validated by CI."""

import subprocess
import sys
from pathlib import Path

import pytest

CHECK_RELEASE = Path(__file__).resolve().parents[1] / "tools/check_release.py"


@pytest.mark.parametrize(
    ("name", "tag", "code", "expected"),
    [
        ("nemo-compass", "v0.1.0rc1", 0, "prerelease=true"),
        ("nemo-compass", "v0.1.0", 2, "must match pyproject.toml version"),
        ("insight-agent", "v0.1.0rc1", 2, "Rename the distribution"),
    ],
)
def test_release_contract(tmp_path, name, tag, code, expected):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0.1.0rc1"\n')
    result = subprocess.run(
        [sys.executable, str(CHECK_RELEASE), tag],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == code, result.stderr
    assert expected in (result.stderr if code else result.stdout)
