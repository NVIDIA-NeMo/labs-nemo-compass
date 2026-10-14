# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise release validation through its CLI using actual distribution archives."""

import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

CHECK_RELEASE = Path(__file__).resolve().parents[1] / "tools/check_release.py"


@pytest.mark.parametrize(
    ("version", "fault", "expected"),
    [
        ("0.1.0", "", "prerelease=false"),
        ("0.1.0rc1", "", "prerelease=true"),
        ("0.1.0.dev1", "", "prerelease=true"),
        ("0.1.0.post1", "", "prerelease=false"),
        ("0.1.0", "name", "Rename the distribution"),
        ("0.1.0", "tag", "must match pyproject.toml version"),
        ("0.1.0", "extra-wheel", "dist/ must contain only"),
        ("0.1.0", "missing-sdist", "dist/ must contain only"),
        ("0.1.0", "unbundled", "must bundle trace_ingest"),
        ("0.1.0", "wheel-metadata", "metadata must match"),
        ("0.1.0", "sdist-metadata", "metadata must match"),
        ("0.1.0", "dependency", "must not depend on a separate trace-ingest"),
    ],
)
def test_release_contract(tmp_path, version, fault, expected):
    name = "insight-agent" if fault == "name" else "nemo-compass"
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "{version}"\n')
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / ".gitignore").write_text("*")
    stem = f"nemo_compass-{version}"
    metadata = f"Name: nemo-compass\nVersion: {version}\n"
    if fault == "dependency":
        metadata += 'Requires-Dist: Trace_Ingest[langsmith]>=0.1; extra == "langsmith"\n'
    with zipfile.ZipFile(dist / f"{stem}-py3-none-any.whl", "w") as wheel:
        wheel.writestr(
            f"{stem}.dist-info/METADATA",
            metadata.replace(version, "9.9.9") if fault == "wheel-metadata" else metadata,
        )
        wheel.writestr("insight_agent/__init__.py", "")
        if fault != "unbundled":
            wheel.writestr("trace_ingest/__init__.py", "")
    if fault != "missing-sdist":
        contents = metadata.replace(version, "9.9.9") if fault == "sdist-metadata" else metadata
        with tarfile.open(dist / f"{stem}.tar.gz", "w:gz") as sdist:
            member = tarfile.TarInfo(f"{stem}/PKG-INFO")
            member.size = len(contents.encode())
            sdist.addfile(member, io.BytesIO(contents.encode()))
    if fault == "extra-wheel":
        (dist / "trace_ingest-0.1.0-py3-none-any.whl").touch()

    result = subprocess.run(
        [sys.executable, str(CHECK_RELEASE), "v9.9.9" if fault == "tag" else f"v{version}"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == (2 if fault else 0), result.stderr
    assert expected in (result.stderr if fault else result.stdout)
