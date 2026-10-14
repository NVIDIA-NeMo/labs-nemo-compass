#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the single Compass release and print its GitHub prerelease flag."""

from __future__ import annotations

import argparse
import email
import re
import tarfile
import zipfile
from pathlib import Path

import tomllib


def check_release(tag: str, root: Path) -> bool:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if project["name"] != "nemo-compass":
        raise ValueError("Rename the distribution to nemo-compass before preparing a release.")
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} must match pyproject.toml version: v{version}")

    stem = f"nemo_compass-{version}"
    wheel_name = f"{stem}-py3-none-any.whl"
    sdist_name = f"{stem}.tar.gz"
    dist = root / "dist"
    # uv creates a .gitignore alongside the distributions.
    if {path.name for path in dist.iterdir()} - {".gitignore"} != {wheel_name, sdist_name}:
        raise ValueError(f"dist/ must contain only {wheel_name} and {sdist_name}.")

    with zipfile.ZipFile(dist / wheel_name) as wheel:
        metadata = [wheel.read(f"{stem}.dist-info/METADATA")]
        for package in ("insight_agent", "trace_ingest"):
            if f"{package}/__init__.py" not in wheel.namelist():
                raise ValueError(f"The Compass wheel must bundle {package}.")
    with tarfile.open(dist / sdist_name) as sdist:
        source = sdist.extractfile(f"{stem}/PKG-INFO")
        if source is None:
            raise ValueError("Source distribution is missing package metadata.")
        metadata.append(source.read())

    for contents in metadata:
        message = email.message_from_bytes(contents)
        if message["Name"] != "nemo-compass" or message["Version"] != version:
            raise ValueError("Wheel and source metadata must match the project name and version.")
        for requirement in message.get_all("Requires-Dist", []):
            name = re.split(r"[\s\[<>=!~;(@]", requirement, maxsplit=1)[0]
            if re.sub(r"[-_.]+", "-", name).lower() == "trace-ingest":
                raise ValueError("Compass must not depend on a separate trace-ingest distribution.")

    # Built metadata uses normalized PEP 440 versions; post releases remain stable.
    return re.search(r"(?:a|b|rc|dev)\d+", version) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="release tag, exactly v<project.version>")
    args = parser.parse_args()
    try:
        prerelease = check_release(args.tag, Path.cwd())
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"prerelease={str(prerelease).lower()}")


if __name__ == "__main__":
    main()
