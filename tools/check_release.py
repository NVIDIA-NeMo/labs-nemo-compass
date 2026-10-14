#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check release identity and print the GitHub prerelease flag."""

from __future__ import annotations

import argparse
from pathlib import Path

import tomllib
from packaging.version import Version


def check_release(tag: str, root: Path) -> bool:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if project["name"] != "nemo-compass":
        raise ValueError("Rename the distribution to nemo-compass before preparing a release.")
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} must match pyproject.toml version: v{version}")

    return Version(version).is_prerelease


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
