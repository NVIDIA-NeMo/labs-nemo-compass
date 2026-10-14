<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Preparing a Compass release

A `v<version>` tag runs CI and creates a **draft GitHub release** with the validated
wheel, source archive, and `SHA256SUMS`. This workflow does not publish to Python indexes.

1. Set `[project].version` in `pyproject.toml`, run `uv lock`, and merge the change.
   The distribution name must be `nemo-compass`. Tags must match the version exactly;
   a tag does not change the package version.
2. Tag the intended commit and push it. For example:

   ```bash
   git tag -a v0.1.0rc1 <commit> -m "NeMo Compass 0.1.0rc1"
   git push origin v0.1.0rc1
   ```

3. After CI succeeds, review the draft's generated notes and artifacts. Prerelease
   versions are marked automatically. The release job reuses the validated files.
4. Coordinate the publishing handoff with Phil's team before publishing the GitHub
   release. Share the tag, version, wheel URL, and checksums. Draft asset URLs are
   private; agree how the publishing workflow will retrieve the wheel and what event
   triggers it. See the [shared KitMaker documentation](https://kitmaker.gitlab-master-pages.nvidia.com/kitmaker-docs/users/portal-api/wheel-release-api.html).

## Recovery

Use **Re-run failed jobs** in Actions to reuse successful jobs and their artifacts.
To start a fresh run for a tag, use `gh workflow run ci.yaml --ref v0.1.0rc1`.
Manual branch runs perform CI checks only.

Existing releases are never overwritten. If draft creation partially failed, inspect
the assets, delete the incomplete unpublished draft, and rerun the failed release job.
Preserve published artifacts and shared tags; changes require a new version.
