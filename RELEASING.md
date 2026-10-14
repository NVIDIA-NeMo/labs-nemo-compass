<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Preparing a Compass release

The **Prepare release** GitHub Action runs CI, tags the validated commit, and creates
a **draft GitHub release** with its wheel, source archive, and `SHA256SUMS`.
This workflow does not publish to Python indexes.

1. Set `[project].version` in `pyproject.toml`, run `uv lock`, and merge the change.
   The distribution name must be `nemo-compass`.
2. Open **Actions → Prepare release → Run workflow**. Choose the branch to release,
   or enter a specific commit SHA, then click **Run workflow**. The action must be
   merged into the default branch before GitHub shows this button.
3. After CI succeeds, the action creates the version tag (for example, `v0.1.0`)
   on the exact commit it checked and opens a draft release. Review its generated
   notes and artifacts. Prerelease versions are marked automatically. The release
   job reuses the validated files without rebuilding them.
4. Coordinate the publishing handoff with Phil's team before publishing the GitHub
   release. Share the tag, version, wheel URL, and checksums. Draft asset URLs are
   private; agree how the publishing workflow will retrieve the wheel and what event
   triggers it. See the [shared KitMaker documentation](https://kitmaker.gitlab-master-pages.nvidia.com/kitmaker-docs/users/portal-api/wheel-release-api.html).

## Recovery

Use **Re-run failed jobs** in Actions to reuse successful jobs and their artifacts.
To repeat all checks, start **Prepare release** again with the same commit SHA.

An existing tag is reused only if it already points directly to the validated commit;
tags are never moved. Existing releases are never overwritten. If draft creation
partially failed, inspect the assets, delete the incomplete unpublished draft, and
rerun the failed release job. Preserve published artifacts and shared tags; changes
require a new version.
