<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Preparing a Compass release

Pushing a `v<version>` tag runs CI and creates a **draft GitHub release** containing
one wheel, one source distribution, and `SHA256SUMS`. Nothing is uploaded to PyPI,
NVIDIA's Python index, or KitMaker by this workflow.

## Before the first release

- Merge the packaging consolidation: the wheel must contain both `insight_agent`
  and `trace_ingest`, with no dependency on a separate `trace-ingest` distribution.
- Set `[project].name` to `nemo-compass`, update the lockfile and packaging references
  (including CI's wheel install command and the distribution license checker), and
  merge that change to `main`.
- Confirm the release name, version, legal/security approvals, and KitMaker project
  owner with the publishing team. Arrange KitMaker access and the handoff with Philip.
- Allow GitHub Actions to create releases with the repository's `GITHUB_TOKEN`.
  No PyPI or KitMaker credentials are needed for artifact preparation.

The release validator rejects the old `insight-agent` distribution name and any
extra artifacts. The automation can land before the packaging changes, but release
tags will fail validation until those prerequisites are merged.

## Prepare a version

1. Set `[project].version` in `pyproject.toml`, run `uv lock`, and merge through the
   usual CI checks. The tag does **not** set or override this version. For example,
   version `0.1.0rc1` requires tag `v0.1.0rc1`; version `0.1.0` requires `v0.1.0`.
2. Tag the intended commit on `main` and push the tag. For a release candidate:

   ```bash
   git fetch origin main
   git tag -a v0.1.0rc1 origin/main -m "NeMo Compass 0.1.0rc1"
   git push origin v0.1.0rc1
   ```

3. Review the draft under the repository's **Releases** page after CI succeeds.
   CI runs the normal checks, builds and checks distributions, and tests a clean
   wheel installation. It then verifies the tag, project and artifact metadata,
   bundled packages, and absence of a separate ingestion dependency.
   Alpha, beta, release candidate, and development versions are marked prereleases.
4. Review the generated notes and artifact checksums. The draft contains the same
   bytes checked by CI; its creation job downloads those artifacts without rebuilding.

The tagged commit must be reachable from `main`. The workflow grants write access
only to the draft-creation job, which does not check out or execute repository code.

## Handoff to KitMaker

Once approved, publish the reviewed **GitHub release** to make its asset URLs
available to KitMaker. A draft's private download URLs do not work for KitMaker's
public GitHub asset source. Publishing the GitHub release and publishing to Python
indexes are separate actions; coordinate any external KitMaker webhook first.

Give the publishing owner the version, commit/tag, direct release asset URL(s), and
`SHA256SUMS`. Register the intended wheel/source artifacts with KitMaker, validate
with `upload=false`, review the results, then request the separately approved
`upload=true` publication of those same files. KitMaker's static validation does not
replace the required legal/security approvals.

See the [KitMaker wheel release API](https://kitmaker.gitlab-master-pages.nvidia.com/kitmaker-docs/users/portal-api/wheel-release-api.html).
Once a version is shared, preserve its tag and artifact bytes; use a new version for changes.

## Retry a failed run

Use **Re-run failed jobs** in Actions. To start the workflow again for an existing tag:

```bash
gh workflow run ci.yaml --ref v0.1.0rc1
```

Manual runs on branches perform CI checks without creating releases. A successful
release is never overwritten: creating a release for an existing tag fails. If draft
creation failed partway through, inspect its assets first. An incomplete, unpublished
draft may be deleted before retrying the failed draft-creation job, which reuses the
original Actions artifacts. Do not delete a published release or move a shared tag.
