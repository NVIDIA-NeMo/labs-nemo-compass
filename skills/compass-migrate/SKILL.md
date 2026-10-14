---
name: compass-migrate
description: Migrate a downstream NeMo Compass integration across documented breaking changes to configuration, CLI flags, or Python APIs. Use when upgrading Compass or repairing an integration after an upgrade.
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Migrate a Compass integration

Establish the consumer's installed or pinned Compass revision and target revision first.
Inspect its dependency declaration and lockfile, generated configs, CLI wrappers, and
Python callers. A matching field name alone does not prove that it belongs to Compass.

Read only the guides relevant to the upgrade:

- [Top-level ETHOS path](references/top-level-ethos-path.md): moving
  `compass_signals.ethos_divergence.ethos_path` to `ethos_path`, including Python
  registry callers and older integrations using evidence streams.

Apply the documented changes to the consumer's templates and generators as well as
saved configurations. Preserve unrelated settings, disabled signals, and path resolution.
If old and new values conflict, establish the intended document before overwriting either.
Do not rewrite historical run artifacts or unrelated schemas.

Validate config parsing and exercise the adapter with a fixture ETHOS document and a
mock model. Verify that the document reaches detection and that disabling the signal
still works. If the upgrade also crosses undocumented API changes, inspect those changes
and report remaining gaps; this catalog is not a complete history of Compass.

Update dependency pins and lockfiles only as part of the requested upgrade, using a
revision containing the migration. Report the tested revisions, affected interfaces,
and any runtime verification still needed. Upgrading a repository does not authorize
publishing a package or deploying the consumer.
