<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Top-level ETHOS path

Applies to the change immediately after Compass revision
`8cdf744f417b89d46cd91543c0fe771868850c42` that introduces `RunConfig.ethos_path`.
Check the target source/schema for this field; the package version alone is not enough
to distinguish source revisions. The old location is rejected, with no compatibility alias.

## YAML and CLI

Before:

```yaml
compass_signals:
  ethos_divergence:
    ethos_path: ETHOS.md
```

After:

```yaml
ethos_path: ETHOS.md
compass_signals:
  ethos_divergence: true
```

The signal is enabled by default, so its entry may be omitted. Preserve an explicit
`false`; supplying an ETHOS document does not enable a disabled signal.
Replace `--compass-signals.ethos-divergence.ethos-path` with `--ethos-path`.
Update JSON object arguments and config generators too. Relative paths still resolve
from the process working directory, not the YAML directory. A supplied path must be a
file even if ethos divergence is disabled. Enabled ethos divergence rejects empty
content and skips when no document is supplied.

## Python callers

Move `ethos_path=path` from `EthosDivergenceConfig(...)` to `RunConfig(...)`.
Keep `EthosDivergenceConfig()` as the signal selection/config object. At the execution
boundary, pass the shared path explicitly:

```python
registry = registered_builtin_signals(
    ethos_divergence=config.compass_signals.ethos_divergence,
    ethos_path=config.ethos_path,
    llm_factory=llm_factory,
)
```

Preserve the other registry arguments. Direct signal callers use
`EthosDivergenceCompassSignal(config=EthosDivergenceConfig(), llm=llm, ethos_path=path)`.
Callers of the private `_run_compass_signals` helper must also pass `ethos_path=`.
Avoid `model_copy(update={"ethos_path": path})` on the signal config: Pydantic copies
can bypass validation, but the signal no longer reads that field.

## Known downstream patterns

NeMo Helix's inspected adapter writes inline ETHOS text to a temporary file and inserts
its path into the signal config. Pass that path to the registry instead, retaining the
temporary directory for the entire analysis. Keep Helix's external inline `ethos` field
and `--ethos` option: those are Helix interfaces, not Compass config names.
The inspected Helix dependency pin was `cd639d223e06ee9d27783df6b461d302ceb09137`;
it is unaffected until upgraded. Its adapter also uses `EvidenceStreamsConfig` and
`registered_builtin_streams`, so an upgrade from that revision needs the intervening
Compass Signal API migration as well.

Older analyst-evals runners similarly construct nested configs and call private
execution helpers. Migrate those callers when selecting a new Compass revision,
without changing archived run manifests. Consumers importing only `trace-ingest`
are unaffected by this configuration change. These are inspected integration patterns,
not a guarantee that every downstream consumer has been identified or tested.
