<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze traces from another system

Use a conversion utility for your trace store, written in any language. It writes
canonical JSONL to disk. Trace Analyst can run that utility before each analysis,
or read a file you have already converted.

## Choose your input

Start with a small set of completed agent runs. Give your coding agent either:

- **An export:** the path to a file containing complete traces, including child spans,
  messages, and tool calls and results. Summary tables usually omit this detail.
- **API access:** the store's API documentation and URL, the project and time window,
  and the name of the environment variable holding your API key. Keep the key itself
  in your environment or `.env`.

Select the traces during export or retrieval; the analyst reads the entire converted file.

## Ask your coding agent to convert the traces

Give your agent the input details above and this prompt:

```text
Write and run a small script to convert my traces into traces.jsonl for
Trace Analyst: https://github.com/NVIDIA-NeMo/labs-trace-intel
Use the export or API access details I provided.

Use its current canonical Trace model and write one complete trace per
JSONL line. Preserve trace IDs, span hierarchy, user messages, tool calls
and results, and recorded errors and evaluation scores. Do not invent
missing data. Read any API credentials from the environment or .env.

Validate the output with the canonical filesystem loader. Tell me how many
traces were converted and what data could not be preserved. Leave me the
script and the command to rerun it.
```

The [canonical format](files.md#canonical-jsonl) and
[Trace model](../../packages/trace-ingest/src/trace_ingest/models.py) are the conversion references.
Compare a converted trace with its original before analyzing a larger export.

## Run the analyst

With [uv and Git installed](../../README.md#start-here), install the CLI:

```bash
uv tool install \
  'insight-agent @ git+https://github.com/NVIDIA-NeMo/labs-trace-intel.git@main'
```

[Configure your inference model and key](../model-access.md#choose-a-model),
which is separate from your trace store's key. From the folder containing `traces.jsonl`, run:

```bash
insight-agent --trace.filesystem.path traces.jsonl --max-tokens 16384
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped evidence streams.


## Run your utility from YAML

Configure its executable, arguments, and the destination for validated traces:

```yaml
max_tokens: 16384
trace:
  custom:
    command: ./bin/export-traces
    args:
      - --project
      - my-agent
      - --limit
      - '100'
      - --output
      - '{output_path}'
    output_path: exports/traces.jsonl
    timeout_seconds: 300
```

The flags above belong to your utility; use the flags it actually supports. Each
`args` entry is one literal argument, including values containing spaces. Quote
numeric values as strings. At least one argument must contain `{output_path}`;
`--output={output_path}` also works. The analyst replaces only that placeholder
with a fresh temporary path. The utility must write one complete canonical trace
per line there and exit with status zero. Stdout is for diagnostics, not traces.

Validate the configuration and generated traces without inference:

```bash
insight-agent --config config.yaml --validate-only
```

This **executes the utility**, including any retrieval it performs. It needs the
utility's credentials but no inference key. It reports missing executables,
nonzero exits, timeouts, missing output, empty files, and invalid canonical traces.
On success, it publishes the validated file at `trace.custom.output_path`.
An unsuccessful export leaves an existing destination file unchanged and is
never analyzed. Temporary files are removed after success or failure.

Run the complete analysis with the same configuration:

```bash
insight-agent --config config.yaml
```

Each invocation exports again. To analyze a saved export without rerunning the
utility, use `trace.filesystem.path: exports/traces.jsonl` instead. Set trace limits
using the utility's arguments; `trace.max_traces` is not supported for custom
commands. The analyst validates and reads the entire file. File links refer to
the published JSONL, not the temporary file.

### Executables, environment, and trust

`command` is one executable name on `PATH`, or an executable path. For a Python
script, use its interpreter as the command and the script as the first argument:

```yaml
command: /path/to/exporter/.venv/bin/python
args: [/path/to/exporter/export.py, --output, '{output_path}']
```

Install a third-party CLI in an environment you control and point to its executable,
or put its bin directory on `PATH`. Python imports are resolved by the utility's
own interpreter and environment; there is no class-path import into the analyst.
For a local Python package, install it into that environment (for example, with
`uv pip install --python /path/to/exporter/.venv/bin/python -e /path/to/package`).

The utility inherits the analyst's working directory and environment, including
credentials loaded from `.env`. All relative paths resolve from that working
directory, not the YAML file's directory. Keep secrets in environment variables,
not YAML arguments. There is no shell interpolation of `$VARIABLE`, `~`, globs,
pipes, or redirects. Both stdout and stderr are forwarded to the analyst's stderr.
The utility must avoid logging credentials and wait for its own workers before exiting.

Only run trusted utilities and review command configurations before using them.
They execute with your account's permissions and environment; they are not sandboxed.
The timeout stops the invoked process, but utilities are responsible for any
background processes they spawn.
