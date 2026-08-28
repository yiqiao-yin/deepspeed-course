# Security & Secrets

This repository is **public**. Everything in it — code, configs, docs, sample
outputs — is world-readable. That is intentional; it is a teaching course. What
must never be public is your **credentials**.

## The rule

**No secret is ever committed, and no secret is ever printed.** Every credential
is read from an environment variable at run time:

| Variable | Used by | Required? |
|---|---|---|
| `RUNPOD_API_KEY` | `runpod/runpod_ctl.py` | Only for renting GPUs |
| `HF_TOKEN` | examples downloading gated models (Llama, etc.) | Per example |
| `WANDB_API_KEY` | optional experiment tracking | Never — runs skip W&B if unset |

```bash
export RUNPOD_API_KEY=...    # https://console.runpod.io/user/settings
export HF_TOKEN=...          # https://huggingface.co/settings/tokens
```

Prefer a `.env` file that is **not** tracked, or your shell profile. Do not edit
keys into scripts you might commit.

## Placeholders in shell scripts — and why they are commented

Credential lines in the SLURM scripts look like this:

```bash
# Set this to enable WANDB_API_KEY (optional; scripts skip it when unset):
# export WANDB_API_KEY="your_value_here"
```

They stay **commented and quoted** for two reasons.

**Correctness.** An uncommented `export WANDB_API_KEY=<ENTER_KEY_HERE>` is a
**bash syntax error** — `<` is a redirection operator — so the script aborts on
that line and never reaches the training command. Seven scripts shipped that way
and could never run. `tests/test_runpod_ctl.py` now runs `bash -n` over every
shell script in the repository to stop that recurring.

**Safety.** A commented line cannot accidentally export a placeholder that then
gets logged.

## The results transport is public

`runpod_ctl.py run --collect` has the pod push its progress and log to
[ntfy.sh](https://ntfy.sh), because RunPod exposes **no log endpoint** — verified
against both the REST OpenAPI spec (the `Pod` schema has `portMappings` and
`ports`, nothing log-shaped) and GraphQL introspection (no log/output/console
field on any type). The pod cannot be read from, so it pushes.

**Topics are unguessable but not private.** Each is a random 20-hex-character
string (`dsc-c1b3231b898f4856af25`), so nobody will stumble onto yours — but
anyone who *learns* the topic can read it, and it appears in your terminal
scrollback.

### What is pushed

Only these, by design:

- `nvidia-smi` output
- Python / torch / CUDA / DeepSpeed version banners
- `ds_report` (first 30 lines)
- the training script's stdout

### What must never be pushed

The bootstrap contains no credential references at all, and the test suite
enforces it:

```python
for danger in ("$RUNPOD_API_KEY", "$HF_TOKEN", "$WANDB_API_KEY", "env |", "printenv"):
    assert danger not in bootstrap(...)
```

If you extend the bootstrap, keep it that way. **Never add `env`, `printenv`, or
anything that echoes a token** — it would publish your key to a public feed.

### If that is not good enough

ntfy is self-hostable. Point the tool at your own instance:

```bash
export DSC_NTFY_SERVER=https://ntfy.your-domain.tld
```

Or skip `--collect` entirely and read logs from the RunPod web console.

## The API key never leaves your machine

A deliberate design choice: the pod is **never given** `RUNPOD_API_KEY`.

It would be convenient to let the pod delete itself when finished, but that
means writing your API key onto rented hardware you do not control — a key that
can start pods, and therefore spend money. Instead:

- **Termination is driven locally.** `run --wait --terminate` blocks on your
  machine and deletes the pod in a `finally`, so a crash, a timeout, or Ctrl-C
  still shuts it down.
- **A watchdog inside the pod needs no key.** It sleeps for `--max-hours`
  (default 6) and then kills the container's main process. Belt and braces for
  the case where your laptop closes mid-run.

## Cost is a security property here

An abandoned pod bills until terminated. Treat a forgotten pod like a leaked
key — check for it:

```bash
uv run runpod/runpod_ctl.py pods          # should say "Nothing is billing."
uv run runpod/runpod_ctl.py terminate --all
```

## Reporting a problem

If you find a committed secret or a path that could leak one, open an issue
**without including the secret itself**, and rotate the credential immediately:

- RunPod → https://console.runpod.io/user/settings
- HuggingFace → https://huggingface.co/settings/tokens
- W&B → https://wandb.ai/authorize
