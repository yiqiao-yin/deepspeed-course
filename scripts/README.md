# Maintenance Scripts

Advisory tooling. **Nothing here gates CI** — see the note on false positives.

## `audit_readmes.py`

```bash
uv run scripts/audit_readmes.py
```

Compares every example README against the code it documents and reports likely
drift:

- referenced `.py` / `.sh` / `.json` files that do not exist
- quoted config values that disagree with the real `ds_config.json`
- documented symbols absent from the neighbouring source

### Read the output; do not auto-fix it

This tool **over-reports by design** and needs human triage. Teaching READMEs
legitimately contain code the reader is invited to write, and remediation advice
that deliberately differs from the shipped config. Both look like drift to a
static checker.

Measured false-positive rate on real runs:

| Run | Flagged | Actually real |
|---|---|---|
| First (before de-noising) | 31 | **2** |
| After skipping remediation blocks | 17 | **0** |

The two real findings it did catch were worth the noise: a README documenting a
deleted function with a line-number citation, and a reference to a
`ds_config_zero1.json` that never existed.

Typical false positives, all legitimate content:

- `"train_batch_size": 16` under *"CUDA Out of Memory → reduce batch size"* —
  remediation advice, not a claim about the config
- `def math_accuracy_reward(...)` under *"Create domain-specific rewards"* —
  a suggestion for the reader
- `tokenizer.json` — an output artifact of training, not a repo file

**This is why it is not a CI gate.** A checker that cries wolf on every push
gets ignored, and then misses the one finding that mattered. Run it when you
change an example, and read the findings yourself.

For checks that *are* deterministic enough to gate — config validity, shell
syntax, SLURM coverage — see [`tests/`](../tests/README.md), which does run in CI.
