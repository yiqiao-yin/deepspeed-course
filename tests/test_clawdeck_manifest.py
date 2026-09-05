# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""
Regression test: the Clawdeck lab manifest (clawdeck.yaml).

Run:
    uv run tests/test_clawdeck_manifest.py

Why this suite exists
---------------------
clawdeck-app.com boots a GPU box, clones this repo, and builds a Lab picker
from `clawdeck.yaml`. It is the ONLY integration point between the two
products, which makes it the only place a change here can silently break
something over there.

It already happened once: Clawdeck hardcoded the path `01_basic_neuralnet`,
this repo reorganised into `01_basics/…`, and every Clawdeck GPU boot printed
"dependency pre-install FAILED" until someone noticed. Nothing in either
codebase could have caught that, because the coupling was a string in a
different repository.

So the manifest is checked here, in a suite that FAILS CI — not in
`scripts/check_contract.py`, which is advisory by design (CLAUDE.md is explicit
that older examples legitimately differ from that contract, and making it a
gate would force the checks to be watered down until they catch nothing). The
manifest has no such grey area: an id either resolves to a real lab or it does
not.

The checks that matter, in order of how quietly they fail:

  * **A lab directory missing from the manifest.** Add a topic, forget the
    manifest, and it simply never appears in Clawdeck. No error anywhere. This
    is the check the whole file exists for.
  * **--num_gpus disagreeing with gpu.count.** Several ds_config.json files pin
    the GPU count through DeepSpeed's batch invariant
    (train_batch_size == micro x accum x num_gpus). Ship a command whose
    --num_gpus contradicts the declared count and the user clicks Run and gets
    an assertion from inside DeepSpeed, having already paid for the box.
  * **A cmd naming a script that does not exist** — a rename that updated the
    folder but not the command.
  * **Zero or several `primary: true`** — the UI has one primary button.
"""

import os
import re
import shlex
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "clawdeck.yaml"

# The picker renders in a ~320px sidebar. These are the widths that fit.
MAX_TITLE = 40
MAX_SUMMARY = 95

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def lab_dirs_on_disk() -> set:
    """Every directory that is a uv project, i.e. every runnable lab."""
    out = set()
    for p in REPO.rglob("pyproject.toml"):
        rel = p.parent.relative_to(REPO)
        parts = rel.parts
        if not parts or parts[0].startswith("."):
            continue
        if parts[0] in ("docusaurus-docs", "node_modules", "scripts", "tests",
                        "runpod"):
            continue
        if any(x in (".venv", "node_modules") for x in parts):
            continue
        out.add(str(rel))
    return out


def main() -> None:
    bar = "=" * 74
    print(bar)
    print("  test_clawdeck_manifest.py")
    print(bar)

    check("clawdeck.yaml exists at the repo root", MANIFEST.is_file(),
          "clawdeck-app.com fetches this path from GitHub")
    if not MANIFEST.is_file():
        print(f"\n  {PASS} passed, {FAIL} failed")
        sys.exit(1)

    try:
        m = yaml.safe_load(MANIFEST.read_text())
        parsed = True
    except Exception as exc:
        m, parsed = None, False
        check("clawdeck.yaml is valid YAML", False, str(exc)[:200])
    if not parsed:
        sys.exit(1)
    check("clawdeck.yaml is valid YAML", True)

    check("version is 1", m.get("version") == 1, f"got {m.get('version')!r}")
    labs = m.get("labs") or []
    check("labs is a non-empty list", isinstance(labs, list) and labs)

    ids = [l.get("id") for l in labs]
    check("no duplicate ids", len(ids) == len(set(ids)),
          f"duplicates: {[i for i in set(ids) if ids.count(i) > 1]}")

    # ---- default_lab -------------------------------------------------------
    print("\n  -- default_lab --")
    dl = m.get("default_lab")
    check(f"default_lab ({dl!r}) is one of the ids", dl in ids,
          "Clawdeck pre-warms this lab's uv cache on every boot; if it does "
          "not resolve, every boot prints a pre-install failure")

    # ---- every id resolves -------------------------------------------------
    print("\n  -- ids resolve to real uv projects --")
    for lab in labs:
        i = lab.get("id", "")
        d = REPO / i
        check(f"{i}: directory exists", d.is_dir())
        check(f"{i}: has pyproject.toml", (d / "pyproject.toml").is_file(),
              "Clawdeck runs `uv run` in this directory")

    # ---- THE important one: nothing on disk is missing ---------------------
    print("\n  -- every lab on disk is registered --")
    on_disk = lab_dirs_on_disk()
    declared = set(ids)
    missing = sorted(on_disk - declared)
    extra = sorted(declared - on_disk)
    check(f"all {len(on_disk)} lab directories appear in the manifest",
          not missing,
          f"missing: {missing} — these exist and are runnable, but would "
          "never appear in Clawdeck, silently")
    check("no manifest id points at a directory that is not a lab", not extra,
          f"unknown: {extra}")

    # ---- run entries -------------------------------------------------------
    print("\n  -- run entries --")
    for lab in labs:
        i = lab.get("id", "")
        runs = lab.get("run") or []
        check(f"{i}: has at least one run entry", bool(runs))
        primaries = [r for r in runs if r.get("primary") is True]
        check(f"{i}: exactly one primary ({len(primaries)})",
              len(primaries) == 1,
              "the UI has one primary button")
        for r in runs:
            check(f"{i}: run entry has a label and a cmd",
                  bool(r.get("label")) and bool(r.get("cmd")))

    # ---- commands actually reference something that exists -----------------
    print("\n  -- commands resolve --")
    for lab in labs:
        i = lab.get("id", "")
        d = REPO / i
        for r in lab.get("run") or []:
            cmd = r.get("cmd", "")
            toks = shlex.split(cmd)
            check(f"{i}: cmd starts with `uv run` ({r.get('label')!r})",
                  toks[:2] == ["uv", "run"],
                  f"got {cmd!r}; the box has uv and a committed lock, so uv "
                  "run is the only prefix that needs no setup")
            script = next((t for t in toks if t.endswith(".py")), None)
            check(f"{i}: names a .py file ({r.get('label')!r})",
                  script is not None, f"got {cmd!r}")
            if script:
                check(f"{i}: {script} exists", (d / script).is_file(),
                      "a rename updated the folder but not the command")

    # ---- --num_gpus must match the declared count --------------------------
    # This is the batch-invariant trap. Two EXAMPLES entries in this repo were
    # already wrong this way when the manifest was written.
    print("\n  -- --num_gpus agrees with gpu.count --")
    for lab in labs:
        i = lab.get("id", "")
        want = (lab.get("gpu") or {}).get("count")
        for r in lab.get("run") or []:
            mm = re.search(r"--num_gpus=(\d+)", r.get("cmd", ""))
            if not mm:
                continue
            got = int(mm.group(1))
            check(f"{i}: --num_gpus={got} matches gpu.count={want} "
                  f"({r.get('label')!r})",
                  want == got,
                  "DeepSpeed's batch invariant is checked at startup; a "
                  "mismatch aborts the run after the user has paid for a box")

    # ---- gpu.count must satisfy the ds_config batch invariant -------------
    # The manifest is not the only place this count lives. Where a
    # ds_config.json hardcodes train_batch_size, micro batch and grad accum,
    # DeepSpeed solves for num_gpus at startup and aborts on a mismatch. Two
    # entries in runpod_ctl.py's EXAMPLES table were already wrong this way
    # when the manifest was written, so this cross-check is not hypothetical.
    print("\n  -- gpu.count satisfies the DeepSpeed batch invariant --")
    import glob as _glob
    import json as _json
    for lab in labs:
        i = lab.get("id", "")
        want = (lab.get("gpu") or {}).get("count")
        cfgs = sorted(_glob.glob(str(REPO / i / "ds_config*.json")))
        if not cfgs or want is None:
            continue
        try:
            c = _json.loads(Path(cfgs[0]).read_text())
        except Exception:
            continue
        tb = c.get("train_batch_size")
        mb = c.get("train_micro_batch_size_per_gpu")
        ga = c.get("gradient_accumulation_steps", 1)
        if not (isinstance(tb, int) and isinstance(mb, int) and isinstance(ga, int)):
            continue        # "auto", or omitted, so any count is fine
        implied = tb / (mb * ga)
        check(f"{i}: gpu.count={want} satisfies {tb} == {mb} x {ga} x N",
              implied == want,
              f"{Path(cfgs[0]).name} implies N={implied:g}; DeepSpeed asserts "
              "this at startup, so the Run button would abort")

    # ---- gpu shapes Clawdeck can actually book ----------------------------
    # MIRROR OF CLAWDECK'S MACHINE CATALOG, not a property of this repo.
    # Clawdeck matches a lab to the cheapest machine satisfying it, and requires
    # an EXACT GPU count, because DeepSpeed is launched with --num_gpus=N and
    # aborts if that disagrees with the hardware. A lab whose shape no catalog
    # entry satisfies renders as "Needs a different machine" with nothing to
    # switch to -- a dead end, and nothing anywhere logs a word about it.
    #
    # count -> the largest per-GPU VRAM available at that count.
    BOOKABLE = {1: 180, 2: 180, 4: 80, 8: 80}
    print("\n  -- gpu shapes are bookable on Clawdeck --")
    for lab in labs:
        i = lab.get("id", "")
        g = lab.get("gpu")
        if not g:
            continue                      # no gpu block == CPU lab, always fine
        cnt, vram = g.get("count"), g.get("min_vram_gb")
        check(f"{i}: gpu.count={cnt} is a bookable count {sorted(BOOKABLE)}",
              cnt in BOOKABLE,
              "Clawdeck books an EXACT count. 3 is a reasonable thing to write "
              "and cannot be booked; the lab would show as 'Needs a different "
              "machine' with no machine to switch to.")
        if cnt in BOOKABLE:
            check(f"{i}: {cnt} x {vram} GB exists in the catalog "
                  f"(max {BOOKABLE[cnt]} GB at that count)",
                  vram <= BOOKABLE[cnt],
                  f"no Clawdeck machine offers {vram} GB per GPU at count "
                  f"{cnt}. This is a PLATFORM CAPACITY limit, not a typo in "
                  "your lab -- either lower the requirement or use a count "
                  "that offers bigger cards.")

    # ---- entries with no --num_gpus are advertised as needing no GPU -------
    # Clawdeck's rule is exactly `is_cpu_only = "--num_gpus" not in cmd`, and
    # such entries are shown FIRST, under "Runs now - no GPU needed", drawn
    # even from locked labs. It is what a learner clicks while the machine is
    # still installing. An entry that lands there and then needs a GPU is worse
    # than a locked lab: the learner was told it would work.
    #
    # `needs_gpu: true` marks an entry that genuinely needs a GPU but cannot say
    # so through --num_gpus, because its example deliberately does not use the
    # deepspeed launcher (CLAUDE.md lists five such examples; using a
    # distributed launcher where there is nothing to distribute is cargo cult,
    # and faking one here purely to smuggle a GPU signal would be worse).
    # CLAWDECK DOES NOT READ THIS FIELD YET -- see the note in clawdeck.yaml.
    GPU_LAUNCHERS = ("torchrun", "accelerate launch", "mpirun",
                     "deepspeed.init_distributed", "torch.distributed.run")
    # Flags whose code path returns before require_gpu() is ever reached.
    CPU_SAFE_FLAGS = ("--plan", "--verify-arch", "--list-methods",
                      "--list-models", "--dry-run")
    print("\n  -- entries advertised as 'no GPU needed' really are --")
    for lab in labs:
        i = lab.get("id", "")
        d = REPO / i
        for r in lab.get("run") or []:
            cmd = r.get("cmd", "")
            if "--num_gpus" in cmd:
                continue
            label = r.get("label")
            if r.get("needs_gpu") is True:
                check(f"{i}: {label!r} is marked needs_gpu", True)
                continue
            script = next((t for t in shlex.split(cmd) if t.endswith(".py")), None)
            if not script or not (d / script).is_file():
                continue                  # already reported above
            src = (d / script).read_text(errors="ignore")

            # A GPU launcher invoked from inside the script is unambiguous.
            found = [t for t in GPU_LAUNCHERS if t in src]
            check(f"{i}: {label!r} launches no GPU-shaped process",
                  not found,
                  f"{script} references {found}; Clawdeck decides CPU-only by "
                  "the absence of --num_gpus, so this would be advertised "
                  "under 'Runs now - no GPU needed' and fail when clicked. "
                  "Route it through `deepspeed --num_gpus=N`, or add "
                  "`needs_gpu: true`.")

            # require_gpu() means the script exits without a GPU, unless this
            # invocation takes a documented early-return path.
            if "require_gpu()" in src:
                safe = [f for f in CPU_SAFE_FLAGS if f in cmd]
                check(f"{i}: {label!r} reaches a CPU path despite require_gpu()",
                      bool(safe),
                      f"{script} calls require_gpu(), and this command passes "
                      "no flag that returns before it "
                      f"({', '.join(CPU_SAFE_FLAGS)}). Clawdeck would advertise "
                      "it under 'Runs now - no GPU needed' and the learner "
                      "would get the no-GPU preflight instead. Add "
                      "`needs_gpu: true`, or give it a real CPU path.")

    # ---- fields the UI depends on -----------------------------------------
    print("\n  -- fields the picker renders --")
    for lab in labs:
        i = lab.get("id", "")
        t, s = lab.get("title", ""), lab.get("summary", "")
        check(f"{i}: has a title", bool(t))
        check(f"{i}: title fits the sidebar ({len(t)} <= {MAX_TITLE})",
              len(t) <= MAX_TITLE, f"{t!r}")
        check(f"{i}: has a summary", bool(s))
        check(f"{i}: summary fits ({len(s)} <= {MAX_SUMMARY})",
              len(s) <= MAX_SUMMARY, f"{s!r}")
        em = lab.get("est_minutes")
        check(f"{i}: est_minutes is a positive int", isinstance(em, int) and em > 0,
              f"got {em!r}")
        g = lab.get("gpu")
        if g is not None:
            check(f"{i}: gpu has min_vram_gb and count",
                  isinstance(g.get("min_vram_gb"), (int, float))
                  and isinstance(g.get("count"), int) and g["count"] >= 1,
                  f"got {g!r}")

    print("\n" + bar)
    print(f"  {PASS} passed, {FAIL} failed")
    print(bar)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
