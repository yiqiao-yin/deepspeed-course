#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Every protein memory figure in the docs must trace to the shipped function.

WHY THIS EXISTS
---------------
CLAUDE.md published "the AF2->AF3 saving is 58% at 128 residues and 12% at
256". Both figures were real and one length was wrong: 58.5% is the saving at
32 residues, not 128, where the true value is 24.2% -- less than half what was
printed. It passed review and a green CI run, because nothing compared the
prose against the table it came from.

The repository already had the rule it broke ("a measured claim is scoped to
the configuration it was measured in"). A rule with no check is a hope, so
this is the check.

HOW IT WORKS
------------
Two different kinds of number live in these pages and they need different
treatment:

  * **Computable** -- the analytic activation tables. These are pure
    arithmetic over tensor shapes, so the test RECOMPUTES them by running
    `trunk_activation_table()` out of the shipped `pairformer.py` and
    compares. If someone edits the model widths, the published tables go red.

  * **Measured** -- numbers that came off a real GPU and cannot be recomputed
    on CI. For these the test enforces a SINGLE SOURCE OF TRUTH: the measured
    table in `pairformer.md` owns them, and every other mention must agree
    with that table. That is exactly the cross-document link that was missing.

The function is extracted with `ast` via `tests/_srcload.py`, so this runs
against the real shipped source without importing torch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _srcload import Results, load_function  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PAIRFORMER_PY = REPO / "06_protein_folding" / "03_pairformer" / "pairformer.py"
PAIRFORMER_MD = REPO / "docusaurus-docs/docs/tutorials/protein/pairformer.md"
EVOFORMER_MD = REPO / "docusaurus-docs/docs/tutorials/protein/evoformer.md"
ZERO_MD = REPO / "docusaurus-docs/docs/getting-started/deepspeed-zero-stages.md"
CLAUDE_MD = REPO / "CLAUDE.md"


def saving(table: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    """Totals and the percentage AF3 saves, from a raw activation table."""
    af2 = sum(v["af2"] for v in table.values())
    af3 = sum(v["af3"] for v in table.values())
    return af2, af3, (af2 - af3) / af2 * 100


def test_analytic_tables_are_recomputable(r: Results) -> None:
    """
    The published analytic tables must equal what the shipped code computes.

    Not "look plausible" -- equal. These are closed-form expressions over
    tensor shapes, so there is no excuse for a published figure that the
    function does not reproduce.
    """
    fn = load_function(PAIRFORMER_PY, "trunk_activation_table")
    md = PAIRFORMER_MD.read_text()

    # The "lesson" table, which is where the decay with length is argued.
    # n_seq=128 throughout; confirmed by the AF2 totals reproducing exactly.
    published = {128: 47.1, 256: 32.0, 512: 19.5, 1024: 11.0}
    for n_res, pct in published.items():
        _, _, got = saving(fn(n_res=n_res, n_seq=128))
        r.check(abs(got - pct) < 0.1,
                f"AF3 saving at {n_res} residues is {pct}% as published",
                f"the shipped function computes {got:.1f}%")

    # The detailed per-tensor table. Its configuration is n_res=384, which is
    # `print_comparison`'s default -- and is NOT a row label anywhere, which
    # is precisely why a reader can mistake it for one of the rows above.
    t = fn(n_res=384, n_seq=128)
    af2, af3, pct = saving(t)
    for label, key, want in [
        ("MSA row attention logits", "MSA row attention logits", 151.0),
        ("pair representation", "pair representation", 37.7),
        ("triangle attention logits", "triangle attention logits", 453.0),
    ]:
        r.check(abs(t[key]["af2"] - want) < 0.1,
                f"detailed table: {label} is {want} MB at 384 residues",
                f"computed {t[key]['af2']:.1f} MB")
    r.check(abs(af2 - 648.0) < 0.5 and abs(af3 - 491.0) < 0.5,
            "detailed table totals are 648.0 / 491.0 MB",
            f"computed {af2:.1f} / {af3:.1f}")
    r.check(abs(pct - 24.2) < 0.1,
            "the headline 24.2% saving is at 384 residues",
            f"computed {pct:.1f}%")
    r.check("**24.2%**" in md,
            "pairformer.md still prints that 24.2%")


def test_evoformer_cubic_row(r: Results) -> None:
    """
    The 1024-residue row is quoted on three pages; all three must agree.

    268.4 MB of pair representation against 8589.9 MB of triangle logits is
    the single most-reused pair of numbers in the section -- it appears in
    evoformer.md's table, in its prose, and in the ZeRO page's section 6.2,
    which is two documents away from the code that produces it.
    """
    fn = load_function(PAIRFORMER_PY, "trunk_activation_table")
    t = fn(n_res=1024, n_seq=128)
    pair = t["pair representation"]["af2"]
    tri = t["triangle attention logits"]["af2"]

    r.check(abs(pair - 268.4) < 0.1, "pair representation at 1024 is 268.4 MB",
            f"computed {pair:.1f}")
    r.check(abs(tri - 8589.9) < 0.1, "triangle logits at 1024 are 8589.9 MB",
            f"computed {tri:.1f}")
    r.check(abs(tri / pair - 32.0) < 0.1, "the ratio is the published 32.0x",
            f"computed {tri / pair:.1f}x")

    evo = EVOFORMER_MD.read_text()
    r.check("268.4 MB" in evo and "8589.9 MB" in evo,
            "evoformer.md's table still carries both figures")

    zero = ZERO_MD.read_text()
    r.check("268 MB" in zero and "8,590 MB" in zero,
            "the ZeRO page's section 6.2 rounds the same two figures",
            "if these diverge, a reader comparing the pages sees a "
            "contradiction with no way to tell which is right")


def test_measured_figures_have_one_owner(r: Results) -> None:
    """
    Measured numbers cannot be recomputed, so they get a single source.

    The measured table in pairformer.md is that source. Anything quoting it
    elsewhere -- CLAUDE.md does -- must match, WITH its residue count. The
    bug this suite exists for was exactly a correct figure attached to the
    wrong length, so checking the number alone would not have caught it.
    """
    md = PAIRFORMER_MD.read_text()

    # Pull the measured table straight out of the page.
    rows = dict(re.findall(r"^\|\s*(\d+)\s*\|[^|]*\|[^|]*\|\s*\**([\d.]+)%",
                           md, re.M))
    for n_res, pct in [("32", "58.5"), ("128", "24.2"), ("256", "12.5")]:
        r.check(rows.get(n_res) == pct,
                f"pairformer.md measures {pct}% at {n_res} residues",
                f"table says {rows.get(n_res)!r} -- the owning table moved, "
                f"so every page quoting it is now stale")

    claude = CLAUDE_MD.read_text()
    for n_res, pct in [("32", "58.5"), ("128", "24.2"), ("256", "12.5")]:
        r.check(f"{pct}% ({n_res}" in claude,
                f"CLAUDE.md pairs {pct}% with {n_res} residues",
                "a measured claim is scoped to the configuration it was "
                "measured in -- this exact pairing shipped wrong once")

    r.check("58.5% (32 residues), 24.2% (128), 12.5% (256)" in claude,
            "CLAUDE.md keeps all three points, not one",
            "quoting a single figure from a decaying curve is the defect "
            "itself, not a shortening of it")


def main() -> int:
    r = Results("Published protein numbers trace to the shipped function")
    test_analytic_tables_are_recomputable(r)
    test_evoformer_cubic_row(r)
    test_measured_figures_have_one_owner(r)
    return r.finish()


if __name__ == "__main__":
    raise SystemExit(main())
