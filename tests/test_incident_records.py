#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
POSTMORTEMS.md's incident records must stay true, and must stay complete.

WHY THIS EXISTS
---------------
For thirteen months POSTMORTEMS.md narrated fifteen real defects and cited
zero commits. Every claim in it was unverifiable and nothing in it was
countable -- the difference between an anthology and a dataset. The records
fix that, and this suite keeps them honest, because a stale commit reference
is worse than none: it looks like evidence.

WHAT IT CHECKS
--------------
  1. every postmortem section carries a record -- so a new war story cannot
     be added without its commits, which is how the file drifted before;
  2. every SHA resolves to a real commit in this repository;
  3. every detector path exists on disk;
  4. no incident claims a fix dated before its own introduction, which means
     the record has mixed two instances together (this happened on the first
     run, and produced a -10 day latency);
  5. incidents.csv matches what the generator produces right now.

SHALLOW CLONES
--------------
Check 2 needs real history. `actions/checkout` defaults to `fetch-depth: 1`,
which would make every old SHA unresolvable and this suite would pass by
finding nothing to check. So the shallow case is detected and FAILS loudly
rather than being skipped -- a check that quietly does nothing is the exact
failure mode this repository keeps rediscovering.
"""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _srcload import Results  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

POSTMORTEMS = REPO / "POSTMORTEMS.md"
CSV_PATH = REPO / "incidents.csv"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args],
                                   text=True, stderr=subprocess.DEVNULL).strip()


def test_history_is_available(r: Results) -> bool:
    """A shallow clone makes every other check in here vacuous."""
    shallow = (REPO / ".git" / "shallow").exists()
    r.check(not shallow,
            "the clone has full history, so commit references are checkable",
            "shallow clone -- set `fetch-depth: 0` on actions/checkout, or "
            "this suite silently verifies nothing")
    return not shallow


def test_every_postmortem_has_a_record(r: Results) -> None:
    """A new war story must arrive with its commits, or not at all."""
    import incidents

    text = POSTMORTEMS.read_text()
    headings = [ln[3:].strip() for ln in text.splitlines()
                if ln.startswith("## ") and ln[3:].strip() != "Contents"]
    recorded = {row["title"] for row in incidents.parse()}

    missing = [h for h in headings if h not in recorded]
    r.check(not missing,
            f"every postmortem carries an incident record ({len(headings)})",
            f"no record for: {missing}")


def test_references_resolve(r: Results, have_history: bool) -> None:
    """A stale SHA looks like evidence, which makes it worse than none."""
    import incidents

    rows = incidents.parse()
    bad_sha, bad_det = [], []
    for row in rows:
        shas = [row["introduced"]] if row["introduced"] else []
        shas += row["fixed"].split()
        if have_history:
            for sha in shas:
                try:
                    git("cat-file", "-e", f"{sha}^{{commit}}")
                except subprocess.CalledProcessError:
                    bad_sha.append(sha)
        if row["detector"] and not (REPO / row["detector"]).exists():
            bad_det.append(row["detector"])

    if have_history:
        r.check(not bad_sha, "every referenced commit exists in this repo",
                f"unresolvable: {bad_sha}")
    r.check(not bad_det, "every referenced detector exists on disk",
            f"missing: {bad_det}")


def test_no_incident_is_fixed_before_it_exists(r: Results,
                                               have_history: bool) -> None:
    """
    Negative latency means the record describes two instances as one.

    Caught on the first run: the scoped-claim section covers both the 11_moe
    reversal and the protein length error, and listing all their commits
    together dated the "fix" ten days before the "introduction".
    """
    if not have_history:
        return
    import incidents

    negative = [row["title"] for row in incidents.parse()
                if isinstance(row["latent_days"], int)
                and row["latent_days"] < 0]
    r.check(not negative,
            "no incident is fixed before it is introduced",
            f"record mixes instances: {negative}")


def test_csv_is_current(r: Results) -> None:
    """
    The committed dataset must equal what the generator emits now.

    Same discipline as the published counts: a derived artifact checked in
    without a freshness gate becomes wrong quietly.
    """
    import incidents

    if not CSV_PATH.exists():
        r.check(False, "incidents.csv exists",
                "run: uv run scripts/incidents.py --csv")
        return

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=incidents.FIELDS)
    w.writeheader()
    w.writerows(incidents.parse())

    # Read with newline="" so the comparison is byte-for-byte. `read_text()`
    # applies universal-newline translation, turning csv's \r\n into \n, so
    # a naive compare fails on every platform forever -- which is a check
    # that cries wolf, not a check.
    with CSV_PATH.open(newline="") as fh:
        on_disk = fh.read()

    r.check(buf.getvalue() == on_disk,
            "incidents.csv matches POSTMORTEMS.md",
            "stale -- regenerate with: uv run scripts/incidents.py --csv")


def main() -> int:
    r = Results("Incident records are complete, resolvable and current")
    have = test_history_is_available(r)
    test_every_postmortem_has_a_record(r)
    test_references_resolve(r, have)
    test_no_incident_is_fixed_before_it_exists(r, have)
    test_csv_is_current(r)
    return r.finish()


if __name__ == "__main__":
    raise SystemExit(main())
