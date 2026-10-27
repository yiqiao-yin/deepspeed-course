#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Turn POSTMORTEMS.md's incident records into a machine-readable table.

POSTMORTEMS.md is the single source of truth: it is what a human reads, and
every fact here is parsed back out of it. This script derives nothing it
cannot trace to a record block, and computes the dates by asking git rather
than trusting anything written down -- a hardcoded "17 days latent" rots the
moment history is rewritten.

    uv run scripts/incidents.py            # print the table
    uv run scripts/incidents.py --csv      # emit incidents.csv

WHY THIS EXISTS
---------------
POSTMORTEMS.md narrates fifteen real defects. For thirteen months it cited no
commits at all, so not one claim in it could be checked against the code, and
nothing in it could be counted. That is the difference between an anthology
and a dataset, and it is why the "is there a paper here" question could not
be answered honestly either way.

With the records in place, the corpus supports claims of the form "N of 15
incidents produced no error at all" and "the median defect survived D days",
which are measurements rather than impressions.

THE CLASSES
-----------
    silent-wrong    ran clean, produced plausible output, was incorrect
    silent-noop     reported success having done nothing
    false-green     a check or harness passed input it should have rejected
    hang            no error, no progress
    fails-loud      crashed -- the easy class, and the rare one here
    doc-drift       a published claim diverged from the code it described
    cross-cutting   a lesson spanning several incidents, not a defect itself
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POSTMORTEMS = REPO / "POSTMORTEMS.md"
OUT = REPO / "incidents.csv"

RECORD = re.compile(
    r"^## (?P<title>.+?)\n+> \*\*Incident record\*\*(?P<body>(?:.|\n)*?)(?=\n\n)",
    re.M,
)
FIELDS = ["title", "class", "introduced", "fixed", "detector",
          "latent_days", "fix_span_days"]


def commit_date(sha: str) -> date | None:
    """Ask git, so the dates cannot drift from the history."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "log", "-1", "--format=%as", sha],
            text=True, stderr=subprocess.DEVNULL).strip()
        return date(*map(int, out.split("-")))
    except (subprocess.CalledProcessError, ValueError):
        return None


def parse() -> list[dict]:
    text = POSTMORTEMS.read_text()
    rows = []
    for m in RECORD.finditer(text):
        body = m.group("body")
        cls = re.search(r"class `([a-z-]+)`", body)
        intro = re.search(r"introduced \[`([0-9a-f]+)`\]", body)
        fixed = re.findall(r"\[`([0-9a-f]{7,40})`\]", body.split("fixed")[-1]) \
            if "fixed" in body else []
        det = re.search(r"detector \[`([^`]+)`\]", body)

        row = {
            "title": m.group("title").strip(),
            "class": cls.group(1) if cls else "",
            "introduced": intro.group(1) if intro else "",
            "fixed": " ".join(fixed),
            "detector": det.group(1) if det else "",
            "latent_days": "",
            "fix_span_days": "",
        }

        d_intro = commit_date(row["introduced"]) if row["introduced"] else None
        d_fixes = [d for d in (commit_date(f) for f in fixed) if d]
        if d_intro and d_fixes:
            row["latent_days"] = (min(d_fixes) - d_intro).days
        if len(d_fixes) > 1:
            row["fix_span_days"] = (max(d_fixes) - min(d_fixes)).days
        rows.append(row)
    return rows


def main() -> int:
    rows = parse()
    if not rows:
        print("No incident records found in POSTMORTEMS.md", file=sys.stderr)
        return 1

    if "--csv" in sys.argv:
        with OUT.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {OUT.relative_to(REPO)} ({len(rows)} incidents)")
        return 0

    width = max(len(r["title"]) for r in rows)
    print(f"{'INCIDENT'.ljust(width)}  {'CLASS':<14} {'LATENT':>7}")
    print("-" * (width + 24))
    for r in rows:
        lat = f"{r['latent_days']}d" if r["latent_days"] != "" else "—"
        print(f"{r['title'].ljust(width)}  {r['class']:<14} {lat:>7}")

    # The headline the corpus actually supports.
    loud = sum(1 for r in rows if r["class"] == "fails-loud")
    cross = sum(1 for r in rows if r["class"] == "cross-cutting")
    defects = len(rows) - cross
    lats = [r["latent_days"] for r in rows if r["latent_days"] != ""]
    bad = [r["title"] for r in rows
           if isinstance(r["latent_days"], int) and r["latent_days"] < 0]
    print()
    print(f"{len(rows)} records · {defects} defects, {cross} cross-cutting")
    print(f"{defects - loud} of {defects} defects did NOT crash")
    if bad:
        # A fix dated before its own introduction means the record mixes two
        # instances. Surfaced rather than silently absorbed into the median.
        print(f"WARNING negative latency -- record mixes instances: {bad}")
    if lats:
        lats = sorted(lats)
        print(f"latency: median {lats[len(lats) // 2]}d, "
              f"min {lats[0]}d, max {lats[-1]}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
