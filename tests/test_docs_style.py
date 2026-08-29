# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Regression test: documentation-site conventions, including the mermaid theme.

Run:
    uv run tests/test_docs_style.py

Why this suite exists
---------------------
The book is dark-only, and the mermaid theme — ELK layout, dark-blue boxes and
containers, white type, grey arrows — is set globally in `docusaurus.config.js`.
Nothing in the Docusaurus build enforces that a *diagram* honours it, so a
single off-palette diagram renders as a bright grey box on a black page and is
instantly obvious to a reader while being invisible to CI.

Three things are pinned here:

  1. **Every mermaid diagram declares the house classDefs.** Copy-pasted from
     the block in CONTRIBUTING.md §8.
  2. **No diagram carries an inline theme override.** `%%{init: ...}%%` or a
     per-diagram `layout:` fights the global config and drifts silently the
     moment the config changes.
  3. **The global config still says what CONTRIBUTING.md claims it says.** The
     documented hexes and the configured hexes must not diverge — a contributor
     copying the palette out of the docs has to get the colours that are
     actually rendered.

Plus the structural rules that have bitten before: every page needs
`sidebar_position` frontmatter, and every page must be listed in `sidebars.js`
or it is silently orphaned.

Pure stdlib. No GPU, no network, no Node.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docusaurus-docs" / "docs"
CONFIG = REPO_ROOT / "docusaurus-docs" / "docusaurus.config.js"
SIDEBARS = REPO_ROOT / "docusaurus-docs" / "sidebars.js"

# The house palette, exactly as CONTRIBUTING.md §8 publishes it.
PALETTE = {
    "deep":   ("#08182a", "#2d5a86"),
    "dark":   ("#0a1f33", "#2d5a86"),
    "base":   ("#16324f", "#3f6f9f"),
    "bright": ("#1e5f8f", "#63a3d0"),
    "steel":  ("#28527a", "#6aa2cd"),
}


def mermaid_blocks(text: str):
    """Every ```mermaid fenced block in a page."""
    return re.findall(r"```mermaid\n(.*?)```", text, re.S)


def pages():
    return sorted(DOCS.rglob("*.md"))


def test_global_theme(r: Results) -> None:
    """The config must still configure what CONTRIBUTING.md documents."""
    cfg = CONFIG.read_text()

    r.check("layout: 'elk'" in cfg or 'layout: "elk"' in cfg,
            "ELK layout is set globally",
            "contributors must not need to declare it per diagram")
    r.check("nodePlacementStrategy" in cfg,
            "ELK node placement strategy is configured")

    for key, want in [("mainBkg", "#16324f"),
                      ("clusterBkg", "#08182a"),
                      ("clusterBorder", "#2d5a86"),
                      ("lineColor", "#98a6b5"),
                      ("arrowheadColor", "#98a6b5")]:
        found = re.search(rf"{key}:\s*'([^']+)'", cfg)
        r.check(found is not None and found.group(1) == want,
                f"config {key} is {want}",
                f"got {found.group(1) if found else 'MISSING'} — "
                "CONTRIBUTING.md publishes this value, so a drift here makes "
                "the documented palette wrong")

    r.check(re.search(r"primaryTextColor:\s*'#ffffff'", cfg) is not None,
            "node text is white")
    r.check(re.search(r"darkMode:\s*true", cfg) is not None,
            "mermaid is in dark mode")

    # The site is dark-only, so the theme toggle must stay disabled — a diagram
    # tuned for black renders badly on white.
    r.check(re.search(r"disableSwitch:\s*true", cfg) is not None,
            "the light/dark toggle is disabled (the palette assumes dark)")


def test_diagrams_use_the_palette(r: Results) -> None:
    """Every diagram declares the house classDefs, with the right hexes."""
    with_mermaid = [p for p in pages() if mermaid_blocks(p.read_text())]
    r.check(len(with_mermaid) > 0,
            f"found {len(with_mermaid)} pages with mermaid diagrams")

    missing, wrong_hex = [], []
    for p in with_mermaid:
        text = p.read_text()
        rel = p.relative_to(REPO_ROOT)

        if "classDef" not in text:
            missing.append(str(rel))
            continue

        for name, (fill, stroke) in PALETTE.items():
            for got_fill, got_stroke in re.findall(
                rf"classDef\s+{name}\s+fill:(#[0-9a-fA-F]+),stroke:(#[0-9a-fA-F]+)",
                text,
            ):
                if got_fill.lower() != fill or got_stroke.lower() != stroke:
                    wrong_hex.append(
                        f"{rel}: classDef {name} is {got_fill}/{got_stroke}, "
                        f"house is {fill}/{stroke}"
                    )

    r.check(not missing,
            "every page with a diagram declares the house classDefs",
            "; ".join(missing[:5]))
    r.check(not wrong_hex,
            "no page redefines a house class with off-palette colours",
            "; ".join(wrong_hex[:5]))


def test_no_inline_theme_overrides(r: Results) -> None:
    """
    No diagram may override the global theme.

    An inline `%%{init}%%` wins over the config, so the diagram stops tracking
    the house theme the moment the config changes — and nothing warns.
    """
    offenders, layout_offenders = [], []
    for p in pages():
        text = p.read_text()
        for block in mermaid_blocks(text):
            if "%%{init" in block or "%%{ init" in block:
                offenders.append(str(p.relative_to(REPO_ROOT)))
            if re.search(r"^\s*layout:\s*elk", block, re.M):
                layout_offenders.append(str(p.relative_to(REPO_ROOT)))

    r.check(not offenders,
            "no diagram carries an inline %%{init}%% theme override",
            "; ".join(sorted(set(offenders))[:5]))
    r.check(not layout_offenders,
            "no diagram declares `layout: elk` (it is global)",
            "; ".join(sorted(set(layout_offenders))[:5]))


def test_diagram_hygiene(r: Results) -> None:
    """Label quoting — the failure mode that breaks the build cryptically."""
    unquoted = []
    for p in pages():
        for block in mermaid_blocks(p.read_text()):
            # A node label containing punctuation MUST be quoted. Catch the
            # bracket form with bare parens/commas inside.
            for m in re.finditer(r"^\s*\w+\[(?!\")([^\]\"]*[(),][^\]\"]*)\]",
                                 block, re.M):
                unquoted.append(
                    f"{p.relative_to(REPO_ROOT)}: {m.group(1)[:40]}")

    r.check(not unquoted,
            "node labels containing punctuation are quoted",
            "; ".join(unquoted[:5])
            + " — unquoted punctuation breaks the mermaid parser and the "
              "build error names the page, not the line")


def test_page_structure(r: Results) -> None:
    """Frontmatter and sidebar registration — orphaned pages are silent."""
    sidebars = SIDEBARS.read_text()
    listed = set(re.findall(r"'([a-z0-9/-]+)'", sidebars))

    no_frontmatter, no_position, orphans = [], [], []
    for p in pages():
        text = p.read_text()
        rel = str(p.relative_to(DOCS)).replace(".md", "")
        if not text.startswith("---"):
            no_frontmatter.append(rel)
        elif "sidebar_position:" not in text.split("---")[1]:
            no_position.append(rel)
        if rel not in listed:
            orphans.append(rel)

    r.check(not no_frontmatter, "every page opens with frontmatter",
            "; ".join(no_frontmatter[:5]))
    r.check(not no_position, "every page sets sidebar_position",
            "; ".join(no_position[:5]))
    r.check(not orphans,
            "every page is listed in sidebars.js",
            "; ".join(orphans[:5])
            + " — a page missing from sidebars.js is ORPHANED and nothing "
              "in the build warns you")

    # Duplicate positions inside one directory make ordering arbitrary.
    from collections import defaultdict
    seen = defaultdict(list)
    for p in pages():
        m = re.search(r"sidebar_position:\s*(\d+)", p.read_text())
        if m:
            seen[(p.parent, m.group(1))].append(p.name)
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    r.check(not dups, "no duplicate sidebar_position within a directory",
            "; ".join(f"{k[0].name}/{k[1]}: {v}" for k, v in list(dups.items())[:3]))


def test_referenced_tests_exist(r: Results) -> None:
    """
    Every `tests/test_*.py` named in code or docs must actually exist.

    Merging or renaming a suite leaves docstrings pointing at a file that is
    gone, and a reader following the instruction gets "no such file" — the
    kind of rot nothing else catches. Four of these appeared the moment two
    modules were merged into one suite.
    """
    import re

    have = {p.name for p in (REPO_ROOT / "tests").glob("test_*.py")}
    # Placeholders inside contributor-facing templates are illustrative, not
    # references to real files.
    PLACEHOLDERS = {"test_my_topic.py", "test_your_topic.py"}

    bad = []
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in {".py", ".md", ".sh", ".yml"}:
            continue
        if {"node_modules", "build", ".venv", ".git"} & set(p.parts):
            continue
        for m in re.findall(r"tests/(test_\w+\.py)",
                            p.read_text(errors="ignore")):
            if m not in have and m not in PLACEHOLDERS:
                bad.append(f"{p.relative_to(REPO_ROOT)} -> tests/{m}")

    r.check(not bad,
            "every referenced tests/test_*.py exists",
            "; ".join(sorted(set(bad))[:5]))

    # And the guard must not be vacuous — the suites it protects must be real.
    r.check(len(have) >= 10,
            f"the test directory is populated ({len(have)} suites)",
            "if this were near zero the check above would pass trivially")


def main() -> int:
    r = Results("Docs site style — mermaid house theme and page structure")
    test_global_theme(r)
    test_diagrams_use_the_palette(r)
    test_no_inline_theme_overrides(r)
    test_diagram_hygiene(r)
    test_page_structure(r)
    test_referenced_tests_exist(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
