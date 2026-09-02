# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
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


def test_suite_registration_is_complete(r: Results) -> None:
    """
    Every suite on disk must be in CLAUDE.md's count, run_all.sh, AND CI.

    Adding a suite changes three things and nothing notices if you miss one:
    `./tests/run_all.sh` silently becomes a PARTIAL run wearing the name of a
    full one, CI silently stops covering the new code, and CLAUDE.md
    advertises a stale number a future agent will trust.

    Deliberately cheap — globbing and string matching, no subprocesses. An
    earlier version of this check executed all sixteen suites to total their
    assertions, which made a docs-style test take ten minutes. A guard nobody
    will wait for is a guard that gets deleted. CLAUDE.md no longer quotes a
    check TOTAL for the same reason: it rots on every commit and the suite
    count carries the useful information.
    """
    import re

    on_disk = {t.name for t in (REPO_ROOT / "tests").glob("test_*.py")}
    r.check(len(on_disk) >= 10,
            f"the test directory is populated ({len(on_disk)} suites)",
            "if this were near zero the checks below would pass vacuously")

    listed = set(re.findall(r"tests/(test_\w+\.py)",
                            (REPO_ROOT / "tests" / "run_all.sh").read_text()))
    r.check(listed == on_disk,
            "run_all.sh lists every suite on disk",
            f"missing: {sorted(on_disk - listed)}; "
            f"stale: {sorted(listed - on_disk)}")

    in_ci = set(re.findall(r"tests/(test_\w+\.py)",
                           (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text()))
    r.check(in_ci == on_disk,
            "the CI workflow runs every suite on disk",
            f"missing from CI: {sorted(on_disk - in_ci)}")

    # ---- the CI workflow must be VALID YAML --------------------------------
    # A malformed workflow does not fail loudly -- GitHub refuses to run it, so
    # every suite silently stops executing while the repository looks fine
    # locally. This is not hypothetical: a scripted edit added a SECOND `run:`
    # key to an existing step, which is a duplicate mapping key, and six
    # consecutive pushes reported failure with no test output at all.
    import yaml

    class _NoDuplicates(yaml.SafeLoader):
        """
        SafeLoader that REJECTS duplicate mapping keys.

        Plain yaml.safe_load accepts them and keeps the last, so the exact bug
        this check exists for -- a second `run:` added to an existing step --
        parses cleanly and the guard passes while CI stays broken. Verified:
        the first version of this check used safe_load and did not catch the
        bug it was written for.
        """

    def _no_dup(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark)
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _NoDuplicates.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup)

    wf = REPO_ROOT / ".github" / "workflows" / "tests.yml"
    try:
        parsed = yaml.load(wf.read_text(), Loader=_NoDuplicates)
        ok, why = True, ""
    except Exception as exc:
        parsed, ok, why = None, False, str(exc).replace("\n", " ")[:200]
    r.check(ok, "the CI workflow is valid YAML",
            f"{why} -- a workflow that does not parse runs NOTHING, and the "
            "only symptom is a red check with an empty log")

    if parsed:
        jobs = parsed.get("jobs") or {}
        steps = [st for job in jobs.values() for st in (job.get("steps") or [])]
        r.check(len(steps) > 5, f"the workflow defines steps ({len(steps)})")
        # Each step that runs a suite must have a name, or a failure in CI is
        # reported against an anonymous step.
        unnamed = [st for st in steps
                   if "tests/test_" in str(st.get("run", "")) and not st.get("name")]
        r.check(not unnamed, "every suite step in CI has a name",
                f"{len(unnamed)} unnamed")

    claude = (REPO_ROOT / "CLAUDE.md").read_text()
    r.check(f"{len(on_disk)} suites" in claude,
            f"CLAUDE.md's suite count is current ({len(on_disk)})",
            "it has gone stale twice already — 11 suites, then 14")


def test_published_counts_are_current(r: Results) -> None:
    """
    Counts the docs publish about themselves must match reality.

    The root README advertises how many pages the site has, and CONTRIBUTING.md
    says the mermaid palette is "load-bearing across N pages". Both are the
    first thing a visitor reads and both rot silently every time a page is
    added -- the page count had drifted to 26 against an actual 39, and the
    SLURM guide claimed "all fourteen examples" when there were 23.

    Kept to counts that are cheap to derive and genuinely user-facing. This is
    deliberately NOT extended to a total assertion count, which rots on every
    commit and was removed from CLAUDE.md for exactly that reason.
    """
    import re

    pages = list(DOCS.rglob("*.md"))
    diagram_pages = [p for p in pages if mermaid_blocks(p.read_text())]

    readme = (REPO_ROOT / "README.md").read_text()
    m = re.search(r"(\d+) pages with", readme)
    r.check(m is not None and int(m.group(1)) == len(pages),
            f"README.md's page count is current ({len(pages)})",
            f"README says {m.group(1) if m else 'MISSING'}, site has {len(pages)}")

    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text()
    m = re.search(r"load-bearing across (\d+) pages", contributing)
    r.check(m is not None and int(m.group(1)) == len(diagram_pages),
            f"CONTRIBUTING.md's diagram-page count is current ({len(diagram_pages)})",
            f"CONTRIBUTING says {m.group(1) if m else 'MISSING'}, "
            f"{len(diagram_pages)} pages carry a mermaid block")


def test_readme_folder_tree(r: Results) -> None:
    """
    The README's folder tree must match the repository.

    It had drifted three separate ways before this check existed: six paths
    renamed by the section reorganisation, three examples added months earlier
    and never listed, and files that no longer existed. A tree that is wrong is
    worse than no tree, because it is the first thing a visitor reads and they
    have no reason to doubt it.

    Checked both directions -- nothing listed that is absent, nothing present
    that is unlisted -- because either alone passes on a tree that is half
    right.
    """
    import re

    readme = (REPO_ROOT / "README.md").read_text()
    if "## Folder Structure" not in readme:
        r.check(False, "README has a Folder Structure section")
        return

    start = readme.index("## Folder Structure")
    fence = readme.index("```", start)
    tree = readme[fence:readme.index("```", fence + 3)]

    # 1. every path the tree shows must exist
    shown = {m for m in re.findall(r"([0-9]{2}_[a-z_0-9]+)/", tree)}
    sections = {p.name for p in REPO_ROOT.glob("0[1-5]_*") if p.is_dir()}
    topics = {t.name for sec in REPO_ROOT.glob("0[1-5]_*") if sec.is_dir()
              for t in sec.iterdir() if t.is_dir() and t.name[:2].isdigit()}
    ghosts = sorted(shown - sections - topics)
    r.check(not ghosts,
            "every folder in the README tree exists",
            f"{ghosts} -- renamed or deleted without updating the tree")

    # 2. every example must appear
    absent = sorted(topics - shown)
    r.check(not absent,
            f"every example appears in the README tree ({len(topics)} examples)",
            f"missing: {absent}")

    # 3. and every section
    missing_secs = sorted(sections - shown)
    r.check(not missing_secs, "every section appears in the README tree",
            f"missing: {missing_secs}")


def main() -> int:
    r = Results("Docs site style — mermaid house theme and page structure")
    test_global_theme(r)
    test_diagrams_use_the_palette(r)
    test_no_inline_theme_overrides(r)
    test_diagram_hygiene(r)
    test_page_structure(r)
    test_referenced_tests_exist(r)
    test_suite_registration_is_complete(r)
    test_published_counts_are_current(r)
    test_readme_folder_tree(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
