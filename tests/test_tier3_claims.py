#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Labs that publish a comparative result must declare a baseline and a budget.

SCOPE, WHICH IS MOST OF THE POINT
---------------------------------
This applies to ONE tier of lab: those claiming that something is better
than something else on a measured axis. Three tiers exist here and only the
third needs any of this:

  1. no efficacy claim -- `01_neuralnet` fits y = 2x + 1; `analyze_kimi_k3`
     reads a config. Demanding a baseline is cargo cult, the same objection
     as a distributed launcher with nothing to distribute.
  2. a systems claim -- memory, throughput, tokens per page. Budget and
     configuration matter; a "baseline" often does not.
  3. a comparative claim -- "listwise beats pointwise", "load balancing
     makes it worse", "these five OCR models rank thus". THIS tier.

Tier 3 is where you can be confidently wrong, and the repository has been
twice: `11_moe`'s load-balancing finding reversed at world size 2, and
`03_learning_to_rank`'s spread between objectives is 0.041 at one epoch and
0.001 at forty. Both were published before being caught, and both are
scoping failures rather than measurement failures.

WHY AN EXPLICIT MARKER RATHER THAN A KEYWORD SEARCH
---------------------------------------------------
The first version of this check grepped for the word "baseline" and flagged
`05_video_speech/04_omni_eval`, which has had a proper control all along --
model B ignores the video entirely and the harness must catch it -- but
never uses that word. Meanwhile a lab could pass by writing "baseline" in a
sentence about something else.

A substring is not a fact about the document. So tier-3 READMEs declare it
in a fixed, visible form that a reader benefits from too:

    **Baseline:** <what the comparison is against>
    **Budget:** <epochs / steps / repeats / seeds the numbers came from>

Visible, not an HTML comment, because the reader is the point. The checker
only confirms the declaration exists -- whether the baseline is SENSIBLE is
a human judgement no checker can make, and pretending otherwise would be
the "three green checkers, one broken lab" failure again.

The falsifier is deliberately NOT enforced. It is a one-line prose
convention (`03_llms/12_prefill_decode` has one, and it earned its place by
failing on the first dry run), but any check for it would be a check for a
phrase.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _srcload import Results  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Explicit, because tier membership is a judgement about what a lab CLAIMS,
# not something inferable from its files. Adding a lab here is how you opt
# into the requirement; the reason is recorded so the list can be argued
# with rather than merely obeyed.
TIER3 = {
    "01_basics/03_convnet_cifar10": "ranks three architectures by accuracy",
    "02_intermediate/03_learning_to_rank": "ranks four objectives by NDCG",
    "02_intermediate/04_groupwise_ranking": "ranks three architectures",
    "03_llms/03_ocr": "ranks five OCR models by accuracy and tokens",
    "03_llms/11_moe": "claims load balancing makes the model worse",
    "03_llms/12_prefill_decode": "ranks three scheduling policies",
    "04_video_text/05_video_eval": "claims compression costs temporal accuracy",
    "05_video_speech/04_omni_eval": "claims a model does or does not fuse",
    "07_physical_ai/01_obstacle_hopper": "compares a height-blind policy against a seeing one",
    "07_physical_ai/02_biped_stairs": "compares four morphologies on one staircase",
    "07_physical_ai/03_terrain_vision": "claims a depth camera beats proprioception",
}

BASELINE = re.compile(r"\*\*Baseline:\*\*\s*\S", re.I)
BUDGET = re.compile(r"\*\*Budget:\*\*\s*\S", re.I)


def test_tier3_labs_declare_a_baseline(r: Results) -> None:
    """
    Every comparative claim needs something to be compared against.

    The argument is `04_video_text/05_video_eval`, whose eval harness once
    scored a RANDOM baseline at 100% because the RNG was correlated with the
    answer key. Nothing in the numbers looked wrong. The only thing that
    could have revealed it was a baseline, and the only reason it was caught
    is that someone ran one.
    """
    for lab, why in sorted(TIER3.items()):
        readme = REPO / lab / "README.md"
        if not readme.exists():
            r.check(False, f"{lab}: README exists")
            continue
        r.check(bool(BASELINE.search(readme.read_text())),
                f"{lab}: declares a baseline",
                f"it {why}, so add a line '**Baseline:** <what>' naming what "
                f"the comparison is against")


def test_tier3_labs_declare_a_budget(r: Results) -> None:
    """
    A measured number without its budget is not reproducible or comparable.

    `02_intermediate/03_learning_to_rank` publishes a spread between
    objectives of 0.041 at one epoch and 0.001 at forty. Either number
    alone, stated as "listwise beats pointwise by X", would be misleading.
    The budget is not metadata about the result; it is part of the result.
    """
    for lab, why in sorted(TIER3.items()):
        readme = REPO / lab / "README.md"
        if not readme.exists():
            continue
        r.check(bool(BUDGET.search(readme.read_text())),
                f"{lab}: declares a budget",
                "add '**Budget:** <epochs / steps / repeats / seeds>' -- the "
                "configuration a number came from is part of the number")


def test_the_registry_is_honest(r: Results) -> None:
    """
    Every lab named here must exist, and the list must not be empty.

    A registry that silently points at a renamed folder checks nothing, and
    an empty one passes trivially -- the shape of several checks this
    repository has shipped unable to fail.
    """
    missing = [lab for lab in TIER3 if not (REPO / lab).is_dir()]
    r.check(not missing, f"every tier-3 lab exists ({len(TIER3)} labs)",
            f"renamed or deleted: {missing}")
    r.check(len(TIER3) >= 5,
            "the tier-3 registry is populated",
            "an empty registry makes every check above vacuous")


def main() -> int:
    r = Results("Tier-3 labs declare a baseline and a budget")
    test_the_registry_is_honest(r)
    test_tier3_labs_declare_a_baseline(r)
    test_tier3_labs_declare_a_budget(r)
    return r.finish()


if __name__ == "__main__":
    raise SystemExit(main())
