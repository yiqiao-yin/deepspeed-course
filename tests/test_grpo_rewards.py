# /// script
# requires-python = ">=3.9"
# ///
"""
Regression test: multi-agent GRPO uses a verifiable, correctly-aligned reward.

Run:
    uv run tests/test_grpo_rewards.py

Background
----------
`07_huggingface_trl_multi_agency` had three related problems.

1. It loaded AutoModelForCausalLMWithValueHead. The value head is a CRITIC —
   a PPO construct. GRPO exists precisely to remove the critic, replacing the
   learned baseline V(s) with the group mean reward.

2. The reward was string similarity to a reference. That rewards SURFACE FORM,
   not correctness: '42' and '-42' are ~95% similar as strings and one is
   wrong, while a correct solution phrased differently scores poorly.

3. The similarity reward was MISALIGNED. It closed over the dataset's
   completion list and zipped it positionally against generated completions.
   GRPO samples G rollouts per prompt, so generation i does not correspond to
   dataset row i, and each rollout was scored against the wrong reference.

Pure stdlib — no dependencies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results, load_function, source_code_contains, source_contains  # noqa: E402

MAIN = "07_huggingface_trl_multi_agency/main.py"
MATH = "07_huggingface_trl_multi_agency/train_grpo_math.py"


def main() -> int:
    r = Results("Multi-agent GRPO — reward and model-class regression test")

    # ---- 1. The critic is gone ----------------------------------------
    for script in (MAIN, MATH):
        name = Path(script).name
        r.check(
            not source_code_contains(script, "AutoModelForCausalLMWithValueHead"),
            f"{name}: does not instantiate a value head",
            "GRPO has no critic; the value head is a PPO construct.",
        )
        r.check(
            source_contains(script, "AutoModelForCausalLM.from_pretrained"),
            f"{name}: uses a plain causal LM",
        )
        r.check(
            source_contains(script, "model=self.model,"),
            f"{name}: passes the LOADED model to GRPOTrainer, not the name",
            "Passing the name made GRPOTrainer load a second copy of the "
            "weights, leaving the model built in __init__ untrained.",
        )
        # Comment-aware: these files deliberately DESCRIBE the old bug in
        # prose, so a naive text search would match that explanation.
        r.check(
            not source_code_contains(script, "self.model.base_model"),
            f"{name}: no value-head-specific base_model access in executable code",
        )

    # ---- 2. Similarity reward is gone ---------------------------------
    r.check(
        not source_code_contains(MATH, "SequenceMatcher"),
        "train_grpo_math.py: string-similarity reward removed",
    )
    r.check(
        source_contains(MATH, "reward_funcs=reward_answer_correct"),
        "train_grpo_math.py: uses the verifiable reward",
    )
    r.check(
        source_contains(MAIN, "reward_funcs=reward_answer_correct"),
        "main.py: uses the verifiable reward, not reward_unique_chars",
    )

    # ---- 3. Answer extraction behaves ---------------------------------
    extract = load_function(MAIN, "extract_final_answer", extra_globals={"re": __import__("re")})

    cases = [
        ("Janet has 16 - 3 - 4 = 9 eggs. 9 * 2 = 18.\n#### 18", 18.0, "GSM8K #### convention"),
        ("The answer is 42", 42.0, "trailing number, no ####"),
        ("Total: 1,234 dollars", 1234.0, "comma-separated thousands"),
        ("Result is -7", -7.0, "negative number"),
        ("It costs 3.5", 3.5, "decimal"),
        ("#### 18 ", 18.0, "whitespace around the answer"),
        ("no numbers here", None, "returns None when no number present"),
        ("Step 1: 5 apples. Step 2: 3 more.\n#### 8", 8.0, "#### wins over earlier numbers"),
    ]
    for text, expected, label in cases:
        got = extract(text)
        r.check(got == expected, f"extract_final_answer: {label}", f"got {got!r}, expected {expected!r}")

    # ---- 4. The reward is genuinely verifiable ------------------------
    reward = load_function(
        MAIN, "reward_answer_correct", extra_globals={"extract_final_answer": extract}
    )

    completions = [
        "reasoning ...\n#### 18",     # correct
        "reasoning ...\n#### 15",     # wrong
        "a totally different phrasing that still concludes\n#### 18",  # correct, different wording
        "#### -18",                   # sign error -> wrong
    ]
    references = ["#### 18"] * 4
    got = reward(completions, completion=references)

    r.check(got == [1.0, 0.0, 1.0, 0.0], "binary reward on exact answer match", f"got {got}")
    r.check(
        got[2] == 1.0,
        "differently-worded correct answer still scores 1.0",
        "String similarity would have penalised this.",
    )
    r.check(
        got[3] == 0.0,
        "sign error scores 0.0",
        "'18' vs '-18' are ~95% similar as strings; similarity reward would "
        "have scored this near-perfect.",
    )

    # Alignment: G rollouts per prompt must each be scored against THEIR OWN
    # reference, which is what reading from kwargs guarantees.
    two_prompts = ["#### 5", "#### 5", "#### 9", "#### 9"]     # G=2, 2 prompts
    refs_expanded = ["#### 5", "#### 5", "#### 9", "#### 9"]
    r.check(
        reward(two_prompts, completion=refs_expanded) == [1.0, 1.0, 1.0, 1.0],
        "G rollouts per prompt align with their own references",
    )
    mixed = reward(["#### 5", "#### 9"], completion=["#### 9", "#### 5"])
    r.check(mixed == [0.0, 0.0], "misaligned answers correctly score 0.0", f"got {mixed}")

    # Missing references must fail loudly rather than score silently.
    try:
        reward(["#### 1"])
        r.check(False, "missing references raises", "Scored silently instead.")
    except ValueError:
        r.check(True, "missing references raises ValueError")

    # ---- 5. The dummy reward is retained but clearly labelled ---------
    r.check(
        source_contains(MAIN, "DUMMY reward"),
        "main.py: reward_unique_chars is explicitly labelled a dummy",
    )

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
