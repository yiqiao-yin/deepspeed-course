# /// script
# requires-python = ">=3.9"
# ///
"""
Audit every example README against the code it documents.

    uv run scripts/audit_readmes.py

Reports likely drift: referenced files that do not exist, quoted config values
that disagree with the real ds_config.json, and documented symbols absent from
the neighbouring source.

ADVISORY, NOT A GATE. This over-reports by design and needs human triage — a
JSON snippet under "CUDA Out of Memory -> reduce batch size" is remediation
advice, not a claim about the shipped config, and an illustrative `def` block is
not a claim that a function exists. On its first run it flagged 31 candidates of
which 2 were real. Read the findings; do not auto-fix them.
"""
import ast
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
findings = []


def add(readme, kind, msg):
    findings.append((str(readme.relative_to(ROOT)), kind, msg))


readmes = [p for p in ROOT.rglob("README.md")
           if "docusaurus-docs" not in p.parts and ".git" not in p.parts
           and "node_modules" not in p.parts and p != ROOT / "README.md"
           and "tests" not in p.parts]

for rm in sorted(readmes):
    text = rm.read_text(encoding="utf-8", errors="ignore")
    folder = rm.parent

    # ---- 1. referenced .py / .sh / .json files that do not exist ---------
    for m in re.finditer(r"[`\"']([a-zA-Z0-9_./-]+\.(?:py|sh|json))[`\"']", text):
        name = m.group(1)
        if name.startswith(("http", "/")) or "*" in name:
            continue
        base = name.split("/")[-1]
        # search the folder tree and one level up
        hits = list(folder.rglob(base)) or list(folder.parent.rglob(base))
        if not hits and base not in {"pyproject.toml", "uv.lock", "requirements.txt",
                                     "ds_config.json", "config.json", "run_all.sh",
                                     "train.py", "script.py"}:
            add(rm, "missing-file", f"references {name!r} — not found")

    # Strip sections that are explicitly REMEDIATION advice rather than claims
    # about the shipped config — the main source of false positives.
    remediation = re.compile(
        r"(?is)(out of memory|reduce batch|if you change|troubleshoot|slow training"
        r"|modifying the config|down from)")
    scrubbed = "\n".join(
        blk for blk in re.split(r"\n(?=#{2,4} )", text) if not remediation.search(blk))

    # ---- 2. quoted JSON config values vs the real ds_config.json ---------
    cfgs = list(folder.glob("ds_config*.json")) + list(folder.glob("*_config.json"))
    if cfgs:
        real = {}
        for c in cfgs:
            try:
                real[c.name] = json.loads(c.read_text())
            except json.JSONDecodeError:
                add(rm, "bad-json", f"{c.name} does not parse")
        for c, cfg in real.items():
            # compare a few scalar keys the READMEs commonly quote
            for key in ("train_batch_size", "train_micro_batch_size_per_gpu",
                        "gradient_accumulation_steps", "gradient_clipping"):
                if key not in cfg:
                    continue
                actual = cfg[key]
                for m in re.finditer(rf'"{key}"\s*:\s*([0-9.e+-]+)', scrubbed):
                    quoted = m.group(1)
                    try:
                        q = float(quoted)
                    except ValueError:
                        continue
                    if isinstance(actual, (int, float)) and abs(q - float(actual)) > 1e-9:
                        add(rm, "config-drift",
                            f'quotes "{key}": {quoted} but {c} has {actual}')
                        break
            # optimizer type
            opt = cfg.get("optimizer", {}).get("type")
            if opt:
                for m in re.finditer(r'"type"\s*:\s*"(\w+)"', text):
                    if m.group(1) != opt and m.group(1) in {"Adam", "AdamW", "SGD", "Lamb"}:
                        add(rm, "config-drift",
                            f'quotes optimizer "{m.group(1)}" but {c} uses "{opt}"')
                        break
            # fp16 / bf16 enabled
            for prec in ("fp16", "bf16"):
                if prec in cfg and isinstance(cfg[prec].get("enabled"), bool):
                    actual_on = cfg[prec]["enabled"]
                    m = re.search(rf'"{prec}"\s*:\s*{{\s*"enabled"\s*:\s*(true|false)', text)
                    if m and (m.group(1) == "true") != actual_on:
                        add(rm, "config-drift",
                            f'shows {prec}.enabled={m.group(1)} but {c} has {str(actual_on).lower()}')

    # ---- 3. referenced symbols that no longer exist in sibling .py -------
    pys = [p for p in folder.rglob("*.py") if "__pycache__" not in p.parts]
    symbols = set()
    for p in pys:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    symbols.add(a.name.split(".")[-1])
    if pys:
        # symbols the README shows being *called* or *constructed*
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9]{4,})\.from_pretrained\(", text):
            if m.group(1) not in symbols:
                add(rm, "stale-symbol", f"shows {m.group(1)}.from_pretrained(...) — not in code")
        for m in re.finditer(r"\bdef ([a-z_][a-z0-9_]{3,})\(", text):
            if m.group(1) not in symbols:
                add(rm, "stale-symbol", f"documents def {m.group(1)}(...) — not in code")

print(f"Audited {len(readmes)} example READMEs\n")
if not findings:
    print("  no drift detected")
else:
    cur = None
    for f, kind, msg in findings:
        if f != cur:
            print(f"\n{f}")
            cur = f
        print(f"  [{kind}] {msg}")
    print(f"\n{len(findings)} finding(s)")
