"""
Writes -> ../data/mono/target/mono.txt
"""

import re, random
from pathlib import Path
import pandas as pd

# ---- settings ----
ROOT = Path(__file__).resolve().parents[2]  # go up from src/data to project root
OUT_FILE = ROOT / "data/mono/target/mono.txt"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

SRC_TSVS = [
    ROOT / "data/processed/train.tsv",
    ROOT / "data/processed/dev.tsv",
]

SEED = 42
LIMIT = 20000
MIN_LEN = 6
MAX_LEN = 240

def clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\u200b\u2060\uFEFF]", "", s)
    return s

def sentences_from_tgt(paths):
    for p in paths:
        if not p.exists():
            print(f"⚠️ Missing: {p}")
            continue
        df = pd.read_csv(p, sep="\t", header=None, names=["src", "tgt"])
        for t in df["tgt"].astype(str):
            t = clean_text(t)
            if not t:
                continue
            parts = re.split(r"(?<=[\.!\?])\s+", t) if len(t) > MAX_LEN else [t]
            for sent in parts:
                sent = clean_text(sent)
                if MIN_LEN <= len(sent) <= MAX_LEN:
                    yield sent

# collect → dedupe → shuffle → limit
pool = list(sentences_from_tgt(SRC_TSVS))
before = len(pool)
pool = list(dict.fromkeys(pool))
after_dedupe = len(pool)
random.Random(SEED).shuffle(pool)

if LIMIT:
    pool = pool[:LIMIT]

# write
with OUT_FILE.open("w", encoding="utf-8") as f:
    for s in pool:
        f.write(s + "\n")

print(f"✅ Wrote {len(pool):,} lines → {OUT_FILE}")
print(f"   Source lines: {before:,} | After dedupe: {after_dedupe:,} | After limit: {len(pool):,}")
print(f"   Length filter: {MIN_LEN}–{MAX_LEN} chars | Seed: {SEED}")
