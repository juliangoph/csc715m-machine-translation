"""
Clean and split parallel text files into train/dev/test TSVs.

Usage:
    python src/prepare/make_splits.py \
        --src data/raw/source/source.txt \
        --tgt data/raw/target/target.txt \
        --out data/processed
"""

import argparse
import random
import re
from pathlib import Path


def normalize(text: str) -> str:
    """Basic text cleaning."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    text = re.sub(r"\u00A0", " ", text)  # non-breaking space
    text = re.sub(r"\t+", " ", text)
    return text


def read_lines(path: Path):
    """Read non-empty lines."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = normalize(line)
            if not line:
                continue
            yield line


def main(args):
    src_path = Path(args.src)
    tgt_path = Path(args.tgt)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_lines = list(read_lines(src_path))
    tgt_lines = list(read_lines(tgt_path))

    n = min(len(src_lines), len(tgt_lines))
    pairs = list(zip(src_lines[:n], tgt_lines[:n]))

    # Filter too short or long
    filtered = [
        (s, t)
        for s, t in pairs
        if 2 <= len(s.split()) <= 100 and 2 <= len(t.split()) <= 100
    ]

    random.seed(42)
    random.shuffle(filtered)

    total = len(filtered)
    n_train = int(total * 0.8)
    n_dev = int(total * 0.1)
    n_test = total - n_train - n_dev

    splits = {
        "train.tsv": filtered[:n_train],
        "dev.tsv": filtered[n_train:n_train + n_dev],
        "test.tsv": filtered[n_train + n_dev:],
    }

    for name, data in splits.items():
        out_path = out_dir / name
        with open(out_path, "w", encoding="utf-8") as f:
            for s, t in data:
                f.write(f"{s}\t{t}\n")
        print(f"Wrote {len(data):6d} pairs → {out_path}")

    print(f"\nTotal usable pairs: {total:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split parallel data into train/dev/test TSVs")
    ap.add_argument("--src", required=True, help="Path to source.txt")
    ap.add_argument("--tgt", required=True, help="Path to target.txt")
    ap.add_argument("--out", required=True, help="Output directory for TSVs")
    args = ap.parse_args()
    main(args)
