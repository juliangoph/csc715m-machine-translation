"""
Compute BLEU and chrF2 metrics using sacrebleu.

Usage:
    python src/eval/score.py \
        --ref experiments/baseline/test.ref \
        --hyp experiments/baseline/hyp.txt \
        --out experiments/baseline/metrics.json
"""
import argparse
import json
from pathlib import Path
import sacrebleu

def read_lines(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [l.rstrip("\n") for l in f]

def main(args):
    ref_lines = [l for l in read_lines(Path(args.ref)) if l.strip()]
    hyp_lines = [l for l in read_lines(Path(args.hyp)) if l.strip()]

    if len(ref_lines) != len(hyp_lines):
        n = min(len(ref_lines), len(hyp_lines))
        print(f"⚠️  Length mismatch: ref={len(ref_lines)} hyp={len(hyp_lines)} — truncating to {n}")
        ref_lines = ref_lines[:n]
        hyp_lines = hyp_lines[:n]

    # Compute scores (works on sacrebleu>=2 and older)
    try:
        # Preferred API (explicit metric classes)
        from sacrebleu.metrics import BLEU, CHRF
        bleu_metric = BLEU()                   # defaults: tokenize='13a', use_effective_order=True
        chrf_metric = CHRF(word_order=2)       # chrF2
        bleu = bleu_metric.corpus_score(hyp_lines, [ref_lines])
        chrf = chrf_metric.corpus_score(hyp_lines, [ref_lines])

        # Try to capture a signature string if available
        try:
            signature = bleu_metric.get_signature().format()
        except Exception:
            signature = None
    except Exception:
        # Fallback to legacy convenience functions
        bleu = sacrebleu.corpus_bleu(hyp_lines, [ref_lines])
        chrf = sacrebleu.corpus_chrf(hyp_lines, [ref_lines])
        # Best-effort signature on older versions
        try:
            signature = bleu.signature.format()   # may not exist; ignore if it doesn't
        except Exception:
            signature = None

    metrics = {
        "BLEU": round(float(bleu.score), 2),
        "chrF2": round(float(chrf.score), 2),
        "ref_len": getattr(bleu, "ref_len", None),
        "sys_len": getattr(bleu, "sys_len", None),
        "signature": signature,
        "sacrebleu_version": getattr(sacrebleu, "__version__", "unknown"),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute BLEU and chrF2 using sacrebleu")
    ap.add_argument("--ref", required=True, help="Reference text file (one line per translation)")
    ap.add_argument("--hyp", required=True, help="Hypothesis text file (one line per translation)")
    ap.add_argument("--out", required=True, help="Output JSON path for metrics")
    args = ap.parse_args()
    main(args)
