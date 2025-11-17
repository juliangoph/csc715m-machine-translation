"""
Fast batched translation script for Hugging Face Seq2Seq models (e.g. NLLB).

✅ Features
- GPU auto-detection (CUDA if available)
- Mixed precision (bfloat16 / autocast)
- Batch translation for speed
- Explicit language tags (e.g., '<<ceb_Latn>>' → Tagalog)
- Compatible with transformers ≥ 4.33 and ≤ 4.46

Usage:
    python src/decode/translate_simple.py \
        --model facebook/nllb-200-distilled-600M \
        --src data/test.src \
        --out data/hyp.txt \
        --src_code ceb_Latn \
        --tgt_code tgl_Latn \
        --batch_size 12 \
        --num_beams 2 \
        --max_new_tokens 80
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

torch.set_float32_matmul_precision("high")  # enable TF32 for RTX GPUs

def resolve_lang_token_id(tokenizer, code: str):
    """
    Return (token_id, token_string) for a language code across tokenizer variants.
    Tries multiple conventions used by NLLB/M2M tokenizers.
    """
    candidates = [f"<<{code}>>", f"__{code}__", code]
    for cand in candidates:
        tid = tokenizer.convert_tokens_to_ids(cand)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid, cand
    return None, None

def chunked(seq, n):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
    parser = argparse.ArgumentParser(description="Batch translation with HF Seq2Seq models")
    parser.add_argument("--model", required=True, help="Model name (e.g., facebook/nllb-200-distilled-600M)")
    parser.add_argument("--src", required=True, help="Path to source text file (one sentence per line)")
    parser.add_argument("--out", required=True, help="Output file for translations")
    parser.add_argument("--src_code", default="tgl_Latn", help="Source language code (default: ceb_Latn)")
    parser.add_argument("--tgt_code", default="ceb_Latn", help="Target language code (default: tgl_Latn)")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size (reduce if VRAM is low)")
    parser.add_argument("--num_beams", type=int, default=2, help="Beam search width (1 = greedy)")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Max tokens to generate per sentence")
    parser.add_argument("--limit", type=int, default=None, help="Translate only first N lines (for testing)")
    args = parser.parse_args()

    # ---------- Load data ----------
    src_path = Path(args.src)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [l.strip() for l in src_path.open(encoding="utf-8") if l.strip()]
    if args.limit:
        lines = lines[: args.limit]
    if not lines:
        print("⚠️  No input lines found.")
        return
    print(f"Loaded {len(lines):,} lines from {src_path}")

    # ---------- Load model/tokenizer ----------
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # resolve language tokens across tokenizer variants (fast/slow)
    if hasattr(tokenizer, "lang_code_to_id"):
        # Preferred path (slow tokenizer exposes this)
        if args.tgt_code not in tokenizer.lang_code_to_id:
            raise ValueError(f"Unknown target code: {args.tgt_code}")
        forced_bos = tokenizer.lang_code_to_id[args.tgt_code]
        # For encoder, we still prefix an explicit tag to be safe
        src_tag_text = f"<<{args.src_code}>>"
    else:
        # Fallback for fast tokenizer or older transformers
        forced_bos, tgt_tag_text = resolve_lang_token_id(tokenizer, args.tgt_code)
        if forced_bos is None:
            raise ValueError(f"Cannot resolve target code {args.tgt_code} to a token id. "
                            "Update transformers & sentencepiece, or check the code spelling.")
        src_id, src_tag_text = resolve_lang_token_id(tokenizer, args.src_code)
        if src_id is None:
            raise ValueError(f"Cannot resolve source code {args.src_code} to a token id. "
                            "Update transformers & sentencepiece, or check the code spelling.")

    print(f"Using source={args.src_code} tag='{src_tag_text}' → target={args.tgt_code} (id={forced_bos})")
    print(f"Device: {device} | batch={args.batch_size} | beams={args.num_beams}")

    # ---------- Translation loop ----------
    preds = []
    model.eval()

    with torch.no_grad():
        for batch in chunked(lines, args.batch_size):
            # prepend source language tag for every sentence
            tagged_batch = [f"{src_tag_text} {line}" for line in batch]

            enc = tokenizer(
                tagged_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                gen = model.generate(
                    **enc,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    forced_bos_token_id=forced_bos,
                )

            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            preds.extend([t.strip() for t in texts])

    # ---------- Write output ----------
    with out_path.open("w", encoding="utf-8") as f:
        for t in preds:
            f.write(t + "\n")

    print(f"✅ Wrote {len(preds)} translations → {out_path}")


if __name__ == "__main__":
    main()
