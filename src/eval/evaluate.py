"""
Evaluate a fine-tuned NLLB translation model.

Loads a trained model and tokenizer, translates the test set, and computes
BLEU and chrF metrics using SacreBLEU. Supports CUDA acceleration and
language tag resolution for consistency with the training setup.

Usage:
    python src/eval/evaluate_finetune.py \
        --test data/processed/test.tsv \
        --model_dir experiments/finetune

Outputs:
    - metrics.json (BLEU, chrF2, system/reference lengths)
    - Console summary of metric scores
"""

import os, json, argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def resolve_lang_token_id(tokenizer, code: str):
    # Preferred: direct NLLB table
    if hasattr(tokenizer, "lang_code_to_id") and isinstance(tokenizer.lang_code_to_id, dict):
        if code in tokenizer.lang_code_to_id:
            return tokenizer.lang_code_to_id[code], code
    # Fallback: try common spellings
    for cand in (f"<<{code}>>", f"__{code}__", code):
        tid = tokenizer.convert_tokens_to_ids(cand)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid, cand
    return None, None

def main():
    ap = argparse.ArgumentParser(description="Evaluate a fine-tuned NLLB model on test.tsv")
    ap.add_argument("--model_dir", required=True, help="Path to saved model folder (e.g., ../experiments/finetune)")
    ap.add_argument("--test_tsv", required=True, help="Path to test.tsv (src\\ttgt)")
    ap.add_argument("--out_json", required=True, help="Where to write metrics JSON")
    ap.add_argument("--save_hyp", default=None, help="Optional path to write hypotheses (one per line)")
    ap.add_argument("--code_src", default="tgl_Latn", help="Source language code")
    ap.add_argument("--code_tgt", default="ceb_Latn", help="Target language code")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_src_len", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--num_beams", type=int, default=5)
    args = ap.parse_args()

    test_df = pd.read_csv(args.test_tsv, sep="\t", header=None, names=["src", "tgt"])
    src_texts = test_df["src"].astype(str).tolist()
    refs = test_df["tgt"].astype(str).tolist()

    # Load model/tokenizer from finetuned folder
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    # dtype hint (safe on CPU)
    dtype = None
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, torch_dtype=dtype)

    # Resolve NLLB tags and enforce decoder BOS
    src_id, src_tag_text = resolve_lang_token_id(tokenizer, args.code_src)
    tgt_id, tgt_tag_text = resolve_lang_token_id(tokenizer, args.code_tgt)
    if src_id is None or tgt_id is None:
        maybe = []
        if hasattr(tokenizer, "lang_code_to_id") and isinstance(tokenizer.lang_code_to_id, dict):
            maybe = sorted(list(tokenizer.lang_code_to_id.keys()))[:30]
        raise ValueError(f"Could not resolve language tokens (src:{args.code_src}, tgt:{args.code_tgt}). "
                         f"Check codes/transformers install. Some available codes: {maybe}")
    model.config.forced_bos_token_id = tgt_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    hyps = []
    with torch.no_grad():
        for batch in tqdm(list(batches(src_texts, args.batch_size)), desc="Translating"):
            # Prefix encoder inputs with source lang tag
            prefixed = [f"{src_tag_text} {s}" for s in batch]
            enc = tokenizer(
                prefixed,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_src_len,
            ).to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                gen = model.generate(
                    **enc,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    forced_bos_token_id=tgt_id,
                )

            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            hyps.extend([t.strip() for t in texts])

    # Metrics (sacrebleu >= 2.0 – no .signature attr)
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    metrics = {
        "BLEU": round(bleu.score, 2),
        "chrF2": round(chrf.score, 2),
        "ref_len": bleu.ref_len,
        "sys_len": bleu.sys_len,
        "sacrebleu_version": sacrebleu.__version__,
        "n_samples": len(refs),
        "model_dir": str(args.model_dir),
        "codes": {"src": args.code_src, "tgt": args.code_tgt},
        "decoding": {"beams": args.num_beams, "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size},
    }

    Path(os.path.dirname(args.out_json) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if args.save_hyp:
        Path(os.path.dirname(args.save_hyp) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.save_hyp, "w", encoding="utf-8") as f:
            for h in hyps:
                f.write(h + "\n")

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
