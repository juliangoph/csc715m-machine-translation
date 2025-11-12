"""
Fine-tune an NLLB translation model on a parallel corpus.

Adapts a multilingual NLLB model (e.g., facebook/nllb-200-distilled-600M)
to a specific language pair (e.g., Cebuano→Tagalog). Supports automatic
language tag resolution, mixed precision (bf16/fp16), and optional 4-bit
quantization via bitsandbytes.

Usage:
    python src/train/finetune.py \
        --train data/processed/train.tsv \
        --dev data/processed/dev.tsv \
        --out experiments/finetune \
        --code_src ceb_Latn --code_tgt tgl_Latn

Outputs:
    - Fine-tuned model in the specified output directory
    - Training logs and checkpoint files
    - Summary of settings printed to console
"""

import os
import sys
import platform
import json
import random
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.utils import is_bitsandbytes_available


# --------------------
# Utilities
# --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_tsv_to_dataset(tsv_path: str):
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["src", "tgt"])
    return Dataset.from_pandas(df, preserve_index=False)


def resolve_lang_token_id(tokenizer, code: str):
    """
    Resolve an NLLB language token id across tokenizer variants.
    Returns (token_id, token_string).
    """
    # Preferred: NLLB exposes a code->id mapping
    if hasattr(tokenizer, "lang_code_to_id") and isinstance(tokenizer.lang_code_to_id, dict):
        if code in tokenizer.lang_code_to_id:
            return tokenizer.lang_code_to_id[code], code

    # Fallback: try common token spellings in the vocabulary
    for cand in (f"<<{code}>>", f"__{code}__", code):
        tid = tokenizer.convert_tokens_to_ids(cand)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid, cand

    return None, None


def build_preprocess_fn(tokenizer, src_tag_text: str, max_source_length: int, max_target_length: int):
    """
    Preprocessing function factory.
    Prefixes source with the resolved source language tag text,
    and tokenizes labels via text_target=...
    """
    def preprocess(batch):
        src_prefixed = [f"{src_tag_text} {s}" for s in batch["src"]]
        model_inputs = tokenizer(
            src_prefixed,
            truncation=True,
            max_length=max_source_length,
        )
        labels = tokenizer(
            text_target=batch["tgt"],
            truncation=True,
            max_length=max_target_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


# --------------------
# Main training routine
# --------------------
def finetune_model(
    train_path: str,
    dev_path: str,
    output_dir: str,
    model_name: str,
    code_src: str,
    code_tgt: str,
    seed: int = 42,
    max_source_length: int = 256,
    max_target_length: int = 256,
    learning_rate: float = 2e-5,
    train_bs: int = 8,
    eval_bs: int = 8,
    epochs: int = 2,
    use_4bit: bool = True,
):
    # Reproducibility
    set_seed(seed)

    # Prepare output
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ds_train = read_tsv_to_dataset(train_path)
    ds_dev = read_tsv_to_dataset(dev_path)

    # Tokenizer (use_fast=False is safer for NLLB code tables)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Optional 4-bit quantization (auto-disable on Windows or when unavailable)
    dtype = None
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    enable_4bit = use_4bit
    if platform.system().lower() == "windows":
        enable_4bit = False
    if enable_4bit and not is_bitsandbytes_available():
        enable_4bit = False

    bnb_args = {}
    device_map = None
    if enable_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype or torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            bnb_args["quantization_config"] = quant_config
            device_map = "auto"
            print("✅ Using bitsandbytes 4-bit quantization.")
        except Exception as e:
            print(f"⚠️  Disabling 4-bit (prep failed): {e}")
            enable_4bit = False
            bnb_args.clear()
            device_map = None

    # Model load with graceful fallback if 4-bit misbehaves
    def _load_model(_with_bnb: bool):
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,  # ok to be None on CPU
            device_map=("auto" if _with_bnb else None),
            **(bnb_args if _with_bnb else {})
        )

    try:
        model = _load_model(enable_4bit)
    except Exception as e:
        msg = str(e).lower()
        if "bitsandbytes" in msg or "quantization" in msg or "bnb" in msg:
            print(f"⚠️  Reloading without 4-bit due to error: {e}")
            model = _load_model(False)
        else:
            raise

    # Resolve language tokens
    src_id, src_tag_text = resolve_lang_token_id(tokenizer, code_src)
    tgt_id, tgt_tag_text = resolve_lang_token_id(tokenizer, code_tgt)

    if src_id is None or tgt_id is None:
        maybe = []
        if hasattr(tokenizer, "lang_code_to_id") and isinstance(tokenizer.lang_code_to_id, dict):
            maybe = sorted(list(tokenizer.lang_code_to_id.keys()))[:30]
        raise ValueError(
            f"Could not resolve language tokens (src:{code_src} id={src_id}, tgt:{code_tgt} id={tgt_id}). "
            f"Check your Transformers version or spelling. Some available codes: {maybe}"
        )

    # Let tokenizer know (some helpers use these)
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = code_src
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = code_tgt

    # Force decoder BOS to target language
    model.config.forced_bos_token_id = tgt_id

    print(f"Resolved codes → src: '{code_src}' as '{src_tag_text}' (id={src_id}) | "
          f"tgt: '{code_tgt}' as '{tgt_tag_text}' (id={tgt_id})")
    print(f"Train samples: {len(ds_train):,} | Dev samples: {len(ds_dev):,}")

    # Preprocess
    preprocess = build_preprocess_fn(tokenizer, src_tag_text, max_source_length, max_target_length)
    enc_train = ds_train.map(preprocess, batched=True, remove_columns=ds_train.column_names, desc="Tokenizing train")
    enc_dev = ds_dev.map(preprocess, batched=True, remove_columns=ds_dev.column_names, desc="Tokenizing dev")

    # Collator & precision flags
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_ok = torch.cuda.is_available() and not bf16_ok

    # Training args
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        bf16=bf16_ok,
        fp16=fp16_ok,
        # Uncomment if supported by your Transformers version:
        # save_strategy="epoch",
        # evaluation_strategy="epoch",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=enc_train,
        eval_dataset=enc_dev,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    train_output = trainer.train()
    trainer.save_model(str(out_dir))

    summary = {
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "output_dir": str(out_dir),
        "model_name": model_name,
        "code_src": code_src,
        "code_tgt": code_tgt,
        "epochs": epochs,
        "train_samples": len(enc_train),
        "eval_samples": len(enc_dev),
        "bf16": bool(bf16_ok),
        "fp16": bool(fp16_ok),
        "used_4bit": bool(enable_4bit),
    }

    print("\nTraining complete. Summary:")
    print(json.dumps(summary, indent=2))


# --------------------
# CLI
# --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune NLLB on a parallel dataset (TSV with src\\ttgt).")
    p.add_argument("--train", required=True, help="Path to train.tsv")
    p.add_argument("--dev", required=True, help="Path to dev.tsv")
    p.add_argument("--out", required=True, help="Output directory (e.g., ../experiments/finetune)")
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M", help="Base model name")
    p.add_argument("--code_src", default="ceb_Latn", help="Source language code (NLLB)")
    p.add_argument("--code_tgt", default="tgl_Latn", help="Target language code (NLLB)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_src_len", type=int, default=256)
    p.add_argument("--max_tgt_len", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train_bs", type=int, default=8)
    p.add_argument("--eval_bs", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--no_4bit", action="store_true", help="Disable bitsandbytes 4-bit quantization")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune_model(
        train_path=args.train,
        dev_path=args.dev,
        output_dir=args.out,
        model_name=args.model,
        code_src=args.code_src,
        code_tgt=args.code_tgt,
        seed=args.seed,
        max_source_length=args.max_src_len,
        max_target_length=args.max_tgt_len,
        learning_rate=args.lr,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        epochs=args.epochs,
        use_4bit=(not args.no_4bit),
    )
