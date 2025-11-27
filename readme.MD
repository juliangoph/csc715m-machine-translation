# Cebuano → Tagalog Machine Translation Project (NLLB Fine-Tuning + Pivot + Back-Translation)

This project implements a full MT workflow using the **facebook/nllb-200-distilled-600M** model, fine-tuned to translate **Cebuano → Tagalog**.
 It includes:

- Data preparation
- Zero-shot baseline
- Fine-tuning
- Pivot translation
- Back-translation (BT)
- Evaluation
- Error analysis
- Human evaluation

All scripts were rewritten for clarity and compatibility across environments.

------

## **Project Structure**

```
project/
│
├── data/
│   ├── raw/                    # Original parallel text (source.txt, target.txt)
│   ├── processed/              # Cleaned TSV splits (train/dev/test + BT augment)
│   └── mono/
│       └── target/             # Monolingual Tagalog mined for back-translation
│
├── experiments/
│   ├── baseline/               # Zero-shot results
│   ├── finetune/               # Fine-tuned Cebuano→Tagalog model
│   ├── finetune_bt/            # Fine-tuned + Back-Translation model
│   └── pivot/                  # Pivot translation results
│
├── notebooks/
│   ├── 00_overview.ipynb
│   ├── 01_data_prep.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_finetune.ipynb
│   ├── 04_pivot_bt_eval.ipynb
│   ├── 05_error_analysis.ipynb
│   └── 06_report.ipynb         # Final PDF-ready report
│
└── src/
    ├── prepare/
    │   ├── convert_tmx.py
    │   ├── make_splits.py
    │   └── mine_mono_tagalog.py
    │
    ├── decode/
    │   └── translate_simple.py
    │
    ├── train/
    │   └── finetune.py         # Main fine-tuning script
    │
    └── eval/
        └── evaluate.py         # Unified evaluation script (BLEU, chrF, hypotheses)
```

------

## **Language Setup**

- **Source language → Cebuano (`ceb_Latn`)**
- **Target language → Tagalog (`tgl_Latn`)**

These codes are used consistently across training, decoding, and evaluation.

------

## **Notebooks Overview**

### **00_overview.ipynb**

Defines global project settings:

- Language direction (Cebuano → Tagalog)
- Model
- Seed
- Experiment folders
- High-level workflow map

------

### **01_data_prep.ipynb**

Creates train/dev/test splits:

- Cleans raw source/target files
- Builds `train.tsv`, `dev.tsv`, `test.tsv`
- Shows random samples for sanity checking

Uses `make_splits.py`.

------

### **02_baseline.ipynb**

Runs **zero-shot** NLLB translation:

- Extracts test.src and test.ref
- Calls `translate_simple.py` for NLLB
- Computes baseline BLEU/chrF

Output: `experiments/baseline/metrics.json`

------

### **03_finetune.ipynb**

Fine-tunes NLLB for **Cebuano → Tagalog**:

- Loads TSV datasets
- Applies language tags
- Calls the CLI script:

```
!python ../src/train/finetune.py \
  --train ../data/processed/train.tsv \
  --dev ../data/processed/dev.tsv \
  --out ../experiments/finetune
```

------

### **04_pivot_bt_eval.ipynb**

Two tasks:

#### **A) Pivot Translation**

- Cebuano → Waray → Tagalog
- Scored with BLEU/chrF

#### **B) Back-Translation (BT)**

1. Mined Cebuano monolingual text → `mine_mono_tagalog.py`
2. Cebuano → Tagalog synthetic back-translations
3. Create:

```
train_plus_bt.tsv = train.tsv + synthetic BT pairs
```

1. Re-train:

```
!python ../src/train/finetune.py \
  --train ../data/processed/train_plus_bt.tsv \
  --dev ../data/processed/dev.tsv \
  --out ../experiments/finetune_bt
```

------

### **05_error_analysis.ipynb**

Compares:

- baseline
- finetune
- pivot
- finetune_bt

Does:

- Sentence-level similarity (difflib)
- Top improvements vs regressions
- Negation checks
- Numerical handling
- Structure stability

------

### **06_report.ipynb**

A clean, PDF-ready compilation:

- Motivation
- Methodology
- Model setup
- Pivot & BT results
- Human evaluation (50-sentence set)
- Conclusions & future work

------

## **📂 Scripts Overview**

### **prepare/make_splits.py**

Creates train/dev/test TSV files with filtering:

- Removes duplicates
- Removes very short / very long lines
- Shuffles deterministically

------

### **prepare/mine_mono_tagalog.py**

Extracts monolingual Cebuano from existing target training data.
 Used for back-translation.

------

### **decode/translate_simple.py**

Lightweight translation script for:

- Baseline
- Pivot
- Back-translation synthetic generation

Automatically prepends the source language code.

------

### **train/finetune.py**

**Main training script**
 Features:

- Automatic language token resolution
- bf16/fp16 automatic mixed precision
- Fully compatible CLI
- Clean training pipeline

Example:

```
python src/train/finetune.py \
  --train data/processed/train.tsv \
  --dev data/processed/dev.tsv \
  --out experiments/finetune \
  --code_src ceb_Latn --code_tgt tgl_Latn
```

------

### **eval/evaluate.py**

Unified evaluator:

- Loads model directory
- Translates test set in batches
- Computes BLEU, chrF (SacreBLEU 2.x)
- Saves hypotheses
- Outputs metrics JSON

Example:

```
python src/eval/evaluate.py \
  --model_dir experiments/finetune_bt \
  --test_tsv data/processed/test.tsv \
  --out_json experiments/finetune_bt/metrics.json \
  --save_hyp experiments/finetune_bt/hyp.txt \
  --code_src ceb_Latn \
  --code_tgt tgl_Latn
```

