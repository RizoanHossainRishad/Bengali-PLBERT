# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A self-contained package that pretrains a **Bengali PL-BERT** (phoneme-level ALBERT) on a Windows PC with one NVIDIA GPU, then exports a ~24 MB encoder checkpoint that drops into a StyleTTS2 fork on a Linux server in place of the English-only PL-BERT. `README.md` is the operating manual; `DESIGN.md` records the reasoning and measurements behind every decision. Read `DESIGN.md` before changing anything that touches data format, the model, or the checkpoint layout.

Two execution contexts:
- **Windows (this machine)**: root-level scripts — train, check, export. No eSpeak/phonemizer/G2P needed; `data/` is already phonemized.
- **Linux server**: `server_side/` — rebuilds `data/` from the real StyleTTS2 G2P pipeline, verifies the exported checkpoint, grafts it into an existing StyleTTS2 run. These import modules from the StyleTTS2 repo and won't run here.

## Commands

The venv is at `.venv` (Python 3.12; 3.13 is unsupported). Use `.venv/Scripts/python.exe` or activate first.

```bat
setup.bat                                   REM create venv, pip install -r requirements.txt, run check_setup.py
python check_setup.py                       REM preflight: packages, GPU, data, config/stats agreement, 2 real training steps + peak VRAM; finds largest batch_size that fits
python check_setup.py --device cpu          REM same, without a GPU
python train_plbert.py --config config.yml  REM train; auto-resumes from newest checkpoints\step_*.t7
python train_plbert.py --num-steps 20 --batch-size 2 --device cpu   REM quick smoke run
python export_for_styletts2.py --checkpoint checkpoints\step_200000.t7   REM -> export\ (encoder tensors + config.yml + util.py)
```

There is no test suite or linter; `check_setup.py` is the verification tool. Server-side verification is `server_side/verify_checkpoint.py --plbert-dir <dir>` (must report `MATCHED: 25`, `MISSING: 0`).

## Architecture and invariants

Data flow: `data/{train,val}.pkl` + `token_maps.pkl` (word → `{token}` ids, 30,000-word vocab + specials) → `dataset.py` (`PhonemeDataset` masks words/phonemes, crops to `max_mel_length`; `Collater`; `build_dataloader`) → `model.py` `MultiTaskModel` (ALBERT `encoder` + `mask_predictor` over 178 phoneme symbols + `word_predictor` over the word vocab) → loss = CE(vocab) + CE(token), computed in fp32 even under AMP.

Things that break silently if changed — these are the core constraints of the project:

- **`config.yml` → `model_params` is frozen.** StyleTTS2 rebuilds the encoder from these exact numbers (hidden 768, vocab_size 178, etc.).
- **`text_utils.py` symbol table is a verbatim copy** of StyleTTS2's `meldataset.py` (178 slots, `'` duplicated so 177 distinct). Do not edit it.
- **The attribute name `encoder` in `MultiTaskModel` is load-bearing**: StyleTTS2's `load_plbert()` keeps only keys with the `encoder.` prefix.
- **Checkpoint key format** (`train_plbert.py` `save_checkpoint`): every key is prefixed `module.` (the loader strips 7 chars), and `encoder.embeddings.position_ids` is re-inserted on save because the stock loader `del`s it unconditionally and newer transformers no longer emit it. Resume strips both back off.
- **`dataset_params.word_separator` must equal `data/stats.json` `word_separator_id`**, and `token_maps` vocab size must equal `num_vocab_for_model`; training refuses to start otherwise.
- `load_plbert()` uses `strict=False`, so key mismatches leave random weights with no error — any change to naming must be checked with `server_side/verify_checkpoint.py`.

Training loop behavior (`train_plbert.py`): resumes from the newest `step_*.t7` in `log_dir` (model + AdamW + GradScaler state), keeps the last `keep_last_checkpoints` (default 1), validates every `val_interval` on `val_batches` batches with a fixed numpy seed (so val losses are comparable) and saves `best.t7` + `best.json` on each new lowest `val/loss_total` (`best.t7` deliberately doesn't match `step_*.t7`, so resume/prune ignore it; `export_for_styletts2.py` names its output `step_<N>.t7` from the checkpoint's step because `load_plbert()` only loads `step_*`), clips the global grad norm at `grad_clip` (after `scaler.unscale_`; pre-clip max logged as `train/grad_norm_max`), stops with instructions if a validation is more than `collapse_margin` above the best (the first run collapsed at step 25k — all positions produced one identical hidden vector — and sat flat for 80k steps; the collapsed checkpoint is kept in `checkpoints/collapsed/`), always writes `checkpoints/metrics.csv` (an existing file with a different column set is renamed to `metrics_old_<timestamp>.csv`), and logs to W&B optionally (API key read from `.env` → `WANDB_API_KEY` via `env_file.py`, real env vars take precedence; failures never stop training; run id persisted in `checkpoints/wandb_run_id.txt` so resumes continue the same run). Loss baselines for reading curves: random vocab ≈ 10.31, random token ≈ 5.18.

OOM tuning order: `batch_size` (default 8 for 8 GB), then `max_mel_length` (512 → 384), then `precision`. The word-logits tensor (batch × 512 × 30,002) dominates activation memory, which is why the vocab is 30k not 50k.

## Repo notes

- `PLBERT_BN_UPDATE.zip` / `PLBERT_BN_UPDATE/` hold an update bundle (`check_setup.py`, `train_plbert.py`, `config.yml`, docs) whose files match the root copies' timestamps; the root copies are the working versions.
- `server_side/util.py` is a copy of StyleTTS2's PL-BERT loader and is shipped inside `export/`; keep it identical to upstream.
- `server_side/prepare_data.py` deliberately imports the real G2P (`3_g2p_pipeline.py`) and normalizer (`2_text_process.py`) from the StyleTTS2 repo rather than reimplementing them — fidelity to the TTS model's phonemes is the point (see `DESIGN.md` §4).
