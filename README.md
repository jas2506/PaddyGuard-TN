# PaddyGuard-TN

Paddy disease classifier, severity estimator, conformal abstention layer and a rule-based TNAU/CIBRC advisory. Final-year project. The full specification is in `ARCHITECTURE.md` (modules M0-M12); what is done and every decision and measured number so far is in `PROGRESS.md` and `reports/experiment_log.md`.

**Status:** M0-M4 are implemented and tested (106 tests). The EfficientNet-B0 baseline (seed 0) reaches 0.9202 macro-F1 on the Paddy Doctor test split. M5 (cross-dataset evaluation) is next.

## Get started

1. Clone and set up Python 3.10+:

   ```bash
   git clone https://github.com/jas2506/PaddyGuard-TN.git
   cd PaddyGuard-TN
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

2. Unzip the data package you were sent (`PaddyGuard-TN_data.zip`) **at the repository root**. It creates:

   - `data/manifests/manifest.csv`: every image with its dataset, label, duplicate group and split
   - `data/processed/`: the resized images (Paddy Doctor and Dhan-Shomadhan short side 256, Sethy at native size, plus Sethy lesion masks)
   - `runs/cls_effb0-0-20261019-123812/`: the trained seed-0 classifier (`ckpt.pt`) with its metrics and test predictions

   `data/` and `runs/` are git-ignored on purpose. The raw downloads are not in the zip; they are only needed to rebuild `data/processed/` (links in `docs/dataset_description.md`).

3. Point the code at the data and check everything works:

   ```bash
   export PADDYGUARD_DATA=$PWD/data
   export PADDYGUARD_OUT=$PWD
   pytest -q
   ruff check .
   ```

## Rules to know before changing anything

- The splits are fixed (seed 42). Never re-split; `dedup`, `splits` and `manifest` refuse to overwrite without `--force`.
- Never evaluate on the Kaggle Paddy Doctor unlabeled test set, and never tune on a target-domain test split.
- Everything is config-driven (`configs/`), seeds come from the config, and every script takes `--debug`.
- Do not commit data, checkpoints or secrets.
- Training runs on a Kaggle GPU: see `docs/kaggle_guide.md`.

`CLAUDE.md` holds the same rules and the project context for Claude Code, so the assistant picks them up automatically when it is started in this folder.
