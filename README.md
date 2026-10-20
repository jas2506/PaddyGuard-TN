# PaddyGuard-TN

A paddy (rice) leaf-disease system built for Tamil Nadu. From one photo it should say **which disease it is, how severe it is, whether it is sure, and what to do**, using advice from TNAU and the CIBRC pesticide registry. The research question is whether a model trained on Tamil Nadu photos (Paddy Doctor) and tested on other datasets measures leaf damage accurately enough to trigger the right spray rule, and abstains when it cannot.

This is a final-year project. It is built from public datasets only.

| | |
| --- | --- |
| Specification | `ARCHITECTURE.md` (modules M0 to M12) |
| Progress, every decision and measured number | `PROGRESS.md`, `reports/experiment_log.md` |
| Datasets in plain language, with download links | `docs/dataset_description.md` |
| Running training on a Kaggle GPU | `docs/kaggle_guide.md` |
| Architecture diagram | `docs/PaddyGuard.drawio.png` |

## Where the project stands

- **Done (Phase 1):** M0 utilities, M1 data manifest, M2 duplicate removal and fixed splits, M3 label map, loaders and augmentation, M4 classifier training. 106 automatic tests pass.
- **Result so far:** EfficientNet-B0, seed 0, test macro-F1 **0.9202** on the Paddy Doctor test split (accuracy 0.926). The acceptance check passed; the weakest class is brown spot (recall 0.84).
- **Next (Phase 2):** M5 cross-dataset evaluation, then M6 to M12 (severity, simulated shift, abstention, advisory, decision evaluation, app, report). Every module from M5 on is still a docstring-only stub in `src/paddyguard/`. Build them in the order given in `ARCHITECTURE.md`.

## Getting started

You need **three things**: this repository (GitHub), the **data zip** (sent to you separately by the project owner), and Python 3.10 or newer.

### Step 1. Clone the repository

```bash
git clone https://github.com/jas2506/PaddyGuard-TN.git
cd PaddyGuard-TN
```

### Step 2. Install the Python packages

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

A GPU is not needed to read, test or run the evaluation code. Training runs on a Kaggle GPU (see `docs/kaggle_guide.md`).

If `import paddyguard` fails after `pip install -e .` (this happens on some Macs), run commands as `PYTHONPATH=src python -m paddyguard.<module>` instead. `pytest` does not need this.

### Step 3. Add the data

The datasets are too large for GitHub, so they travel as one zip file: **`PaddyGuard-TN_data.zip` (about 750 MB), which you get from the project owner.**

1. Copy the zip into the **root of the cloned repository**, the folder that contains `README.md` and `src/`.
2. Unzip it there:

   ```bash
   unzip PaddyGuard-TN_data.zip       # or right-click and Extract Here
   ```

3. You should now have this layout:

   ```
   PaddyGuard-TN/
   ├── data/
   │   ├── manifests/manifest.csv     every image, its dataset, label, duplicate group and split
   │   └── processed/
   │       ├── S/    Paddy Doctor, resized (short side 256)
   │       ├── T1/   Sethy images at native size, plus the *_mask.png lesion masks
   │       └── T2/   Dhan-Shomadhan (field and white background), resized (short side 256)
   ├── runs/cls_effb0-0-20261019-123812/   trained seed-0 classifier (already in the repo)
   └── src/, configs/, tests/, ...
   ```

   The zip contains only `data/`. Do not move or rename anything inside it; the manifest refers to these paths. `data/` is git-ignored, so it will never be committed by accident.

The zip holds the processed data, which is everything the code reads. The original raw downloads are **not** needed unless you want to rebuild the processed data; the links to them are in `docs/dataset_description.md`.

### Step 4. Check that everything works

```bash
pytest -q
ruff check .
```

Expected: **106 passed** and `All checks passed!`. If you see `105 passed, 1 skipped`, the data is not in the right place (one test is skipped when the manifest is missing); recheck Step 3.

The code finds the data automatically in `<repo>/data`. To keep the data elsewhere, set `PADDYGUARD_DATA` to that folder, and set `PADDYGUARD_OUT` to where run folders should be written (default: the repo root).

You are now ready to work. Read `PROGRESS.md` first: it records everything M0 to M4 did, every decision and every measured number, and the agreed plan for M5.

## What is in the repository

| Folder | Contents |
| --- | --- |
| `src/paddyguard/` | The code, one package per area: `data`, `classify`, `segment`, `shift`, `reliability`, `advisory`, `evaluation`, `app`, `utils` |
| `configs/` | One YAML file per experiment or topic. Values live here, not in the code |
| `tests/` | The automatic tests |
| `runs/cls_effb0-0-20261019-123812/` | The trained seed-0 classifier: `ckpt.pt`, `metrics.json`, `preds_S_test.parquet`, config and log |
| `reports/` | Dataset card, duplicate and split reports, the experiment log |
| `docs/` | Dataset description, Kaggle guide, project guide, architecture diagram |
| `scripts/`, `notebooks/` | Helper script and the thin Kaggle notebook (it only clones the repo and calls the scripts) |

## Useful commands

```bash
# the classifier (debug mode: 200 images, 1 epoch, CPU)
python -m paddyguard.classify.train --config configs/cls_effb0.yaml --seed 0 --debug

# data steps (already run; the splits are fixed and refuse to overwrite without --force)
python -m paddyguard.data.manifest --debug
python -m paddyguard.data.dedup --debug

# format code before committing
ruff format src tests
```

Every script accepts `--debug`, which runs on a few hundred images on a CPU so it can be tested on a laptop.

## Rules that matter

These are the rules the project depends on. Breaking them silently invalidates results.

1. **The splits are fixed (seed 42). Never re-split.** Duplicates were removed and grouped so that near-copies never sit on both sides of a split.
2. **Never evaluate on the Kaggle Paddy Doctor unlabeled test set.** Never tune anything on a target-domain test split (Sethy or Dhan-Shomadhan).
3. **Everything is config-driven.** No magic numbers in the code; seeds come from the config.
4. **Do not commit data, checkpoints or secrets.** The one exception is the seed-0 baseline folder `runs/cls_effb0-0-20261019-123812/`. Other `runs/` folders are git-ignored.
5. **Save per-image predictions** (probabilities, not only metrics) for every evaluation, so later analysis never needs retraining.
6. **No pesticide names, doses or thresholds in Python.** They belong in `advisory/kb/*.json` with a source.
7. **Call the shift experiment "simulated"** in every filename, table and figure caption.
8. **Do not invent facts.** If a value is unknown, ask the project owner.
9. Run `pytest -q` and `ruff check .` before pushing.

## Working with Claude Code

`CLAUDE.md` and `.claude/` are part of the repository on purpose. If you use Claude Code, start it inside the cloned folder. It reads `CLAUDE.md`, which tells it to read `PROGRESS.md` first and holds the rules above, so it picks up where the project stands.

## Data credits

Paddy Doctor (Tamil Nadu), Sethy and RiceSeg (Odisha) and Dhan-Shomadhan (Bangladesh) are public datasets, reported by the project owner as CC BY 4.0, which requires credit. Sources, links and citations are in `docs/dataset_description.md`.
