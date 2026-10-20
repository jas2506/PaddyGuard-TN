# PaddyGuard-TN: Technical Architecture and Build Spec (public-data-only version)

Single-developer build spec, written to be read by Claude Code. Companion to the *PaddyGuard-TN: Project Guide* (problem statement, literature, datasets). This file says **what to build, in what order, with what interfaces, and when each part is done**.

Constraints that shape every decision below:

- One developer, about 12 steady weeks of work inside a 4 to 5 month window.
- Laptop has **no useful GPU**. All training runs on **Kaggle notebooks (or Colab free)**. The laptop is for writing code, unit tests and tiny CPU smoke runs.
- Only fine-tuning of small pretrained models. No pretraining, no large sweeps.
- **Public data only.** The developer will not photograph, annotate or collect anything. Where a real-world shift test would have used own photos, this spec uses a **controlled simulated shift** instead (module M7) and must always call it "simulated".
- English-only advice, three diseases, no soil or weather modifiers.

---

## 0. How to use this file with Claude Code

1. Create the repo (`paddyguard/`), put this file at the root as `ARCHITECTURE.md`.
2. Copy the block below into a root `CLAUDE.md` so every session starts with the rules.
3. Work **one module at a time** in the order of section 4. For each module, open a session and say: *"Implement module Mx from ARCHITECTURE.md. Follow its interfaces and acceptance checks. Write tests first where the module says so."*
4. After each module, run its acceptance checks, commit, then move on. Do not start a module whose dependencies are not done.

```markdown
# CLAUDE.md (copy to repo root)
Project: PaddyGuard-TN. Spec: ARCHITECTURE.md (read the relevant module before coding).
Rules:
- Python 3.10+, type hints, small functions, no notebooks as source of truth (notebooks only call scripts).
- Everything is config-driven: YAML in configs/, one config per experiment. No magic numbers in code.
- Fix seeds (python, numpy, torch) from config; log config + git commit + seed with every run.
- Never commit data, checkpoints or secrets. Paths come from env vars PADDYGUARD_DATA and PADDYGUARD_OUT.
- Every script supports --debug (200 images, 1 epoch, CPU) so it can be tested on a laptop.
- Save per-image predictions (probabilities, not just metrics) for every evaluation so later analysis never needs retraining.
- Never evaluate on the Kaggle Paddy Doctor unlabeled test set. Never tune on a target-domain test split.
- Never hard-code pesticide names, doses or thresholds in Python. They live only in advisory/kb/*.json with a source field.
- Call the shift experiment "simulated" in every filename, table and figure caption.
- Do not invent facts. If a value is unknown (mask encoding, class folder names, TNAU numbers), stop and ask.
- Run `pytest -q` and `ruff check .` before declaring a task done.
```

---

## 1. System overview

```
                         (training time)                                      (inference time)
 Paddy Doctor ──┐                                              leaf photo
 Sethy + masks ─┼─► M1 acquire ─► M2 dedup+splits ─► M3 loaders      │
 Dhan-Shomadhan ┘                                   │                ├──► M4 classifier ──► M8 conformal gate ──┐
                                                    ├─► M4 train cls │                                           ├─► M9 advisory ─► M11 app
                                                    ├─► M5 cross-dataset eval + ablations                       │   (KB: TNAU + CIBRC)
                                                    ├─► M6 train severity (RiceSeg masks)   └──► M6 severity ───┘
                                                    ├─► M7 simulated-shift test sets (Sethy leaves on blurred field backgrounds)
                                                    ├─► M8 calibrate (source vs target)
                                                    └─► M10 decision-level eval (clean + shifted) ─► M12 reports
```

Two models run on every photo (disease classifier, lesion/leaf segmenter). The conformal gate decides answer or abstain. The advisory engine is **rule-based**: it reads a JSON knowledge base built from TNAU and CIBRC. There is no generative text in the core path.

### Domains used throughout

| Code | Dataset | Role |
| --- | --- | --- |
| `S` | Paddy Doctor (Tamil Nadu) | Source: train, validation, in-domain test; also the pool of field backgrounds for M7 |
| `T1` | Sethy (Odisha) with RiceSeg masks | Cross-dataset target; severity training and test images; source of leaves for M7 |
| `T2` | Dhan-Shomadhan (Bangladesh), field-background images only | Second cross-dataset target (classification and abstention only; no masks). The white-background images are the zero-shot-only row `T2-white` |
| `SX` | Simulated-shift composites built from `T1` leaves | Controlled severity and decision stress test (M7) |

---

## 2. Repository layout

```
paddyguard/
  CLAUDE.md
  ARCHITECTURE.md
  pyproject.toml            # package "paddyguard", editable install
  requirements.txt
  configs/
    data.yaml               # paths, dedup threshold, split ratios
    labelmap.yaml           # class harmonisation across datasets
    cls_effb0.yaml          # (optional) cls_mnv3.yaml cls_convnext_t.yaml
    seg_unet_r34.yaml
    shift.yaml              # simulated-shift levels
    reliability.yaml        # alpha, calibration sizes, margin delta
    severity_bands.yaml     # band edges per disease (design choices unless sourced)
    advisory.yaml
  data/                     # gitignored
    raw/ interim/ processed/ manifests/
  src/paddyguard/
    data/        download.py manifest.py dataset_card.py dedup.py splits.py cache.py labelmap.py datasets.py transforms.py inspect_masks.py
    classify/    models.py mixstyle.py train.py evaluate.py ablations.py
    segment/     models.py train.py leafmask.py severity.py evaluate.py
    shift/       composite.py build_shift_sets.py
    reliability/ calibrate.py conformal.py abstain.py metrics.py
    advisory/    kb/ (blast.json brown_spot.json bacterial_leaf_blight.json tungro.json)  schema.py kb.py engine.py templates/en.yaml checker.py
    evaluation/  decision_eval.py stats.py tables.py figures.py
    app/         streamlit_app.py
    utils/       seed.py io.py logging.py paths.py
  notebooks/     kaggle_run.ipynb   # thin: clone repo, pip install -e ., call scripts
  tests/
  runs/          # gitignored: runs/<run_id>/{config.yaml, metrics.json, preds_<domain>.parquet, ckpt.pt}
  reports/       # committed small results: results/*.csv, figures/*.png, tables/*.md
```

Dependencies (`requirements.txt`): `torch`, `torchvision`, `timm`, `segmentation-models-pytorch`, `albumentations`, `opencv-python-headless`, `pillow`, `imagehash`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `scipy`, `pyyaml`, `jinja2`, `pydantic`, `matplotlib`, `streamlit`, `pytest`, `ruff`. Optional: `mapie`, `onnx`, `onnxruntime`.

---

## 3. Working on Kaggle and Colab

**Pattern.** Code lives in a Git repo. A thin Kaggle notebook clones it and calls scripts.

```python
# notebooks/kaggle_run.ipynb  (cell 1)
!git clone https://github.com/<you>/paddyguard.git
%cd paddyguard
!pip install -q -e .
import os
os.environ['PADDYGUARD_DATA'] = '/kaggle/input'
os.environ['PADDYGUARD_OUT']  = '/kaggle/working'
# cell 2
!python -m paddyguard.classify.train --config configs/cls_effb0.yaml --seed 0
```

If you prefer to let Claude Code launch runs itself, the Kaggle CLI can push a notebook with GPU enabled (`kaggle kernels push`, with `enable_gpu` set in `kernel-metadata.json`). Try one small run first to confirm it works for your account.

**Data on Kaggle.**

- Paddy Doctor: attach the competition via *Add Input* after joining and accepting the rules. It appears under `/kaggle/input/paddy-disease-classification`.
- Sethy, RiceSeg-5932, Dhan-Shomadhan: download the zips from Mendeley in a browser, then create **private Kaggle Datasets** (`paddyguard-sethy`, `paddyguard-riceseg`, `paddyguard-dhan`). Attach them to notebooks like any input. Mendeley has no stable API, so this is the reliable route.
- After M2 and M7, save the resized images, shift composites and `manifest.csv` as one more private dataset (`paddyguard-processed`) so later notebooks skip preprocessing.

**Making runs survive session limits.**

- Checkpoint every epoch to `$PADDYGUARD_OUT/runs/<run_id>/ckpt.pt`; support `--resume`.
- Dataloading is usually the bottleneck on 4 CPU cores. In M2, **pre-resize all images once** (short side 256, JPEG quality 95) and train from those. Use `num_workers=2..4`, `persistent_workers=True`, `pin_memory=True`.
- Use mixed precision (`torch.autocast`, `GradScaler`) everywhere on GPU.
- A finished notebook's `/kaggle/working` output can be attached to later notebooks. Download `reports/` and `preds_*.parquet` to the laptop after each stage.
- Colab: mount Google Drive, set `PADDYGUARD_OUT` to a Drive folder, same scripts.

**Laptop.** `--debug` runs the whole pipeline on 200 images for 1 epoch on CPU. This catches nearly all bugs before spending GPU quota.

**GPU budget (estimates, not measurements; re-measure in week 1).**

| Job | Approx. cost on a T4-class GPU |
| --- | --- |
| EfficientNet-B0, 224 px, 15 epochs on about 7k images | 15 to 25 min |
| Core classification grid: EfficientNet-B0 × 3 variants × 3 seeds = 9 runs | 3 to 4 hours |
| U-Net (ResNet34) at 384 px, 20 epochs on about 4k images | 40 to 90 min per run; 2 runs (with and without paste augmentation) |
| Optional extras (MobileNetV3, ConvNeXt-Tiny, few-shot) | 3 to 8 hours |
| Everything else (shift set generation, evaluation, conformal, advisory) | CPU only |

Core total is roughly 5 to 7 GPU hours. Kaggle has a weekly GPU quota; check the current limit and plan runs around it.

---

## 4. Modules

Build order: **M0 → M1 → M2 → M3 → M4 → M5**, then **M6 → M7**, then **M8**, **M9** (the knowledge base can be written in parallel with M5 to M7), **M10**, **M11**, with **M12** written continuously.

| ID | Module | Depends on | Effort (days, one person) |
| --- | --- | --- | --- |
| M0 | Scaffolding and utilities | none | 1 |
| M1 | Data acquisition and manifests | M0 | 2 |
| M2 | Data quality: duplicates and grouped splits | M1 | 2 |
| M3 | Label harmonisation and loaders | M2 | 1 |
| M4 | Disease classifier training | M3 | 2 |
| M5 | Cross-dataset evaluation and robustness ablations | M4 | 3 |
| M6 | Severity estimation | M2, M3 | 4 |
| M7 | Simulated-shift stress test | M2, M6 | 3 |
| M8 | Reliability: calibration, conformal, abstention | M5, M6 | 3 |
| M9 | Advisory knowledge base and engine | M6 (bands), TNAU/CIBRC reading | 4 |
| M10 | Decision-level evaluation | M7, M8, M9 | 3 |
| M11 | Demo app | M4, M6, M8, M9 | 2 |
| M12 | Reporting and statistics | all | 2 |

Total about 32 working days of effort, which fits the 12-week plan with room for GPU waiting time.

---

### M0. Scaffolding and utilities

**Purpose.** A reproducible skeleton so every later module logs, seeds and saves the same way.

**Deliverables.**

- `utils/seed.py::set_seed(seed:int)` seeds python, numpy, torch. Runs are not bit-exact on GPU; record that.
- `utils/paths.py::data_root()`, `out_root()` read `PADDYGUARD_DATA`, `PADDYGUARD_OUT` with sensible defaults.
- `utils/io.py::load_yaml`, `save_json`, `save_parquet`, `new_run_dir(config_name, seed)` returning `runs/<config>-<seed>-<timestamp>/` and writing `config.yaml` and `git_commit.txt` into it.
- `utils/logging.py` simple console + file logger.
- `tests/` running on CPU with synthetic images.

**Done when.** `python -c "import paddyguard"` works after `pip install -e .`; `pytest -q` passes on the laptop.

---

### M1. Data acquisition and manifests

**Purpose.** Turn three differently organised downloads into one uniform table.

**Inputs.** Raw folders for `S`, `T1` (with RiceSeg masks), `T2`.

**Output.** `data/manifests/manifest.csv`, one row per image:

| Column | Meaning |
| --- | --- |
| `image_id` | Unique string, e.g. `S_000123`, `T1_000456` |
| `dataset` | `S`, `T1`, `T2` |
| `path` | Path relative to the data root |
| `label_raw` | Folder or CSV label as shipped |
| `label` | Harmonised label (filled in M3) |
| `variety`, `age_days` | From Paddy Doctor `train.csv`; empty elsewhere |
| `background` | `field` or `white` for `T2` (from the top-level folder and the info CSV); empty elsewhere. `T2` main target = `field`; `white` = row `T2-white` |
| `width`, `height` | Original size |
| `mask_path` | RiceSeg mask for `T1`; empty otherwise |
| `phash` | 64-bit perceptual hash (hex; filled in M2) |
| `group_id` | Duplicate-cluster id (filled in M2) |
| `split` | `train/val/test` for `S`; see M2 for `T1` and `T2` |

**Implementation notes.**

- `download.py` does **not** scrape Mendeley. It only verifies that expected folders exist, prints counts per class, and prints instructions if something is missing.
- `manifest.py` has one reader per dataset. The Paddy Doctor Kaggle layout is expected to be `train_images/<class>/<file>.jpg` plus `train.csv` (columns image_id, label, variety, age). **Verify this on download; do not assume.**
- Pair RiceSeg masks to Sethy images by **exact filename stem**. Test that every Sethy image has exactly one mask and report unmatched files (expected 5,932 matches). Match case-insensitively on the extension, since the shipped files mix `.jpg` and `.JPG`. A check on 2026-10-19 found all 5,932 stems matching exactly (the filename typos, e.g. `BACTERAILBLIGHT`, are present in both images and masks), with equal image and mask sizes on 60 sampled pairs; the script must still re-verify this and report unmatched files.
- Write a dataset card to `reports/dataset_card.md`: counts per class, image size distribution, licence text copied from each dataset page.

**`data/inspect_masks.py` (week-1 discovery, blocking for M6).** RiceSeg masks are JPG files made in Label Studio, so values are noisy and the **class encoding is not documented in what has been checked**. The script must:

1. Load 50 random masks; print unique values after rounding, the histogram of pixel intensities, and whether masks are grayscale or colour.
2. Save overlay images (mask over photo) for 20 samples to `reports/figures/mask_overlays/`.
3. Write a short markdown note: does a mask mark only lesions, or leaf plus lesion with different values? Is the background pure black or white? Which threshold cleans the JPEG noise?

**Stop and ask the developer** to look at the overlays before M6 starts. Do not guess the encoding.

**Done when.** `manifest.csv` exists for all three datasets, counts match the dataset pages, and the mask note exists.

---

### M2. Data quality: duplicates and grouped splits

**Purpose.** Prevent leakage. Near-duplicate photos in both train and test inflate every number, and the dataset pages report no duplicate counts, so measure it.

**Algorithm.**

1. Compute `imagehash.phash(img, hash_size=8)` (64 bits) for every image; store as hex.
2. Find near-duplicate pairs: convert hashes to `uint64`, compute pairwise Hamming distance in chunks with a vectorised popcount. A 64-bit phash always has 32 set bits, so only even distances occur. **Thresholds are per dataset** (`dedup.hamming: {S, T1, T2}`): the plain-background T2 leaves need a much tighter threshold than the field photos, otherwise different leaves chain into one group (measured: 8 merged 33 different T2 leaves). The pair-count histograms and ~50 pairs around the thresholds are reviewed by eye and the choice recorded in `dedup.review_note` and `reports/dedup_report.md` (values chosen: S 8, T1 14, T2 4). Every full re-run must regenerate `reports/figures/dedup_boundary_pairs.jpg` for a re-check.
3. Union-find over pairs to get `group_id` clusters, **within each dataset**.
4. **Cross-dataset leakage check.** Compute pairs between `S` and each target. Any target image within threshold of a source image is dropped from the target (and counted in the report). A non-zero count is itself a finding.
5. Keep one representative per cluster (the image whose `image_id` equals its `group_id`, see `dedup.is_representative`) for target **evaluation** sets; keep all members of a cluster together in the same split. The cross-dataset leakage threshold is separate (`dedup.leakage_hamming`, 8). Drops are recorded in the manifest `split` column as `dropped_duplicate` / `dropped_leakage`: T2 duplicates are dropped by `dedup.py` (T2 has no training use); T1 duplicates stay as group members and the splitter drops the extra members only inside `seg_test`, so the segmenter keeps its training images. Any evaluation on all of T1 outside `seg_test` (the M5 zero-shot classifier evaluation) must use representatives only.

**Splits (`splits.py`), fixed seed 42, never re-split.**

- `S`: stratified group split of the 10,407 labelled Kaggle images into `train 70% / val 15% / test 15%`, by label, grouped by `group_id` (use `sklearn.model_selection.StratifiedGroupKFold`).
- `T1` (Sethy): zero-shot target for the classifier, so no classifier training split. For the segmenter and the conformal experiments, split by group into `seg_train 70% / seg_val 15% / seg_test 15%` (`StratifiedGroupKFold`, 20 folds, so the ratios are exact in folds). **`seg_test` is the evaluation set for severity, conformal abstention and decision-level evaluation on `T1`.** Conformal calibration subsets are drawn only from `T1` images outside `seg_test`.
- `T2` (Dhan-Shomadhan): **main target = the 337 field-background images** (decision of 2026-10-19; see `reports/experiment_log.md`). After dedup, split by group, stratified by class, into `cal_pool 40%` and `test 60%`; conformal calibration subsets are drawn from `cal_pool`. No training split. **Run dedup first, then print the per-class counts of the three shared classes in the field subset (only these can calibrate the shared-space conformal gate; Leaf Scaled and Sheath Blight cannot).** Calibration sizes for `T2`: every value in `{25, 50, 100}` that is at most 65% of the pool, so a draw never leaves less than about 35% of the pool unused (otherwise the 100 random draws are nearly identical). If a class (expected: brown spot) has fewer than about 30 test images, keep it, report its per-class numbers with bootstrap CIs and flag it as low-n in every `T2` table. **Fallback, only if the post-dedup field counts are unusable:** split all shared-class Dhan images (field and white) as originally specified, stratified by background, and report field and white separately; this enlarges the pool but blurs the shift story. The 769 white-background images are a separate, zero-shot-only secondary row reported as `T2-white`: no `cal_pool`, no calibration. Calibration sizes: `{25, 50, 100}` for `T2` and `{25, 50, 100, 150}` for `T1`.

**Cached resize** (`data/cache.py`, writes `data/processed/`; a `--debug` run writes `data/processed_debug/` because debug image ids differ from the real ones). After dedup, write resized copies to `data/processed/<dataset>/<image_id>.jpg` (short side 256, aspect ratio kept) for `S` and `T2`. All training and evaluation read these. **Exception for `T1`:** Sethy images are only about 300×300 pixels (measured on a sample, 2026-10-19), so downscaling them to a short side of 256 would throw away lesion detail the segmenter needs. For `T1`, cache at native resolution (never downscale; only unify the format) and cache the RiceSeg mask for each image alongside it (`data/processed/T1/<image_id>_mask.png`), with exactly the same geometry as the image. The classifier's `eval` transform resizes `T1` images itself. Raw `T2` images are very large (about 1952×4160), so resizing them is what makes loading fast.

**Run order.** `python -m paddyguard.data.dedup`, then `.splits`, then `.cache`. `dedup` and `splits` write into `manifest.csv` (`phash`, `group_id`, `split`); both refuse to run again once splits exist, and `manifest` refuses to overwrite such a manifest without `--force`, because splits are never redone. Reports: `reports/dedup_report.md`, `reports/splits_report.md` (which also prints the T2 per-class counts, low-n flags and the allowed calibration sizes).

**Measured on the real data (2026-10-19; regenerate from `reports/` if anything changes).**

- Image sizes: Paddy Doctor 480×640 (a few 640×480); Sethy about 300×300 (about 7% of the `.jpg` files are really PNGs); Dhan about 1952×4160, with EXIF rotation tags on part of them.
- Duplicates (thresholds S 8, T1 14, T2 4): Paddy Doctor 10,407 images in 6,141 groups (largest 51); Sethy 5,932 images in 1,847 groups (largest 25), with 5,174 pairs whose hashes are identical; Dhan 1,106 images in 1,099 groups. Sethy is therefore heavily duplicated, which is a finding in itself (the guide warned of it). No target image was within the leakage threshold of a Paddy Doctor image.
- Splits: `S` train 7,334 / val 1,587 / test 1,486. `T1` seg_train 4,074 / seg_val 927 / seg_test 280 (87 bacterial blight, 66 blast, 79 brown spot, 48 tungro), with 651 extra duplicates dropped from seg_test; only 1,847 `T1` representatives exist for any evaluation over all of Sethy. `T2` field cal_pool 135 / test 201, of which the three shared classes are pool 79 / test 119; `T2-white` 763 zero-shot; 7 `T2` duplicates dropped.
- Consequences for later modules: `T2` calibration sizes are {25, 50} (pool 79 x 65%); `T2` brown spot has 28 test images (low-n); `seg_test` has only 280 images, so severity and decision results need bootstrap CIs; the segmenter trains on 4,074 `T1` images, many of them near-duplicates of each other.

**Tests.** (1) No `group_id` appears in two splits. (2) A planted exact duplicate is found. (3) A planted slightly-resized duplicate is found.

**Done when.** `dedup_report.md` and the splits exist, the leakage test passes, processed images are cached.

**Pitfall.** A threshold that is too loose merges different leaves that look alike (plain-background lab shots are very similar). That is why boundary pairs must be inspected by eye.

---

### M3. Label harmonisation and loaders

**Purpose.** One canonical label space, and PyTorch datasets that can restrict predictions to the classes two datasets share.

**`configs/labelmap.yaml` (fill after inspecting real folder names; the values below are expected, not verified).**

```yaml
canonical:            # order defines class indices for the 4-class shared space
  - bacterial_leaf_blight
  - blast
  - brown_spot
  - tungro
extra_source_classes:  # Paddy Doctor-only, used for in-domain training and in-domain test only
  - normal
  - bacterial_leaf_streak
  - bacterial_panicle_blight
  - dead_heart
  - downy_mildew
  - hispa
map:
  S:  {bacterial_leaf_blight: bacterial_leaf_blight, blast: blast, brown_spot: brown_spot, tungro: tungro, normal: normal}   # plus the extra classes as themselves
  T1: {}   # FILL from Sethy folder names after download (expected: bacterial blight, blast, brown spot, tungro)
  T2: {}   # FILL from Dhan-Shomadhan folder names (the page lists "Leaf Scaled", "Rice Turngo", "Steath Blight"; folders may carry the same typos)
shared:
  S_T1: [bacterial_leaf_blight, blast, brown_spot, tungro]
  S_T2: [blast, brown_spot, tungro]
```

**`data/datasets.py`.**

- `PaddyDataset(manifest, split, domain, transform, classes=None, with_mask=False, mask_threshold=None, representatives_only=False)` returns `(image, label_idx, image_id)` (plus the binary lesion mask when asked). `domain` is one of `S`, `T1`, `T2` (field background only) and `T2-white`; `split=None` means every image that was not dropped in M2. Images are read from `data/processed/`. `with_mask` is T1 only and needs `mask_threshold` (`mask_encoding.threshold` in `configs/data.yaml`, 128), because the cached masks keep the raw JPEG values. `representatives_only=True` keeps one image per duplicate group and must be used for any evaluation over all of `T1` (M5 zero-shot); `seg_test` already contains only representatives.
- `classes` restricts to a class subset and re-indexes labels; images whose label is outside the subset are skipped.
- `make_loader(dataset, training, batch_size, seed, num_workers, pin_memory)` with a seeded generator and per-worker augmentation seeds (otherwise every worker repeats the same augmentations). A training loader accepts **only `S/train`** and raises otherwise; this is the code-level assertion that target domains never enter training. `data/labelmap.py` also provides `load_labelmap`, `harmonise` (fills the manifest `label` column; run `python -m paddyguard.data.labelmap` once after M2), `shared_classes(lm, domain)`, `restrict_logits` and `softmax`. The classifier's output columns are `canonical + extra_source_classes` in that order (`LabelMap.source_classes`).

**Restricting predictions (used in M5 and M8).** The classifier is trained on all source classes (10 for the Kaggle version). For cross-dataset evaluation, take the logits, **keep only the columns of the shared classes**, apply softmax over those columns, and predict. Implement as `restrict_logits(logits, class_names, shared)`.

**`data/transforms.py`** (Albumentations 2.x, ImageNet mean/std, 224 px; all numbers in `configs/transforms.yaml`, where the magnitudes the spec does not give are marked as design choices):

- `baseline`: RandomResizedCrop(224, scale=(0.5,1)), HorizontalFlip.
- `strong`: baseline + colour jitter (ColorJitter, RandomBrightnessContrast, HueSaturationValue), GaussianBlur, ImageCompression (JPEG quality 40 to 95), CoarseDropout, small rotation.
- `eval`: Resize(256) → CenterCrop(224).

**Tests.** The label map covers every raw label found in the manifest; shared-space restriction produces rows that sum to 1.

**Done when.** A loader yields correct shapes and class indices for all domains.

---

### M4. Disease classifier training

**Purpose.** Strong, honest in-domain baselines, then the model used everywhere downstream.

**Models (`classify/models.py`).** `timm.create_model(name, pretrained=True, num_classes=C)`:

- `efficientnet_b0` (the main model; everything in the core plan uses it)
- optional: `mobilenetv3_large_100`, `convnext_tiny`, and `resnet34` as the reference used in the Paddy Doctor paper.

**Training recipe (`classify/train.py`, config-driven).**

```yaml
# configs/cls_effb0.yaml
model: efficientnet_b0
img_size: 224
epochs: 15
batch_size: 64
optimizer: {name: adamw, lr: 3.0e-4, weight_decay: 0.01}
scheduler: {name: cosine, warmup_epochs: 1}
loss: {name: ce, label_smoothing: 0.1}
amp: true
augmentation: baseline                   # baseline | strong | strong_mixstyle
seed: 0
early_stopping: {metric: val_macro_f1, patience: 4}
```

- Loop: train epoch → evaluate on `S/val` (macro-F1, accuracy) → checkpoint → early stopping on val macro-F1 → load the best checkpoint and evaluate on `S/test`.
- Save: `metrics.json` (val/test macro-F1, accuracy, balanced accuracy, per-class recall, per-variety macro-F1 on test, per-epoch history, best epoch, and an `acceptance` block), `preds_S_test.parquet` (the section 6 prediction format: image_id, y_true, y_pred, one `logit_<class>` per training class, domain, split), `ckpt.pt` (best weights plus class names and config, what later modules load) and `ckpt_last.pt` (model, optimizer, scheduler, scaler, history and early-stopping state, used by `--resume RUN_DIR`). The classifier's img_size comes from the classifier config; `configs/transforms.yaml` keeps the 256/224 resize ratio. `--debug` uses 200 images per split, one epoch and the CPU; `--no-pretrained` allows offline debugging.

**Running on Kaggle:** see `docs/kaggle_guide.md` and `notebooks/kaggle_run.ipynb`; `scripts/package_for_kaggle.sh` builds the two datasets to upload.

**Acceptance.** The Paddy Doctor paper reports F1 97.5% on the 13-class version with ResNet34. The Kaggle version has 10 classes, so exact numbers differ; the check is "clearly above 90% macro-F1 and no single class below 80% recall". If lower, debug the data pipeline before moving on.

**Pitfalls.** (1) Training on `T1` or `T2` by accident: assert in code that target domains never appear in a training loader. (2) Class imbalance: report macro-F1, not accuracy.

---

### M5. Cross-dataset evaluation and robustness ablations

**Purpose.** Produce the headline benchmark: how much accuracy is lost when the model meets photos from another dataset, and what cheaply helps.

**Evaluations (`classify/evaluate.py`).** For each trained run:

- `S/test` in-domain (all classes).
- `T1` zero-shot on the 4 shared classes; `T2` (field background) zero-shot on the 3 shared classes; `T2-white` (the 769 white-background images) zero-shot on the same 3 classes as a separate secondary row, with no calibration pool.
- Metrics: macro-F1 (primary), balanced accuracy, per-class recall, confusion matrix; per-image probabilities saved to `preds_<domain>.parquet`.
- Gap = in-domain macro-F1 **on the same shared classes** minus target macro-F1. Always compute in-domain on the restricted class set too, so the comparison is like for like.
- Per-variety breakdown on `S/test` using Paddy Doctor's variety column.

**Core ablation grid (`classify/ablations.py` generates configs and the run list).**

| Axis | Values |
| --- | --- |
| Backbone | efficientnet_b0 |
| Training variant | `baseline`, `strong`, `strong_mixstyle` |
| Seeds | 0, 1, 2 |

That is **9 runs**. Optional extras, only if time allows: MobileNetV3 and ConvNeXt-Tiny with `baseline` and the best variant; class-balanced focal loss; few-shot fine-tuning with 5/10/25% of target labels (E8).

**MixStyle (`classify/mixstyle.py`).** Mixes per-sample feature statistics early in the network to reduce reliance on style (background, lighting). Insert after the first two stages via forward hooks.

```python
class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__(); self.p, self.alpha, self.eps = p, alpha, eps
        self.beta = torch.distributions.Beta(alpha, alpha)
    def forward(self, x):
        if torch.rand(1).item() > self.p:      # train/eval switching is done by the hook below
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True, unbiased=False) + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_n = (x - mu) / sig
        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu_mix = mu * lam + mu[perm] * (1 - lam)
        sig_mix = sig * lam + sig[perm] * (1 - lam)
        return x_n * sig_mix + mu_mix
```

For a timm EfficientNet, `model.blocks[1].register_forward_hook(lambda m, i, o: mixstyle(o) if m.training else o)` (and the same for `blocks[2]`). The `m.training` check is essential: the `MixStyle` object is not a registered submodule, so it would otherwise keep mixing at evaluation time. Add a unit test asserting that outputs are identical across two eval-mode forward passes and that shapes are unchanged.

**Statistics.** With 3 seeds, **do not report p-values**; they are meaningless at that size. Report mean ± std over seeds and a **bootstrap 95% CI over test images** (1,000 resamples, stratified by class). Report seed-wise paired differences between variants as a plain table. Grad-CAM, if used, is a qualitative figure only; a 2026 study found it did not separate correct from incorrect cross-domain predictions.

**Output.** `reports/results/cross_dataset.csv`: backbone, variant, seed, domain, n_images, macro_f1, balanced_acc, in_domain_macro_f1_same_classes, gap.

**Done when.** The cross-dataset table exists with CIs. **This is the first publishable result.**

---

### M6. Severity estimation

**Purpose.** Convert a photo into a lesion percentage that can trigger TNAU action thresholds.

**Prerequisite.** The `inspect_masks.py` note from M1. If the masks mark **lesions only**, a leaf mask must be derived separately (below). If they contain **leaf and lesion** classes, use them directly.

**Definition.**

```python
def severity_pct(lesion: np.ndarray, leaf: np.ndarray) -> float:
    """lesion, leaf: boolean HxW arrays. Lesion is clipped to leaf."""
    leaf_px = leaf.sum()
    return 100.0 * (lesion & leaf).sum() / max(int(leaf_px), 1)
```

**`segment/leafmask.py`.** Sethy images are leaves photographed on a plain background, so leaf ground truth for `T1` can be derived by thresholding the distance from the background colour (HSV or Lab), morphological closing, and keeping the largest connected component. Validate by overlay on 30 images and save the overlays. If RiceSeg already carries a leaf class, skip this and use it.

**Segmentation model (`segment/models.py`).** A two-channel (leaf, lesion) multi-label U-Net.

```python
import segmentation_models_pytorch as smp
model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=2)  # sigmoid outputs: leaf, lesion
```

- Input 384×384. Sethy images are only about 300 px wide, so 384 already upsamples them and 512 would add no information; do not spend GPU time on 512. Loss = BCE + Dice per channel. AdamW lr 3e-4, cosine, 20 epochs, batch 8, AMP.
- Data: `T1` images and masks, split `seg_train / seg_val / seg_test` from M2.
- Two training variants, each a separate run:
  - `V0`: standard augmentation (flips, rotation, colour jitter, scale).
  - `V1`: `V0` plus **background-paste augmentation** with probability 0.5, using the compositor from M7 and backgrounds from `S/train` only.
- Post-processing at inference: threshold 0.5; keep the largest connected component of the leaf channel; clip lesion to leaf.

**Where severity applies.** Percent-of-leaf severity is meaningful for lesion diseases (blast leaf lesions, brown spot, bacterial leaf blight). It is **not** the right measure for tungro (a viral disease with leaf discolouration; TNAU advice is about management, not lesion area). The knowledge base (M9) records per disease whether a severity rule exists. Do not force a severity number into a decision where TNAU gives none.

**Bands (`configs/severity_bands.yaml`).** Edges are configuration. The only TNAU threshold the research found is about **5% leaf area for blast**, and the developer must re-confirm it on the TNAU page. Other edges, including a default `[5, 25]` three-band split, are **design choices** and must be labelled as such in every table. Never present a design-choice band as a TNAU rule.

**Evaluation (`segment/evaluate.py`), on `seg_test`.**

- Segmentation: Dice and IoU for leaf and lesion channels.
- Severity: MAE in percentage points, Spearman correlation, band accuracy, quadratic-weighted kappa on bands.
- Save per-image predicted and true severity, band, and signed error.

**Pitfalls.** (1) JPEG-noisy masks: binarise with a validated threshold. (2) Very small lesions make Dice look poor while percent error is small; report both. (3) Leaf-mask errors can dominate severity error; also report severity computed with the ground-truth leaf mask to isolate lesion error.

**Done when.** `seg_test` Dice/IoU and severity MAE are reported for both `V0` and `V1`.

---

### M7. Simulated-shift stress test

**Purpose.** Measure how severity and decisions degrade when the photo no longer looks like a Sethy plain-background lab shot, **with exact ground truth**, without collecting any photos. This replaces a real cross-domain test and must always be called *simulated*.

**Idea.** Take `T1` leaves that have masks, cut them out, paste them onto **heavily blurred Paddy Doctor field photos**, then degrade lighting, sharpness and compression. The leaf and lesion masks are transformed with the same geometry, so the true severity is known and (apart from rounding) unchanged. Blurred backgrounds shift colour, brightness and texture statistics like a field would, but contain **no extra leaves**, so there is no ambiguity about which leaf is being measured.

**`shift/composite.py`.**

```python
def composite(leaf_rgb, leaf_mask, lesion_mask, bg_rgb, rng, *, canvas=384,
              scale=(0.5, 0.9), rot_deg=(-30, 30), bg_blur_sigma=25, feather=1.5):
    """Returns (image, leaf_gt, lesion_gt) as uint8 image and bool masks of the canvas size.
    Steps: crop leaf to its bounding box; rotate image and both masks with the same matrix
    (nearest-neighbour for masks); scale so the longest side is scale * canvas; place fully inside
    the canvas at a random position; background = bg_rgb resized to the canvas then Gaussian-blurred;
    alpha blend using the leaf mask feathered by Gaussian(feather) for RGB only. Ground-truth masks
    stay binary."""
```

**Levels (`configs/shift.yaml`).**

```yaml
levels:
  S0_clean:      {paste: false}
  S1_background: {paste: true}
  S2_mild:       {paste: true, blur_sigma: [0.5, 1.5], brightness: 0.2, jpeg_quality: [60, 90]}
  S3_strong:     {paste: true, blur_sigma: [1.5, 3.0], brightness: 0.4, gamma: [0.6, 1.6],
                  jpeg_quality: [25, 60], noise_std: [5, 15]}
```

Photometric steps use Albumentations (`GaussianBlur`, `RandomBrightnessContrast`, `RandomGamma`, `HueSaturationValue`, `ImageCompression`, `GaussNoise`). `S0_clean` is the plain `seg_test` images.

**`shift/build_shift_sets.py`.**

- Input: `T1/seg_test` images with masks; backgrounds from `S/test` (disjoint from the `S/train` backgrounds used by the `V1` training augmentation).
- For each image and level, derive a per-image seed from `image_id` + level, generate once, and **cache** to `data/processed/shift/<level>/<image_id>.jpg` plus `leaf_gt.png`, `lesion_gt.png`. The evaluation set is therefore fixed and reproducible.
- Also write 12 example composites per level to `reports/figures/shift_examples/` for the report.

**Experiments.**

- E5: for `V0` and `V1`, severity MAE, band accuracy and Dice on every level. The curve of MAE versus level is the result.
- Optional E9: the M4 classifier on the composites of the same leaves (note the leaves come from Sethy, so this tests background robustness of a Paddy-Doctor-trained model, not new diseases).

**Tests.** (1) Predicted-versus-true geometry: compositing with the identity transform returns the original masks. (2) True severity before and after compositing matches within 0.5 percentage points. (3) No background image used for evaluation appears in the `V1` training background pool.

**Limits to state in the report.**

- The shift is simulated: it does not contain real field effects such as shadows, overlapping leaves, water drops or camera motion.
- A model trained with pasted backgrounds (`V1`) can learn the paste edges instead of the leaf, so its gain on the composites may overstate real-world gain. Always report `V0` alongside `V1`.
- An optional **real-world check** (E10, stretch) with 30 to 40 hand-labelled phone photos would address this; it is not part of the core plan.

**Done when.** The four levels are built and cached, the tests pass, and E5 is reported for `V0` and `V1`.

---

### M8. Reliability: calibration, conformal prediction, abstention

**Purpose.** Let the system say "I am not sure", and test whether a small labelled slice of the target dataset makes that trustworthy.

**Inputs.** Saved probabilities from M5, severity predictions from M6.

**Temperature scaling (`reliability/calibrate.py`).** Fit a scalar `T` on source-validation logits by minimising NLL (`scipy.optimize.minimize_scalar`), then `p = softmax(logits / T)`, applied on the restricted label space.

**Split conformal prediction (`reliability/conformal.py`).** Use the simple LAC score (this routine was checked on synthetic exchangeable data and gave about 90% coverage at alpha 0.1).

```python
import numpy as np

def conformal_qhat(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> float:
    n = len(cal_labels)
    scores = 1.0 - cal_probs[np.arange(n), cal_labels]
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:                      # too few calibration points for this alpha
        return np.inf              # => sets contain every class
    return float(np.sort(scores)[k - 1])

def prediction_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    return probs >= 1.0 - qhat     # boolean (N, C); sets can be empty or contain several classes
```

- `alpha = 0.10` (target coverage 90%) by default; config-driven.
- **Abstain rule (`abstain.py`).** Answer only if the prediction set has exactly one class. Otherwise abstain. Optional second rule: flag "borderline" when predicted severity is within `delta` percentage points of a decision threshold in the KB (default `delta = 2`, tuned on calibration data only).

**Experiments.** For `T1` (test = `seg_test`) and `T2` (test = its 70% test split), with the best classifier from M5:

| Method | Calibration data |
| --- | --- |
| MSP baseline | Reject if max probability < τ, with τ chosen to match the same coverage as conformal |
| Conformal, source-only | `S/val` |
| Conformal, target-calibrated | Random subsets from the target calibration pool (`T1` outside `seg_test`, n ∈ {25, 50, 100, 150}; `T2/cal_pool`, n ∈ {25, 50, 100}) |

For the target-calibrated rows, **repeat with 100 random draws** and report the mean and spread of every metric. Draw by `group_id` so near-duplicates never straddle calibration and test.

**Metrics (`reliability/metrics.py`).** ECE (15 equal-width bins), Brier score, empirical coverage versus the 90% target, average set size, abstention rate, selective accuracy, selective risk at 80% coverage, AURC.

**Honest framing.** Conformal coverage is guaranteed only when calibration and test are exchangeable. Source-only calibration is expected to under-cover on a shifted target; whether a few dozen target images fix that is the empirical question. Report it whichever way it comes out. Note that "target" here is a public dataset slice, not a deployment distribution.

**Done when.** The comparison table across the three methods exists with spread from repeated draws, and `abstain.py` exposes `decide(probs, qhat, severity=None, thresholds=None) -> Decision`.

---

### M9. Advisory knowledge base and engine

**Purpose.** Turn (disease, severity) into an action that traces to TNAU or CIBRC, in English, with no unverifiable number.

#### Knowledge base (`advisory/kb/*.json`)

One file per disease: **blast, brown spot, bacterial leaf blight, tungro**. **Every value is typed in by the developer from official pages, with source, edition and page.** Claude Code must not fill in doses, thresholds or product names from its own memory; leave `null` and ask.

```json
{
  "kb_version": "YYYY-MM-DD",
  "disease_id": "blast",
  "names": {"en": "Blast"},
  "severity_rule": {
    "applies": true,
    "bands": [
      {"id": "low",  "min_pct": 0,    "max_pct": null, "action": "monitor", "origin": "design", "note": null},
      {"id": "act",  "min_pct": null, "max_pct": null, "action": "spray",   "origin": "tnau", "source_ref": "S1"}
    ]
  },
  "recommendations": [
    {
      "active_ingredient": null,
      "dose_per_litre": null,
      "dose_per_ha": null,
      "waiting_period_days": null,
      "cibrc_registered_for_paddy": null,
      "source_ref": "S1",
      "conflicts_with": []
    }
  ],
  "cultural_notes": [{"text": null, "source_ref": "S1"}],
  "sources": {
    "S1": {"name": "TNAU Agritech expert system", "url": "http://www.agritech.tnau.ac.in/expert_system/paddy/cpdisblast.html", "retrieved": "YYYY-MM-DD", "edition": null}
  },
  "expert_verified": false
}
```

Rules enforced by `advisory/schema.py` (pydantic) and by tests:

- A recommendation is **usable only if** `cibrc_registered_for_paddy` is `true` and `source_ref` resolves.
- If TNAU and CIBRC disagree (different editions can list different doses), store both entries and put each other's id in `conflicts_with`. The engine shows the TNAU state recommendation and appends "dose differs between TNAU and CIBRC; confirm with the extension officer".
- `origin` of each severity band is `tnau` or `design`. The app and the report show it.
- A disease with no published severity rule (tungro) sets `severity_rule.applies = false`; the engine falls back to a non-severity action ("manage" with the KB's cultural note, or "refer").
- `expert_verified` stays `false` unless someone qualified has reviewed the file. The app and report show "not expert-verified" whenever it is `false`.

#### Engine (`advisory/engine.py`)

```python
@dataclass
class Advice:
    disease: str
    severity_pct: float | None
    band: str | None
    action: Literal['monitor', 'spray', 'manage', 'refer']
    products: list[dict]          # straight from the KB, with source_ref
    notes: list[str]
    sources: list[dict]
    expert_verified: bool
    text: str                     # rendered from the English template

def advise(decision: Decision, severity: float | None, kb: KB) -> Advice: ...
```

Procedure:

1. If `decision.abstain` → `action='refer'`, reason `low_confidence`, no products.
2. If disease is `normal` → `action='monitor'`.
2b. If the disease has no KB file (any class other than blast, brown spot, bacterial leaf blight, tungro, normal) → `action='refer'`, reason `no_advice_for_class`, no products. The rendered text comes from `templates/en.yaml` ("no advice available, please consult an officer"); it is not written in Python.
3. If the disease's `severity_rule.applies` and severity is known: find the band; if the decision is `borderline` (M8 delta rule) → `action='refer'`, reason `near_threshold`.
4. Else use the disease's non-severity default from the KB.
5. Attach usable products (filtered by the CIBRC flag), notes and sources.
6. Render text from `templates/en.yaml` using Jinja2 with only **KB-derived fields as variables**. No free-form generation.

#### Compliance checker (`advisory/checker.py`)

`check(advice_text: str, kb) -> list[Violation]` extracts every active-ingredient and dose pattern from rendered text and verifies it exists in a KB recommendation with a source. A unit test renders advice for every (disease × band) combination and asserts **zero violations**. The same test also asserts **source attribution**: every product line and every severity-band statement in the rendered text carries its `source_ref` (and the band's `origin`), so "source attribution present" is measured, not assumed. This test is the "100% dose compliance" claim in the report.

**Done when.** KB files for blast, brown spot and bacterial leaf blight are filled and validated; tungro is handled as `applies=false` with its management note; engine tests and the compliance test pass.

---

### M10. Decision-level evaluation

**Purpose.** Measure what matters to a farmer: did the system reach the right action?

**Which images.** `T1/seg_test` (masks give the true severity) at level `S0_clean`, and the same images at levels `S1` to `S3` from M7. Because compositing preserves the true disease and true severity, the **reference decision is the same at every level**, so the effect of shift on decisions is isolated. Calibration subsets for conformal come from `T1` images outside `seg_test`. `T2` has no masks, so it appears only in M5 and M8.

**Reference decision.** `action_true = rule(true_disease, true_severity)` using the same KB rules.

**System decision.** `action_sys = advise(predicted disease → conformal gate → predicted severity)` end to end, using the best classifier and the `V0` or `V1` segmenter.

**Metrics (`evaluation/decision_eval.py`).**

| Metric | Meaning |
| --- | --- |
| Spray-decision agreement | `action_sys == action_true` over all images |
| Selective agreement | Agreement over images where the system answered, reported with the abstention rate |
| Missed treatment rate | `action_true = spray` but system says `monitor` (the dangerous error) |
| Unnecessary spray rate | `action_true = monitor` but system says `spray` |
| Decision-flip rate | Share of images where predicted severity changes the action compared with the true-severity action (same true disease), per shift level. This is the headline severity-under-shift number |
| Decision-flip decomposition | Agreement with (true disease + predicted severity), (predicted disease + true severity), and both predicted: shows whether classification or severity error causes the flips |
| Dose compliance | Output of the M9 compliance test (target 100%) |

**Baselines.**

- B1: "disease name → fixed spray" (spray whenever a disease is predicted, ignoring severity).
- B2: full system without abstention.
- B3: full system with source-only conformal.
- Full system: target-calibrated conformal.

**Outputs.** `reports/results/decision.csv` (rows: level × method), confusion matrix over `{monitor, spray, manage, refer}`, and a plot of agreement versus shift level.

**Caution.** The ground truth is rule-derived from masks, not an agronomist's field verdict, and the shifted levels are simulated. State both in the report.

**Done when.** The decision table with baselines and the flip decomposition exists for all four levels.

---

### M11. Demo application

**Purpose.** A small interface proving the pipeline end to end.

**`app/streamlit_app.py`.**

- Input: photo upload (or camera).
- Steps: classify over **all 10 Paddy Doctor classes** (no restriction in the app) → conformal gate with the stored `qhat` → segment → severity → `advise`. Only five classes get advice (blast, brown spot, bacterial leaf blight, tungro as a management note only, and normal). If the predicted class is any other (hispa, dead heart, bacterial leaf streak, bacterial panicle blight, downy mildew), the app shows "no advice available, please consult an officer". Restricting the model to five classes would force, for example, a hispa leaf into the wrong category, which is the unsafe behaviour this project exists to avoid.
- Output panel: predicted disease and confidence set, severity overlay image, severity percent and band (with `origin`), action, products with source references, notes, and an explicit banner when the system abstains.
- Footer: "Decision-support prototype. Not expert-verified. Confirm with your agriculture officer before spraying."
- Models: export the chosen models to `models/` (state dicts plus a small JSON with class names and `qhat`). CPU inference is fast enough for a demo. ONNX export is optional.
- Hosting: run locally for the review; optionally deploy on a free Streamlit or Hugging Face Spaces instance (models are small).

**Tests.** A smoke test that runs `advise` on three bundled sample images and checks the output schema.

**Done when.** The app runs from a clean checkout with `streamlit run src/paddyguard/app/streamlit_app.py` and shows an abstention case and an answered case.

---

### M12. Reporting and statistics

**Purpose.** Every number in the report regenerates from saved predictions with one command.

- `evaluation/tables.py` reads `runs/*/metrics.json` and `preds_*.parquet` and writes `reports/results/*.csv` and `reports/tables/*.md`.
- `evaluation/figures.py` produces: cross-dataset gap bars, per-class recall, severity scatter (true vs predicted) per shift level, MAE-versus-shift-level curves for `V0` and `V1`, risk-coverage curves, coverage-versus-calibration-size plot, decision agreement versus shift level, decision confusion matrix, example composites.
- `evaluation/stats.py`: bootstrap CIs over images and paired seed-wise differences. Wilcoxon tests only when n is large enough.
- Keep `reports/experiment_log.md`: date, run id, what changed, one-line result. This becomes the methods and results chapters.
- Report text the code cannot produce but the report must contain: the limits statements (simulated shift, rule-derived ground truth, small `T2`, not expert-verified), the dataset licences (`dataset_card.md`), and the positioning against Amato 2026 and Paul 2026 (cite them up front).

---

## 5. Experiment index

| ID | Experiment | Module | Output |
| --- | --- | --- | --- |
| E1 | In-domain baseline (EfficientNet-B0) | M4 | `metrics.json`, per-variety table |
| E2 | Zero-shot cross-dataset (`S→T1`, `S→T2`) | M5 | `cross_dataset.csv` |
| E3 | Robustness ablation (baseline, strong, strong + MixStyle; 3 seeds) | M5 | Ablation table with CIs |
| E4 | Severity on `seg_test` (`V0`, `V1`) | M6 | Dice/IoU/MAE |
| E5 | Severity under simulated shift, levels S0 to S3 (`V0`, `V1`) | M7 | MAE-versus-level table and curve |
| E6 | Abstention: MSP vs source conformal vs target conformal | M8 | Coverage/size/risk table |
| E7 | End-to-end decision agreement, clean and shifted, with baselines | M10 | `decision.csv` |
| E8 (stretch) | Few-shot adaptation 5/10/25% | M5 | Curve |
| E9 (stretch) | Classifier on composites; second backbone | M5, M7 | Table |
| E10 (stretch) | Real-world check with 30 to 40 labelled phone photos | M6, M7 | Severity error on real photos |

---

## 6. Interface contracts

**Prediction files** (`runs/<id>/preds_<domain>.parquet`): `image_id: str`, `y_true: str`, `y_pred: str`, `logit_<class>: float32` for every training class, `domain: str`, `split: str`. Anything that restricts the label space does so at read time with `restrict_logits`.

**Severity files** (`runs/<id>/sev_<domain>_<level>.parquet`): `image_id`, `level`, `variant` (`V0|V1`), `sev_pred_pct`, `sev_true_pct`, `leaf_px_pred`, `lesion_px_pred`.

**Config discipline.** `configs/*.yaml` are loaded once into a dataclass; the full resolved config is written to the run directory. A table row in `reports/` must always be traceable to a run directory.

---

## 7. Risks specific to the build

| Risk | Early signal | Response |
| --- | --- | --- |
| RiceSeg mask encoding is not what the page implies | `inspect_masks.py` overlays look wrong | Resolve in week 1; if masks cannot support lesion percent, estimate lesions by colour thresholding on Sethy's plain-background images, and say so |
| Simulated shift is dismissed as unrealistic | Reviewer comment, or `V1` gains vanish on a few real photos | Keep the wording "simulated", report `V0` and `V1`, show example composites, and run E10 if time allows |
| Segmenter learns paste artifacts | `V1` improves on composites but not on `S0_clean` | Report both; add feathering and colour matching to the compositor; treat gains as an upper bound |
| Kaggle competition data licence or rules prevent use | Rules page forbids non-competition use | Use the Hugging Face mirror, or ask the guide about the IEEE DataPort copy |
| GPU quota runs out | Quota warning | Skip optional backbones first; cut to 3 seeds on EfficientNet-B0 only |
| TNAU page numbers cannot be read or differ from CIBRC | Conflicts when filling the KB | Store both with sources, flag the conflict in output, never pick silently |
| Knowledge base is not reviewed by an expert | `expert_verified: false` | Show "not expert-verified" in demo and report; ask a TNAU student or officer to glance at it if possible |
| Time overrun | Milestone slips by more than 2 weeks | Cut order: E8, E9, E10, MobileNetV3/ConvNeXt, MixStyle variant. Never cut E2, E5, E6, E7 or the compliance test |

---

## 8. Week-1 discovery checklist (do before writing much code)

- [ ] Join the Kaggle competition; confirm the class list and `train.csv` columns.
- [ ] Download Sethy, RiceSeg, Dhan-Shomadhan; upload each as a private Kaggle dataset; list the real folder names and fill `labelmap.yaml`.
- [ ] Run `inspect_masks.py`; look at the overlays yourself.
- [ ] Check the licence on every dataset page; note it in `reports/dataset_card.md`.
- [ ] Open the TNAU pages for blast, brown spot, bacterial leaf blight and tungro and note thresholds and sources; download the CIBRC PDFs (31.03.2024 edition).
- [ ] Time one EfficientNet-B0 epoch on Kaggle to replace the estimates in section 3.
- [ ] Non-code items from the Project Guide checklist: confirm the Amato et al. volume and article number on the Elsevier page; remove "SOFS-Rice" from the literature table unless the PDF is held; search Google Scholar for "Paddy Doctor" with "Sethy" before Review 1; ask the guide about an IEEE DataPort subscription (optional); ask a TNAU student or agriculture officer to glance at the knowledge base (otherwise it stays "not expert-verified").

---

## 9. Definition of done for the whole project

Minimum viable result (must exist):

1. E2 cross-dataset table with CIs and the per-class view.
2. E4 and E5 severity results across the four simulated-shift levels, for `V0` and `V1`.
3. E6 abstention comparison with repeated calibration draws.
4. E7 decision table with baselines and the flip decomposition, clean and shifted.
5. Advisory KB for three diseases (plus tungro as management-only), passing the compliance test.
6. Streamlit demo showing an answered and an abstained case.
7. `reports/experiment_log.md` and regenerable tables and figures.

Stretch, in order: E8 few-shot, per-variety write-up, a second backbone, E10 real-world check, ONNX export.
