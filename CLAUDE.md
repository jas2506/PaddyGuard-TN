# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Current state

**Start every new session by reading `PROGRESS.md`** (what M0-M4 did, every decision and measured number, the agreed M5 plan) and `reports/experiment_log.md`. The developer prefers minimal involvement: decide from the data, look at images yourself, record the decision, and ask only when a value is genuinely unknown.

M0 (utilities), M1 (manifest, dataset card, mask inspection), M2 (duplicates, splits, cache), M3 (label map, datasets, transforms) and M4 (classifier training) are implemented. M0-M3 have been run on the real data (`data/manifests/manifest.csv` has `phash`, `group_id`, `split` and the harmonised `label`; `data/processed/` is built). M4 seed 0 (baseline augmentation) has been run on Kaggle and passed acceptance: S/test macro-F1 0.9202, balanced accuracy 0.9163, lowest class recall brown_spot 0.84 (`runs/cls_effb0-0-20261019-123812/`, see `reports/experiment_log.md`); seeds 1 and 2 and the other variants are still to run. GPU runs happen on Kaggle (`docs/kaggle_guide.md`, `notebooks/kaggle_run.ipynb`, `scripts/package_for_kaggle.sh`) and their run folders (`metrics.json`, `preds_S_test.parquet`, `ckpt.pt`) are copied into `runs/`. Every other module under `src/paddyguard/` is still a docstring-only stub. Build modules in the spec's order (M5, then M6 → M7, M8, M9, M10, M11; M12 throughout) and don't start a module before its dependencies are done. Splits are fixed: `dedup`, `splits` and `manifest` refuse to overwrite them without `--force`; do not re-split.

- `docs/PaddyGuard-TN_Project_Guide.docx` is the companion problem statement/literature/datasets doc (summarised below).
- `configs/` holds only values stated in the spec. `labelmap.yaml` `T1`/`T2` maps and `advisory.yaml` are deliberately unfilled; `advisory/kb/*.json` do not exist yet (values must be typed in by the developer).
- Git repo: https://github.com/jas2506/PaddyGuard-TN (branch `main`). `.gitignore` excludes `data/`, `runs/`, `models/`, checkpoints and parquet files.
- `advisory/kb.py` (module) and `advisory/kb/` (JSON directory) coexist as in the spec; `kb/` has no `__init__.py`.

Commands. Use the project venv `.venv` (gitignored, created with `--system-site-packages`, plus pyarrow and ruff). Run scripts as `PYTHONPATH=src .venv/bin/python -m paddyguard.<module>`; `pytest` adds `src` itself (`pythonpath = ["src"]` in pyproject.toml). Quirk: on this Mac, `.pth` files in the venv's site-packages (the `__editable__` one from `pip install -e .` and any hand-made one) get the macOS `hidden` flag and Python ignores them, so don't rely on them. M1 commands: `python -m paddyguard.data.download` (layout check), `python -m paddyguard.data.manifest [--debug]` (writes `data/manifests/manifest.csv` and `reports/dataset_card.md`), `python -m paddyguard.data.inspect_masks [--debug]`; M2, in this order: `python -m paddyguard.data.dedup`, `.splits`, `.cache` (each takes `--debug`; debug runs use `manifest_debug.csv` and `data/processed_debug/`). M3: `python -m paddyguard.data.labelmap` fills the manifest `label` column (already run). Run `ruff format src tests` after writing code.

```
pip install -e .
.venv/bin/python -m pytest -q   # single test: ... -q tests/test_x.py::test_name
.venv/bin/ruff check .
python -m paddyguard.classify.train --config configs/cls_effb0.yaml --seed 0 [--debug]
streamlit run src/paddyguard/app/streamlit_app.py
```

Training runs on Kaggle/Colab GPUs (the laptop has none); `notebooks/kaggle_run.ipynb` only clones the repo and calls these scripts. Checkpoint every epoch to `$PADDYGUARD_OUT/runs/<run_id>/ckpt.pt` and support `--resume`.

## Project context (from docs/PaddyGuard-TN_Project_Guide.docx)

Research question: can a model trained on Tamil Nadu photos (Paddy Doctor) and tested on other datasets measure leaf damage accurately enough to trigger TNAU spray rules correctly, and abstain when it cannot? Public data only; nothing is photographed or labelled by the developer. The claimed contributions are the Paddy Doctor → Sethy benchmark (with per-variety analysis), severity error under the *simulated* shift reported as "decision flips" (how often a severity error changes the TNAU action), and target-calibrated abstention scored by spray-decision agreement with a registry-checked advisory. Prior cross-dataset work (Amato 2026, Paul 2026) is Bangladesh-only, so cite it up front; cross-dataset test + MixStyle + Grad-CAM alone is no longer novel.

- Excluded scope: Tamil text, soil/weather modifiers, LLM chat, panicle/neck blast, own photos. Tungro is classification plus a management note only.
- Only the ~5% leaf-area blast threshold has been found, and the guide says it is unconfirmed; every other band edge is a design choice. TNAU and CIBRC (31.03.2024 edition) doses can disagree: store both with edition and show the TNAU value with the conflict flagged.
- Guide's dataset counts, all matching `data/raw`: Paddy Doctor 10,407 labelled + 3,469 unlabelled test; Sethy and RiceSeg 5,932; Dhan-Shomadhan 1,106 (337 field background, 769 white background).
- Licences: the developer reports CC BY 4.0 for all four datasets (text in `configs/data.yaml` `licences:` and `reports/dataset_card.md`, 2026-10-19). CC BY needs credit, a link to the licence and a note of changes in the report. Paddy Doctor is a Kaggle competition, so also read its competition rules page for reuse limits.
- Decided 2026-10-19: `T2` main target is the 337 field-background Dhan images (40% `cal_pool` / 60% `test`, stratified by class, by group, **after dedup**); the 769 white-background images are the zero-shot-only row `T2-white`. Only the 199 images in the three shared classes can calibrate the gate. `T2` calibration sizes are the values in {25, 50, 100} at most 65% of the pool (about 35% stays unused: pool of ~80 allows {25, 50}; 100 needs a pool of about 154) (`T1` keeps {25, 50, 100, 150}). Brown spot is expected to be low-n: keep it, report with bootstrap CIs, flag low-n in every `T2` table. Fallback (only if field counts are unusable): split all shared-class Dhan images stratified by background and report field and white separately. Frame `T1` as the main benchmark and `T2` as small-sample corroboration. The `labelmap.yaml` `T2` mapping (`Browon Spot`→brown_spot, `Rice Blast`→blast, `Rice Turgro`→tungro) was confirmed by the developer from a montage on 2026-10-19 and is written; the white-background folders are mapped by name only, and `Leaf Scaled`/`Sheath Blight` map to null (no Paddy Doctor counterpart). See `reports/experiment_log.md`.

## Architecture (big picture)

Two models run on every photo: a disease classifier (timm EfficientNet-B0) and a leaf+lesion U-Net segmenter that yields severity (% of leaf area). A conformal gate (M8) decides answer vs abstain, then a **rule-based** advisory engine (M9) maps (disease, severity) to an action from a JSON knowledge base built from TNAU and CIBRC. There is no generative text anywhere in the core path.

Datasets are referred to by domain code throughout code, configs and tables:
- `S` Paddy Doctor (Tamil Nadu): classifier train/val/test, plus the pool of field backgrounds for the shift composites.
- `T1` Sethy (Odisha) with RiceSeg masks: zero-shot classifier target, segmenter train/val/test, and source of leaves for composites.
- `T2` Dhan-Shomadhan (Bangladesh), field-background images only: second zero-shot target; no masks, so it appears only in classification and abstention (M5, M8). The white-background images are the separate zero-shot-only row `T2-white`.
- `SX` simulated-shift composites of `T1` leaves on blurred `S` backgrounds, at levels S0_clean to S3_strong.

Things that span several modules and are easy to get wrong:
- **Splits are fixed (seed 42, never re-split) and leakage-controlled.** pHash near-duplicates are clustered into `group_id`s within each dataset; splits are grouped by `group_id`; any `T1`/`T2` image near-duplicate to an `S` image is dropped. `T1` has no classifier training split; its `seg_test` is the evaluation set for severity, conformal and decision evaluation, and conformal calibration subsets come only from `T1` outside `seg_test` (or `T2/cal_pool`). Assert in code that target domains never enter a training loader.
- **Label spaces differ per dataset.** The classifier trains on all source classes; for cross-dataset eval, `restrict_logits` keeps only the shared-class columns before softmax. In-domain scores used for the "gap" must also be computed on the same restricted classes. `configs/labelmap.yaml` for `T1`/`T2` must be filled from the real folder names, not guessed.
- **Prediction/severity files are the interface between modules** (`runs/<id>/preds_<domain>.parquet`, `sev_<domain>_<level>.parquet`; columns in spec section 6). Downstream calibration, conformal, decision-eval and report code read these rather than rerunning models. Every row in `reports/` must trace to a run directory.
- **Severity doesn't apply to every disease.** Tungro has no severity rule (`severity_rule.applies = false`); band edges are `origin: tnau` or `origin: design` and the origin must be shown in every table and the app. Only the ~5% blast threshold was found, and it must be re-confirmed on the TNAU page.
- **Advisory KB values are typed in by the developer** with source/edition/page. Leave unknown doses, products and thresholds as `null` and ask. A recommendation is usable only if `cibrc_registered_for_paddy` is true and its `source_ref` resolves; TNAU/CIBRC conflicts are stored both ways and flagged, never resolved silently. `expert_verified` stays `false` unless a qualified person reviewed it, and the app/report must say "not expert-verified". The compliance checker test (zero violations across every disease × band) backs the "100% dose compliance" claim.
- **App classifies over all 10 Paddy Doctor classes; only five get advice** (blast, brown spot, bacterial leaf blight, tungro as management note, normal). Any other predicted class (hispa, dead heart, ...) → `refer` with reason `no_advice_for_class` and the text "no advice available, please consult an officer" (from `templates/en.yaml`). Never restrict the app's classifier to five classes.
- **Report `V0` (standard aug) and `V1` (background-paste aug) segmenters together**; `V1` gains on composites may overstate real-world gain.
- Statistics: with 3 seeds report mean ± std and bootstrap 95% CIs over images, no p-values.

## Data layout (`data/`, gitignored)

`archives/` original zips (kept, safe to delete once the data is on Kaggle). `interim/`, `processed/`, `manifests/` are empty and filled by M1/M2. Raw datasets in `raw/`, reorganised from the downloads (only folder levels and two trailing-space folder names changed; no class folder or file was renamed):
- `raw/paddy_doctor/`: `train_images/<class>/` (10 classes: bacterial_leaf_blight, bacterial_leaf_streak, bacterial_panicle_blight, blast, brown_spot, dead_heart, downy_mildew, hispa, normal, tungro), `train.csv` (`image_id,label,variety,age`), `test_images/` (unlabeled; never evaluate on it), `sample_submission.csv`.
- `raw/sethy/images/<Class>/`: `Bacterialblight` 1584, `Blast` 1440, `Brownspot` 1600, `Tungro` 1308 (5,932 total).
- `raw/riceseg/masks/<Class>/`: same four class folders and per-class counts as Sethy (shipped as `mask-vesrion1/`). Filenames carry typos (`BACTERAILBLIGHT3_001`) in both images and masks; all 5,932 stems match exactly (checked 2026-10-19, extensions mix `.jpg`/`.JPG`, so match case-insensitively); M1 must still re-verify and report unmatched files. Sethy images are only ~300×300 px (masks the same size): cache `T1` at native resolution, never downscale. Paddy Doctor is 480×640; Dhan is ~1952×4160.
- `raw/dhan/`: `Dhan-Shomadhan_picture_Information.csv`, `Field Background/` and `White Background/`, each with five class folders whose names carry typos as shipped (e.g. `Browon Spot`, `Rice Turgro`, `Shath Blight`; the two backgrounds spell them differently). Keep raw labels as shipped in `label_raw`; harmonise in `labelmap.yaml`.
- RiceSeg mask encoding (answered by the developer after viewing overlays, 2026-10-19; stored in `configs/data.yaml` `mask_encoding`): white (255) marks **lesions only**, background is pure black (0), and a threshold of 128 removes the JPEG noise. There is no leaf class, so M6 must derive the leaf mask separately (`segment/leafmask.py`, colour distance from the plain background) and validate it by overlay on 30 images.
- Code reads data via `PADDYGUARD_DATA` (point it at `<repo>/data` locally), never a hard-coded path.

## Duplicates and splits (M2 results to remember)

- Duplicate thresholds are per dataset (Hamming, even distances only): S 8, T1 14, T2 4; cross-dataset leakage threshold 8 (no leakage found). Review of the boundary pairs was done visually by the implementer, not the developer (`dedup.review_note`).
- `group_id` is the smallest `image_id` in the group; the representative of a group is the image whose id equals the group id (`dedup.is_representative`). The manifest `split` values are `train/val/test` (S), `seg_train/seg_val/seg_test` (T1), `cal_pool/test` (T2 field), `zero_shot` (T2 white), and `dropped_duplicate`/`dropped_leakage` (excluded everywhere).
- T1 non-representatives are dropped only inside `seg_test`; they stay in seg_train/seg_val. Any evaluation over T1 outside seg_test (M5 zero-shot) must use representatives only.
- T2 shared-class field pool is 79 images (test 119), so T2 calibration sizes are [25, 50]; brown spot has 28 test images and is flagged low-n in every T2 table.
- Measured results (2026-10-19, thresholds above): Paddy Doctor 10,407 images in 6,141 groups (largest 51); Sethy 5,932 in 1,847 groups (largest 25; 5,174 exact-hash pairs, so heavily duplicated: a finding for the report); Dhan 1,106 in 1,099 groups. No leakage from Paddy Doctor into either target.
- Split sizes: S train 7,334 / val 1,587 / test 1,486. T1 seg_train 4,074 / seg_val 927 / seg_test 280 (87 bacterial blight, 66 blast, 79 brown spot, 48 tungro; 651 duplicate members dropped from seg_test only). T2 field cal_pool 135 / test 201 (shared classes only: pool 79 / test 119); T2-white 763 zero_shot; 7 T2 duplicates dropped. Only 1,847 T1 representatives exist for evaluation over all of Sethy.
- Consequences: seg_test has just 280 images, so severity and decision results need bootstrap CIs and a small-sample caveat; T2 results are small-sample corroboration; the segmenter trains on 4,074 T1 images that include many near-duplicates; conformal calibration for T1 draws from the 5,001 non-seg_test T1 images (draw by `group_id`).
- Image facts: Paddy Doctor 480×640; Sethy ~300×300 (about 7% of the `.jpg` files are really PNGs); Dhan ~1952×4160 with EXIF rotation tags (applied when hashing and resizing).
- Processed images: `data/processed/<dataset>/<image_id>.jpg` (S, T2 short side 256; T1 native) and `T1/<image_id>_mask.png` (single channel, same geometry; threshold 128 per `mask_encoding`).

## Loaders and label spaces (M3)

- Classifier output space = `LabelMap.source_classes` = 4 canonical classes then the 6 Paddy Doctor-only classes (10 columns, that order). Cross-dataset evaluation restricts at read time with `labelmap.restrict_logits` then `softmax`; shared sets via `shared_classes(lm, domain)`.
- Domains for `PaddyDataset`: `S`, `T1`, `T2` (field only), `T2-white`. Real sizes: S train 7,334 / val 1,587 / test 1,486; T1 seg_train 4,074 / seg_val 927 / seg_test 280 and 1,847 representatives overall; T2 test 119 / cal_pool 79 on the three shared classes (606 of 1,106 Dhan images have a label, the rest map to null); T2-white 401 shared-class images.
- `make_loader(training=True)` accepts only `S/train` (raises otherwise). Use `representatives_only=True` for any evaluation over all of T1. `with_mask=True` (T1) returns the lesion mask thresholded at `mask_encoding.threshold`; lesions are about 5% of pixels.
- Transform magnitudes beyond the spec (rotation, colour jitter, blur, dropout) are design choices in `configs/transforms.yaml`, not tuned. Loader speed on this laptop: about 400 images/s with `strong`, one process.

## Classifier training (M4)

- `python -m paddyguard.classify.train --config configs/cls_effb0.yaml --seed 0 [--debug] [--no-pretrained] [--resume RUN_DIR]`. Outputs in the run dir: `metrics.json` (with an `acceptance` block: macro-F1 > 0.90 and every class recall >= 0.80, thresholds in the config), `preds_S_test.parquet`, `ckpt.pt` (best weights + class names), `ckpt_last.pt` (resume state).
- Shared metric/prediction helpers live in `classify/evaluate.py` (also used by M5). `--debug` was run on the real data on CPU (about 2 minutes, needs internet for the pretrained weights); its near-chance scores are expected.
- The Kaggle datasets are `paddyguard-processed` (manifest + processed images, from `scripts/package_for_kaggle.sh`) and `paddyguard-code`; the raw Sethy/RiceSeg/Dhan uploads are not used by training.

