# PaddyGuard-TN: progress report, modules M0 to M4

Status date: 2026-10-19. Everything below was done in one working session, from an empty repository
(a spec, a project guide and raw downloads) to a trained and evaluated classifier. Modules M5 to M12
are not started; M5 was deliberately paused so that the project review can show M0 to M4.

Contents

1. [Status at a glance](#1-status-at-a-glance)
2. [The project in brief](#2-the-project-in-brief)
3. [Before M0: reading the documents, organising the repository and the data](#3-before-m0-reading-the-documents-organising-the-repository-and-the-data)
4. [M0: scaffolding and utilities](#4-m0-scaffolding-and-utilities)
5. [M1: data acquisition and manifests](#5-m1-data-acquisition-and-manifests)
6. [M2: duplicates, splits and the cached images](#6-m2-duplicates-splits-and-the-cached-images)
7. [M3: label harmonisation, loaders and transforms](#7-m3-label-harmonisation-loaders-and-transforms)
8. [M4: disease classifier training](#8-m4-disease-classifier-training)
9. [Chronological decision log](#9-chronological-decision-log)
10. [Problems found and how they were fixed](#10-problems-found-and-how-they-were-fixed)
11. [Findings that matter for the review and the report](#11-findings-that-matter-for-the-review-and-the-report)
12. [Open items and what is not done](#12-open-items-and-what-is-not-done)
13. [How to reproduce everything](#13-how-to-reproduce-everything)
14. [Appendix: file inventory, test inventory, numbers](#14-appendix-file-inventory-test-inventory-numbers)

---

## 1. Status at a glance

| Module | What it is | State |
| --- | --- | --- |
| M0 | Seeds, paths, file helpers, run directories, logging | Implemented, tested |
| M1 | Dataset layout check, manifest of all images, dataset card, mask inspection | Implemented, run on the real data |
| M2 | Duplicate detection, duplicate groups, leakage check, splits, cached resized images | Implemented, run on the real data |
| M3 | Label map, PyTorch datasets and loaders, augmentation pipelines | Implemented, run on the real data |
| M4 | EfficientNet-B0 training script, evaluation helpers, Kaggle notebook | Implemented; seed 0 trained on a Kaggle GPU and **passed the acceptance check** |
| M5 to M12 | Cross-dataset evaluation, severity, simulated shift, conformal abstention, advisory, decision evaluation, demo, reporting | Not started (docstring-only stubs) |

Headline result of M4 (EfficientNet-B0, baseline augmentation, seed 0, Paddy Doctor test split of 1,486 images):
**macro-F1 0.9202, balanced accuracy 0.9163, accuracy 0.926**; the acceptance rule (macro-F1 above 0.90 and every
class recall at least 0.80) is met, narrowly (lowest class recall: brown spot, 0.84).

Code quality state: 106 automated tests pass (`pytest -q`), `ruff check .` is clean. The repository is not yet a git
repository (no `git init` has been run), so `git_commit.txt` in every run directory says `unknown`.

---

## 2. The project in brief

PaddyGuard-TN turns a smartphone photo of a paddy (rice) leaf into a traceable spray decision for Tamil Nadu
farmers. It names the disease, measures how much of the leaf is damaged, decides whether it is confident enough to
answer, and then gives the action recommended by Tamil Nadu Agricultural University (TNAU), in English, with every
chemical and dose checked against TNAU and the CIBRC pesticide registry.

The research question (from the project guide): can a paddy disease model trained on Tamil Nadu photos and tested on
other datasets measure leaf damage accurately enough to trigger TNAU's spray rules correctly, and say "I am not sure"
when it cannot? Public data only; nothing is photographed or labelled by the developer.

Claimed contributions: (1) the first Paddy Doctor to Sethy (Indian to Indian) cross-dataset benchmark with per-variety
analysis; (2) severity error under a controlled **simulated** shift, reported as "decision flips"; (3) target-calibrated
abstention scored by spray-decision agreement, with a registry-checked advisory. Prior cross-dataset studies
(Amato et al. 2026, Paul 2026) use Bangladeshi data only and should be cited up front; plain "cross-dataset test plus
MixStyle plus Grad-CAM" is no longer novel on its own.

Constraints: one developer; no useful GPU on the laptop (training runs on Kaggle); fine-tuning of small pretrained models
only; English only; three diseases plus tungro as a management note.

Datasets and the codes used throughout code, configs and tables:

| Code | Dataset | Size | Role |
| --- | --- | --- | --- |
| `S` | Paddy Doctor, Kaggle competition version (Tamil Nadu) | 10,407 labelled photos, 10 classes, plus variety and age; 3,469 unlabelled test photos that are never used | Training, validation, in-domain test; field backgrounds for the simulated shift |
| `T1` | Sethy et al. (Odisha) | 5,932 photos, 4 classes, about 300x300 px | Zero-shot cross-dataset target; leaves and masks for severity |
| (RiceSeg) | RiceSeg-5932 | 5,932 pixel masks, one per Sethy photo | Lesion masks for severity |
| `T2` | Dhan-Shomadhan (Bangladesh) | 1,106 photos, 5 classes: 337 field background, 769 white background | Second zero-shot target |
| `SX` | Simulated-shift composites of T1 leaves on blurred S backgrounds | built in M7 | Not yet built |

Paddy Doctor classes (10): bacterial_leaf_blight, bacterial_leaf_streak, bacterial_panicle_blight, blast, brown_spot,
dead_heart, downy_mildew, hispa, tungro, normal.

---

## 3. Before M0: reading the documents, organising the repository and the data

### 3.1 The two source documents

- `PaddyGuard-TN_Architecture.md` (the technical spec, 764 lines at the start): modules M0 to M12, build order,
  interfaces, acceptance checks, experiment index E1 to E10, risks, week-1 checklist. It is the source of truth and
  was renamed to `ARCHITECTURE.md` (see 3.3).
- `PaddyGuard-TN Project Guide.docx`: problem statement, literature table, datasets and download sources,
  objectives and metrics, scope and risks, timeline (Oct 2026 to Jan 2027), checklist, references. It was moved to
  `docs/PaddyGuard-TN_Project_Guide.docx`. Its text was extracted and read in full.
- A local instructions file (git-ignored, see 3.6) holds the working rules: Python 3.10+, type hints, small functions;
  everything config-driven; fixed seeds logged with the config and git commit; data paths only through the environment
  variables `PADDYGUARD_DATA` and `PADDYGUARD_OUT`; every script supports `--debug`; save per-image predictions;
  never evaluate on the unlabelled Kaggle test set; never tune on a target-domain test split; no pesticide names,
  doses or thresholds in Python; call the shift experiment "simulated" everywhere; do not invent facts (stop and ask);
  run `pytest -q` and `ruff check .` before declaring a task done.

### 3.2 Coverage analysis: do modules M0 to M12 cover the guide?

A full comparison of the guide's objectives, metrics, datasets, scope, risks and checklist against the modules
found that every requirement is covered (classifier and gap: M4/M5; severity and simulated shift: M6/M7; abstention:
M8; advisory and dose compliance: M9; decision evaluation: M10; demo: M11; reporting: M12; licences, duplicates, Kaggle
workflow, risks: M1/M2/section 3/section 7). No module was missing. The analysis did find seven things to correct in
`ARCHITECTURE.md`, and one real data problem:

1. M1: add a `background` (field/white) column for Dhan; match mask stems ignoring the extension case (the files mix
   `.jpg` and `.JPG`).
2. M2: the spec cached everything at short side 256, but Sethy images are only about 300 px, so downscaling would
   destroy lesion detail; Sethy is cached at native size, with its mask beside it.
3. M5: add `T2-white` as a zero-shot-only secondary row.
4. M6: remove the "try 512 px" suggestion (Sethy is about 300 px, so 512 adds no information).
5. M9: the compliance test must also assert source attribution (every product line carries its `source_ref`, every
   severity band its `origin`), because the guide measures "source attribution present".
6. M10: add an explicit decision-flip rate (the guide calls it the headline severity-under-shift number).
7. M12 and the week-1 checklist: add the non-code report items and the guide's checklist items (confirm the Amato
   et al. volume and article number, remove "SOFS-Rice" from the literature table unless the PDF is held, search Google
   Scholar for "Paddy Doctor" with "Sethy", ask the guide about an IEEE DataPort subscription, ask a TNAU student or
   officer to glance at the knowledge base).

Facts measured on the real data during this analysis: all 5,932 Sethy image stems match the 5,932 RiceSeg mask stems
exactly (the filename typos such as `BACTERAILBLIGHT3_001` appear in both), images and masks have identical sizes on
60 sampled pairs, Sethy images are about 300x300, Paddy Doctor images are 480x640, Dhan images are about 1952x4160.

### 3.3 Repository reorganisation

- `PaddyGuard-TN_Architecture.md` renamed to `ARCHITECTURE.md` (the rules refer to it by that name).
- `PaddyGuard-TN Project Guide.docx` moved to `docs/PaddyGuard-TN_Project_Guide.docx`.
- Created the skeleton from the spec's layout: `src/paddyguard/{data,classify,segment,shift,reliability,advisory,
  evaluation,app,utils}/` with an `__init__.py` per package and a docstring-only stub for every module file in the
  spec; `advisory/kb/` and `advisory/templates/` (with `.gitkeep`); `configs/`, `tests/`, `runs/`, `reports/{results,
  figures,tables}/`, `notebooks/`, `docs/`, and `data/{raw,interim,processed,manifests}/`.
- Note: the spec has both `advisory/kb.py` (a module) and `advisory/kb/` (a directory of JSON files). Both exist as in
  the spec; `kb/` has no `__init__.py`.
- Packaging: `pyproject.toml` (package `paddyguard`, setuptools, `src` layout, Python 3.10+, dependencies read
  dynamically from `requirements.txt`, ruff line length 100, pytest `testpaths` and `pythonpath = ["src"]`);
  `requirements.txt` (the spec's list, later `numpy>=2.0` and `albumentations>=2.0`).
- `.gitignore`: `data/`, `runs/`, `models/`, `*.pt`, `*.pth`, `*.ckpt`, `*.parquet`, `.env`, Python and tool caches, `.DS_Store`,
  and the local assistant-instruction files.
- Configs created, containing only values stated in the spec: `data.yaml`, `labelmap.yaml`, `cls_effb0.yaml`,
  `seg_unet_r34.yaml`, `shift.yaml`, `reliability.yaml`, `severity_bands.yaml` (default edges [5, 25] labelled a design
  choice; the only TNAU threshold found is about 5% leaf area for blast and it must be re-confirmed), `advisory.yaml`
  (placeholder). `transforms.yaml` was added in M3.
- The knowledge-base JSON files were deliberately **not** created: the spec requires every dose, threshold and product
  name to be typed in by the developer from TNAU/CIBRC with source, edition and page.

### 3.4 Raw data reorganisation (done once, no image or class folder renamed)

| Before | After |
| --- | --- |
| `raw/paddy_doctor/paddy-disease-classification/...` | `raw/paddy_doctor/{train_images,test_images,train.csv,sample_submission.csv}` (flattened) |
| `raw/sethy/Rice Leaf Disease Images/<Class>/` | `raw/sethy/images/<Class>/` |
| `raw/riceseg/RiceSeg-5932 ... Mas/Rice_Leaf_Disease_Segmentation_Masks.zip` (zip inside a zip, not yet extracted) | extracted to `raw/riceseg/masks/<Class>/` (the shipped folder was `mask-vesrion1`) |
| `raw/dhan/Dhan-Shomadhan/{"Field Background  ", "White Background "}` (trailing spaces) | `raw/dhan/{Field Background, White Background}` plus the info CSV |
| the five original zips (about 4 GB) | kept in `data/archives/` under descriptive names, not deleted |
| `.DS_Store` files | deleted |

Verified after the move: Sethy 1,584 + 1,440 + 1,600 + 1,308 = 5,932 images and the same per-class mask counts;
Paddy Doctor 10,407 labelled and 3,469 test images; Dhan field 49 + 74 + 74 + 76 + 64 = 337, white 90 + 143 + 198 +
119 + 219 = 769. Class folder names are kept as shipped, including the typos (`Browon Spot`, `Rice Turgro`,
`Shath Blight`, `Leaf Scaled`), because the spec says `label_raw` is the label as shipped.

### 3.5 Project decisions made before M0 (with the reasons)

- **Dhan (T2) main target is the 337 field-background images**, split 40% `cal_pool` / 60% `test`, stratified by
  class and by duplicate group, **after** dedup. The 769 white-background images are a separate zero-shot-only
  secondary row, `T2-white`, with no calibration pool. Reason: Paddy Doctor is field photography, so field-versus-
  field is the cleaner measure of dataset shift; the white-background images resemble Sethy's plain-background leaves.
  The guide said "use the field-background images for the main test" while the spec split all of T2 30/70; this
  reconciles them.
- **Only the three shared classes can calibrate the conformal gate.** Leaf Scaled and Sheath Blight have no Paddy
  Doctor counterpart; of the 337 field images only 199 (before dedup) are in the shared classes.
- **T2 calibration sizes** are the values in {25, 50, 100} that are at most **65%** of the shared-class calibration
  pool (about 35% of the pool stays unused so repeated random draws differ). The first version of the rule was 60%/40%
  and contradicted an example; it was loosened to 65%. T1 keeps {25, 50, 100, 150}.
- **If a T2 class has fewer than about 30 test images** (expected: brown spot) it is kept, reported with bootstrap
  CIs, and flagged low-n in every T2 table. Fallback (only if the field counts become unusable): split all
  shared-class Dhan images stratified by background and report field and white separately. Review framing: T1
  (Sethy) is the main benchmark, T2 is small-sample corroboration.
- **Deployable classes (M11 app):** the classifier predicts across all 10 Paddy Doctor classes; only five get advice
  (blast, brown spot, bacterial leaf blight, tungro as a management note, normal). Any other predicted class (hispa,
  dead heart, ...) gives action `refer` with reason `no_advice_for_class` and the text "no advice available, please
  consult an officer" from the template file, never from Python. Restricting the model to five classes would force a
  hispa leaf into a wrong class, which is the unsafe behaviour the project exists to avoid. Added to M9 (engine step
  2b) and M11 of `ARCHITECTURE.md`. Open consequence: the app's conformal gate then has to be calibrated in the
  10-class space (source-only calibration), which differs from the restricted-space calibration used in M8; to be
  decided at M11.
- **No traces of the assistant tooling in the repository:** its instruction files are git-ignored, and no attribution
  lines go into commits. `ARCHITECTURE.md` still mentions the tool in six places (the spec was written to be read by
  it); rewording them is pending the developer's decision.

### 3.6 Environment

- Python 3.12 on macOS. A project virtual environment `.venv` (git-ignored, created with `--system-site-packages`)
  holds `pyarrow`, `ruff`, `imagehash`, `albumentations`, `timm`; torch 2.4, numpy 2.2, pandas 2.2, scikit-learn,
  Pillow came from the system Python.
- A macOS quirk: `.pth` files in the venv's site-packages (the editable-install one written by `pip install -e .`
  and any hand-made one) repeatedly receive the macOS `hidden` flag and Python then ignores them, so
  `import paddyguard` outside pytest does not work from them. The reliable workaround is `PYTHONPATH=src`; pytest uses
  `pythonpath = ["src"]`.
- Style: `ruff format src tests` after writing code; the lint configuration in use is stricter than the default and
  all code passes it.

---

## 4. M0: scaffolding and utilities

**Purpose.** A reproducible skeleton so every later module seeds, logs and saves the same way.

**Implemented (`src/paddyguard/utils/`).**

- `seed.py`: `set_seed(seed)` seeds Python, NumPy, PyTorch, and all CUDA devices when present. GPU runs are not
  bit-exact; this is to be recorded in the report.
- `paths.py`: `data_root()` reads `PADDYGUARD_DATA` (default `<repo>/data`), `out_root()` reads `PADDYGUARD_OUT`
  (default `<repo>`), `runs_root()` is `out_root()/runs`, plus `repo_root()`, `configs_dir()` and (added in M1)
  `reports_root()`.
- `io.py`: `load_yaml` (raises `TypeError` for a non-mapping document), `save_yaml`, `save_json` (understands NumPy
  scalars/arrays and `Path`), `save_parquet`, `git_commit()` (HEAD hash, `-dirty` suffix when the tree has changes,
  `unknown` outside a git repo), and `new_run_dir(config_name, seed, config=None)`.
- `logging.py`: `get_logger(name, log_file)` writes to console and file; calling it repeatedly adds no duplicate
  handlers.

**Decisions.**

- `new_run_dir` takes an optional `config`. The spec's signature has no config argument but says to write the
  resolved config into the run directory. If `config` is omitted it loads `configs/<config_name>.yaml` and raises
  `FileNotFoundError` if that is missing. The `seed` argument overrides any seed in the config, so `config.yaml`
  always shows the seed actually used. The directory is `runs/<config>-<seed>-<timestamp>/` containing `config.yaml`
  and `git_commit.txt`; a name collision raises instead of overwriting.
- Run directories are created by every script that logs a run, including the data-preparation scripts (M1 to M3), so
  `runs/` holds many `data-0-<timestamp>` folders containing only `config.yaml`, `git_commit.txt` and `log.txt`.
  They are logs of data steps, not training runs.
- Synthetic-image test fixtures were postponed to M1/M2, where they are first needed.

**Tests.** `test_seed` (2), `test_paths` (3), `test_io` (9: YAML round trip and rejection, JSON with NumPy types,
parquet round trip, git commit in a clean/dirty/non-repo directory using a throw-away repository, run directory
contents and naming, default config loading, missing config), `test_logging` (2), `test_import` (1).

**Result.** Done criteria met: the package imports and the tests pass.

---

## 5. M1: data acquisition and manifests

**Purpose.** Turn three differently organised downloads into one uniform table, and find out what the RiceSeg masks
mean before severity work begins.

### 5.1 What was implemented

- `configs/data.yaml` (M1 part): relative dataset roots, expected counts (10,407 / 5,932 / 5,932 / 1,106), source
  strings (Kaggle competition page; Mendeley DOIs 10.17632/fwcj7stb8r.1, 92jc6w6mcy.1, znsxdctwtt.1), accepted image
  extensions (compared case-insensitively), the Dhan folder-to-background map, the spellings of "background" used in
  the Dhan info CSV (`Feild`, `field`, `white`), licences, debug size (200 images per dataset), and the mask-inspection
  parameters.
- `data/download.py`: never downloads (Mendeley has no stable API). `verify_layout` checks that each dataset's
  expected folders and files exist and, if not, says where to get it; `class_counts` and `print_class_counts` print
  images per class folder. CLI: `python -m paddyguard.data.download`.
- `data/manifest.py`: one reader per dataset and `build_manifest`, producing `data/manifests/manifest.csv` with one row
  per image: `image_id` (`S_000000`, `T1_000000`, `T2_000000`, assigned in sorted path order, so deterministic),
  `dataset`, `path` (relative to the data root), `label_raw` (the folder or CSV label as shipped), `label` (empty,
  filled in M3), `variety` and `age_days` (Paddy Doctor only), `width`, `height`, `background` (Dhan only, field or
  white), `mask_path` (Sethy only), and the columns M2 fills: `phash`, `group_id`, `split`.
  - Paddy Doctor reader: `train.csv` (`image_id,label,variety,age`) plus the image in `train_images/<label>/`;
    checks that every CSV row has its image in its label folder and that the CSV covers every image on disk.
  - Sethy reader: pairs each image with its RiceSeg mask by exact filename stem, extension ignored; reports images
    without a mask, masks without an image, stems with several masks, and masks in another class folder.
  - Dhan reader: background from the top-level folder; label from the class folder; cross-checks the info CSV
    (`pictureName`, `Diseases` such as `Rice Blast(Feild Background)`) against the folders for both class and
    background.
  - Expected-count checks per dataset; `--debug` keeps 200 images per dataset and writes `manifest_debug.csv`.
  - `read_manifest` keeps text columns as text (a hex perceptual hash such as `0000000000001234` would otherwise be
    read as a number), `suffixed`, `has_m2_columns` and `write_manifest` helpers.
  - A guard added in M2: running the manifest builder again refuses to overwrite a manifest that already has hashes,
    groups or splits, unless `--force`, because splits are never redone.
- `data/dataset_card.py`: renders `reports/dataset_card.md` (counts per class folder, image-size distribution,
  per-dataset source, licence table, and the result of every check).
- `data/inspect_masks.py`: the week-1 mask discovery. It reads 50 random masks and reports image modes, whether RGB
  masks have identical channels, the number of distinct values after rounding to a multiple of 16, the most common
  rounded values, and a 16-bin intensity histogram; it saves 20 overlay images (photo | mask | 50% blend) in
  `reports/figures/mask_overlays/` and writes `reports/mask_inspection.md`. It deliberately makes **no assumption** about
  what the values mean: the note contains three questions for the developer, and (once answered) the answers are
  rendered from the `mask_encoding` block of `configs/data.yaml`, so regenerating the note never erases them.
- Test fixtures: `tests/conftest.py` builds a tiny synthetic copy of all four dataset layouts (6 Paddy Doctor images,
  5 Sethy images with masks, 7 Dhan images, one `.JPG` extension, a Dhan info CSV) and a version of the real
  `data.yaml` with matching expected counts.

### 5.2 Findings from running M1 on the real data

- Manifest: **17,445 rows** (S 10,407, T1 5,932, T2 1,106). All counts match the guide. All checks passed: every
  Paddy Doctor CSV row has its image and the CSV covers all images on disk; all 5,932 Sethy images pair with a mask
  by stem; the Dhan info CSV agrees with the folders on all 1,106 files.
- Paddy Doctor class counts: bacterial_leaf_blight 479, bacterial_leaf_streak 380, bacterial_panicle_blight 337,
  blast 1,738, brown_spot 965, dead_heart 1,442, downy_mildew 620, hispa 1,594, normal 1,764, tungro 1,088. Image
  sizes: 480x640 (10,403 images) and 640x480 (4).
- Sethy class counts: Bacterialblight 1,584, Blast 1,440, Brownspot 1,600, Tungro 1,308.
- Dhan counts by background and class folder: field (Browon Spot 49, Leaf Scaled 74, Rice Blast 74, Rice Turgro 76,
  Sheath Blight 64) and white (Brown Spot 90, Leaf Scaled 143, Rice Blast 198, Rice Tungro 119, Shath Blight 219).
- Mask pixel facts over 50 sampled masks: every mask is stored as RGB with identical channels (so effectively
  grey); about **94% of pixels are near-black and about 5% near-white**, the rest is scattered JPEG noise; there
  is no third plateau; 15 to 17 distinct rounded values per mask.
- **Developer's answers after viewing the overlays (2026-10-19):** white marks **only lesions**; the background is
  **pure black**; a **threshold of 128** removes the JPEG noise (128 was the example proposed in the question; the
  developer confirmed that a cutoff removes the noise). Stored in `configs/data.yaml` as `mask_encoding`
  (`marks: lesions_only`, `background_value: 0`, `lesion_value: 255`, `threshold: 128`).
  **Consequence for M6:** the masks contain no leaf class, so the leaf mask has to be derived separately
  (`segment/leafmask.py`, colour distance from the plain background, largest connected component, validated by
  overlay on 30 images).
- Licences: the developer reported the same CC BY 4.0 text for all four datasets (credit, link to the licence and a
  note of changes are required). Caveat recorded: Paddy Doctor is a Kaggle competition, so its competition rules page
  should also be read for reuse limits.

### 5.3 Decisions and corrections in M1

- The overlays were first taken from a sample sorted by image id, so they came only from the first classes
  alphabetically. Fixed: overlays are now a random mix across classes (all four classes appeared in the 20), with a
  test.
- Because rerunning the inspection would overwrite hand-written answers, the answers live in the config instead of the
  note.
- The mask threshold is stored but not applied in M1; M3's dataset applies it when a mask is requested.
- An educational document for the presentation was created: `docs/dataset_description.md` (the four datasets in plain
  language, what a mask is, why masks matter) with `docs/photo_vs_mask_example.jpg` (three Sethy photos next to their
  RiceSeg masks). The photos in that image come from Sethy and RiceSeg: credit them if the image is shown publicly.
- `docs/` also contains `PaddyGaurdArchDia.drawio` and `PaddyGuard.drawio.png`, which the developer added; they were not
  touched.

**Tests (M1).** `test_download` (3), `test_manifest` (14: columns and counts, ids, relative paths, sizes, Paddy Doctor
metadata, stem pairing with case-different extensions, unmatched files reported, Dhan background and CSV
disagreement, wrong expected count, deterministic debug subsample, round trips, the M2-columns guard, hex hash
survives a CSV round trip), `test_dataset_card` (3), `test_inspect_masks` (6).

---

## 6. M2: duplicates, splits and the cached images

**Purpose.** Prevent leakage. Near-duplicate photos in both train and test inflate every number, and the dataset pages
report no duplicate counts, so it is measured.

### 6.1 Method as implemented

1. **Perceptual hash** (`imagehash.phash`, hash size 8, 64 bits) of every image, stored as hex in the manifest.
   JPEGs are decoded at reduced size (`draft_size` 256) for speed, identically for all images; **EXIF rotation is
   applied before hashing** (Dhan photos carry rotation tags). Hashing 17,445 images takes about 20 seconds using
   all CPU cores.
2. **Pairwise Hamming distance** in chunks (512 rows at a time) with a vectorised popcount (`np.bitwise_count`, hence
   `numpy>=2.0`), within each dataset up to distance 16 (for the report) and between each target and Paddy Doctor.
3. **Union-find** groups per dataset, using a per-dataset threshold. `group_id` is the smallest `image_id` in the
   group; the **representative** of a group is the image whose id equals its group id (`dedup.is_representative`).
4. **Cross-dataset leakage check:** any T1 or T2 image within the leakage threshold of a Paddy Doctor image is
   dropped (`dropped_leakage`).
5. **Drop policy** (recorded in the manifest `split` column):
   - T2: non-representatives are dropped (`dropped_duplicate`), since T2 has no training use.
   - T1: non-representatives are **kept** as group members (they stay in the segmenter's training and validation
     data), and only inside `seg_test` is one image per group kept; the splitter drops the extra members there. Any
     evaluation over all of T1 outside `seg_test` (the M5 zero-shot test) must use representatives only
     (`representatives_only=True` in the dataset).
   - S: duplicates are kept but share a group id so they never straddle a split.
6. **Splits** with fixed seed 42, `StratifiedGroupKFold` with shuffling, whole groups per split, stratified by class
   folder. S and T1 use 20 folds (70/15/15 = 14/3/3 folds); T2 field uses 5 folds (40/60 = 2/3 folds). A ratio that
   does not divide into whole folds is rejected. Values: S `train/val/test`; T1 `seg_train/seg_val/seg_test`; T2 field
   `cal_pool/test`; T2 white `zero_shot`.
7. **Cached resized copies** in `data/processed/<dataset>/<image_id>.jpg`:
   - S and T2: short side 256, never upscaled, JPEG quality 95, EXIF rotation applied.
   - T1: **native resolution, never downscaled**; a real JPEG is copied byte for byte, anything else is re-encoded.
     About 7% of the Sethy files named `.jpg` are actually PNG files (found on a 150-image sample), so they are
     re-encoded to JPEG. The RiceSeg mask is cached beside each T1 image as `<image_id>_mask.png`, converted to a
     single-channel PNG with the same size as the photo (a mismatch raises an error). The raw JPEG values are kept; the
     threshold is applied at load time.
   - Existing files are not rewritten; dropped images are not cached; a `--debug` run writes to
     `data/processed_debug/` because debug image ids differ from the real ones.
8. **Run order:** `python -m paddyguard.data.dedup`, then `.splits`, then `.cache` (each supports `--debug`). `dedup` and
   `splits` refuse to run again once splits exist.

### 6.2 How the thresholds were chosen (the part that needed judgement)

The spec says to start at Hamming distance 8 and look at about 30 boundary pairs. The first full run with 8 everywhere
showed three things:

- A 64-bit phash always has exactly 32 set bits, so only **even** distances occur (odd columns in the histogram are
  zero).
- The pair-count histograms differ by dataset (counts of pairs per distance, d=0,2,...,16):

| Pairs | d=0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S | 229 | 822 | 1,910 | 2,744 | 3,401 | 4,060 | 5,273 | 8,539 | 27,334 |
| T1 | 5,174 | 1,534 | 694 | 437 | 220 | 207 | 192 | 606 | 3,335 |
| T2 | 0 | 4 | 8 | 37 | 104 | 350 | 894 | 2,191 | 4,376 |
| T1 vs S (cross) | 0 | 0 | 0 | 0 | 0 | 8 | 85 | 877 | 7,486 |
| T2 vs S (cross) | 0 | 0 | 0 | 0 | 0 | 0 | 36 | 281 | 1,856 |

- With a single threshold of 8, T2 formed one chain of **33 different leaves** (over-merging: plain-background leaf
  photographs look alike), while Sethy, which has an obvious plateau around 8 to 12 and a jump at 14 and above,
  was under-grouped.

The boundary pairs (a contact sheet, `reports/figures/dedup_boundary_pairs.jpg`, with a mix per dataset and for the
cross-dataset check) were then viewed and the thresholds set per dataset:

| Dataset | Threshold | What the pairs showed |
| --- | --- | --- |
| S (Paddy Doctor) | 8 | Pairs at 8 to 10 are near-identical shots of the same plot. Over-grouping only costs a little class-balancing flexibility while under-grouping leaks, so 8 is the conservative choice. |
| T1 (Sethy) | 14 | Pairs at 8 to 14 are shifted or zoomed copies of the same leaf scene. The knee is at 14. Pairs just above 14 were not inspected, so some duplicates may remain at larger distances. |
| T2 (Dhan) | 4 | Pairs at 4 are repeat shots of the same leaf; pairs at 6 mix repeats and different leaves on the same plain background. |
| Leakage (cross-dataset) | 8 | The closest cross-dataset pairs (distance 10 to 12) are visibly unrelated photos (for example a dark-background leaf against a field shot). |

**Important honesty point:** this review of the boundary pairs was done visually during implementation, **not by the
developer**. This is stated in `configs/data.yaml` (`dedup.review_note`) and in `reports/dedup_report.md`, and the sheet
is kept so it can be re-checked. The spec asks for the developer to view the pairs; the developer asked to keep their
involvement minimal.

### 6.3 Results

- Groups: Paddy Doctor **10,407 images in 6,141 groups** (1,459 groups with more than one image, largest 51); Sethy
  **5,932 images in 1,847 groups** (1,727 multi-image groups, largest 25; **5,174 pairs have identical hashes**, so
  Sethy is heavily duplicated, which is a finding in itself and confirms the guide's warning); Dhan 1,106 images in
  1,099 groups (3 multi-image groups, largest 6).
- **No leakage**: 0 Sethy and 0 Dhan images are within the leakage threshold of a Paddy Doctor image.
- Splits (seed 42):

| Dataset | Split sizes |
| --- | --- |
| S | train 7,334 / val 1,587 / test 1,486 (the split is by group, so the ratios are approximate) |
| T1 | seg_train 4,074 / seg_val 927 / seg_test 280 (87 bacterial blight, 66 blast, 79 brown spot, 48 tungro); 651 extra duplicate members dropped from seg_test only; only **1,847 representatives** exist for an evaluation over all of Sethy |
| T2 field | cal_pool 135 / test 201, of which in the three shared classes: **pool 79 / test 119** (brown spot 21 / 28, blast 28 / 46, tungro 30 / 45); 1 duplicate dropped |
| T2 white | 763 `zero_shot` (6 duplicates dropped); the three shared classes give 401 images (brown spot 90, blast 193, tungro 118) |

  S per-class (train / val / test): bacterial_leaf_blight 334/72/73, bacterial_leaf_streak 286/51/43,
  bacterial_panicle_blight 255/41/41, blast 1,226/242/270, brown_spot 700/134/131, dead_heart 1,006/247/189,
  downy_mildew 446/68/106, hispa 1,115/261/218, normal 1,171/320/273, tungro 795/151/142.
- The pre-agreed T2 rules applied automatically: allowed calibration sizes for T2 = **[25, 50]** (79 x 65% = 51.4);
  brown spot has **28 test images**, flagged LOW-N (kept; bootstrap CIs; low-n flag in every T2 table).
- Cache: **16,787 images**, 791 MB (S 10,407; T1 5,281 images plus 5,281 masks; T2 1,099). Checked: T1 image and mask
  sizes agree on 200 samples; S short side is 256 on 100 samples; Dhan images are 256x546 or 546x256.

### 6.4 Consequences for later modules (written into the spec and notes)

- `seg_test` has only 280 images (48 tungro, which has no severity rule), so severity and decision results need
  bootstrap CIs and a small-sample caveat.
- T2 results are small-sample corroboration with wide CIs; T2 calibration sizes are {25, 50}.
- The segmenter trains on 4,074 T1 images, many of them near-duplicates of each other.
- Conformal calibration for T1 draws from the 5,001 T1 images outside `seg_test`, by `group_id`.
- Dhan images carry EXIF rotation tags, applied when hashing and resizing. Dhan raw width and height in the manifest
  are the stored (not displayed) dimensions.

### 6.5 Spec changes made in M2

`ARCHITECTURE.md` M2 now documents per-dataset thresholds, the leakage threshold, the drop policy, the cache
exception for Sethy and its masks, the run order and the re-split guard, and has a "Measured on the real data" block;
the layout lists `data/dataset_card.py` and `data/cache.py`; M5 notes that T1 evaluations use representatives.

**Tests (M2).** `test_dedup` (8: popcount, near pairs against brute force, cross pairs, union-find, planted exact and
slightly resized duplicates found with nothing else flagged, EXIF-rotated copy hashes close, groups/drops/leakage,
boundary pair selection), `test_splits` (11: ratios, no group in two splits, stratification and determinism,
folds that do not divide are rejected, T1 test keeps one image per group while training keeps members, T2
field/white handling, re-split refusal, allowed calibration sizes for pools of 30, 80, 135, 153, 154),
`test_cache` (2: S/T2 resizing, T1 native size and single-channel masks, a PNG named .jpg, EXIF rotation, dropped rows
skipped, idempotent).

---

## 7. M3: label harmonisation, loaders and transforms

**Purpose.** One canonical label space, and PyTorch datasets that can restrict predictions to the classes two datasets
share.

### 7.1 Implemented

- `configs/labelmap.yaml` filled: canonical 4-class order (bacterial_leaf_blight, blast, brown_spot, tungro) then the
  six Paddy Doctor-only classes (normal, bacterial_leaf_streak, bacterial_panicle_blight, dead_heart, downy_mildew,
  hispa). `LabelMap.source_classes` = the 10 classifier output columns, in that order.
  - S: identity for all 10 folders. T1: Bacterialblight, Blast, Brownspot, Tungro map to the matching canonical
    classes (taken from the guide's list of the four Sethy classes).
  - T2: `Browon Spot` -> brown_spot, `Rice Blast` -> blast, `Rice Turgro` -> tungro (field; **confirmed by the developer
    from a montage of 5 images per folder**), `Brown Spot` -> brown_spot and `Rice Tungro` -> tungro (white
    background; mapped by name only, not visually checked), `Leaf Scaled`, `Sheath Blight`, `Shath Blight` -> null (no
    Paddy Doctor counterpart: kept in the manifest, excluded from the shared label space).
  - Shared sets: `S_T1` = the 4 canonical classes; `S_T2` = blast, brown_spot, tungro.
- `data/labelmap.py`: `load_labelmap` (validates that targets are known classes and shared sets are canonical),
  `missing_raw_labels`, `harmonise` (fills the manifest `label` column; refuses when a raw label is missing),
  `shared_classes(lm, domain)`, `restrict_logits(logits, class_names, shared)` (keeps the shared columns in the
  shared order, works on NumPy and torch, and rejects a class the model does not have or a wrong column count),
  `softmax`, and a CLI (`python -m paddyguard.data.labelmap`, with `--debug`) that fills the `label` column and
  changes nothing else.
- `configs/transforms.yaml` and `data/transforms.py` (Albumentations 2.x, ImageNet mean/std, 224 px):
  - `baseline`: RandomResizedCrop(224, scale 0.5 to 1) + horizontal flip.
  - `strong` (also used for `strong_mixstyle`, since MixStyle is a model-side change in M5): rotation, colour jitter,
    brightness/contrast, hue/saturation/value, Gaussian blur, JPEG compression with quality 40 to 95, coarse dropout.
  - `eval`: shortest side to 256, centre crop 224.
  - **Magnitudes the spec does not give (rotation 15 degrees, jitter 0.3, blur kernel 3 to 7, dropout size 5 to 20%,
    probabilities) are design choices marked as such in the config, not tuned.** Only the JPEG quality range and the
    256/224 sizes come from the spec.
  - A seed makes a pipeline reproducible; masks use nearest-neighbour interpolation so they stay binary.
- `data/datasets.py`:
  - `PaddyDataset(manifest, split, domain, transform, classes=None, with_mask=False, mask_threshold=None,
    representatives_only=False)` returns `(image, label_index, image_id)` (plus the binary lesion mask when
    requested). Domains: `S`, `T1`, `T2` (field only), `T2-white`. `split=None` means every image that was not dropped.
    `classes` restricts and re-indexes the labels (default: all 10 source classes); images with a label outside the
    set, or with no label, are skipped. Images are read from `data/processed/`. Masks are available for T1 only and
    need the threshold. `representatives_only` keeps one image per duplicate group.
  - `make_loader(dataset, training, batch_size, seed, num_workers, pin_memory)` with a seeded generator, a seeded
    shuffle for training, `persistent_workers` when there are workers, and **per-worker augmentation seeds**
    (otherwise every worker repeats the same augmentations). **A training loader accepts only `S/train` and raises
    for anything else**, which is the code-level assertion that target domains never enter training; verified on the
    real data (a T1 training loader is rejected).

### 7.2 Findings and checks on the real data

- The `label` column was filled for all 17,445 rows: S 10,407 of 10,407, T1 5,932 of 5,932, **T2 606 of 1,106**
  (the rest map to null).
- Batches of `(32, 3, 224, 224)` were pulled for all domains and splits with class indices inside the requested label
  space: S train 7,334, val 1,587, test 1,486 (10 classes); T1 representatives 1,847 and `seg_test` 280 (4 classes);
  T2 test 119 and cal_pool 79 (3 classes); T2-white 401 (3 classes).
- T1 with masks: 4,074 training images; the lesion share of a mask is about 4.9% of pixels (consistent with the
  5% found in M1).
- Loader speed with `strong` augmentation: about 400 images per second on one process on the laptop. A 4-worker test
  looked slower only because of macOS process start-up in a 10-batch test; Linux (Kaggle) does not pay that cost.
  A first smoke test appeared to hang because macOS cannot re-import a script piped through standard input into
  spawned worker processes: this is a test-harness artefact, not a loader bug (it works from a script file).

**Tests (M3).** `test_labelmap` (13: real map consistency, validation errors, harmonise, missing labels, coverage of
every raw label of the real manifest, shared classes per domain, restrict_logits order and errors, softmax rows sum to
1 for NumPy and torch, the CLI changes only the label column), `test_transforms` (8: normalised CHW float tensors for
every pipeline, determinism, seeded reproducibility, `strong_mixstyle` equals `strong`, masks follow image geometry
and stay binary, unknown name rejected), `test_datasets` (11: item format, 10 default classes, class subset
re-indexing, domain row selection, dropped rows never returned, representatives, masks, argument validation, the
training-loader guard, reproducible order).

---

## 8. M4: disease classifier training

**Purpose.** A strong, honest in-domain baseline, and the model used downstream (cross-dataset evaluation in M5,
conformal gate in M8, the app in M11).

### 8.1 What was implemented

- `classify/models.py`: `create_model(name, num_classes, pretrained)` = `timm.create_model(...)`. EfficientNet-B0 is the
  main model (4.02 million parameters, ImageNet weights `efficientnet_b0.ra_in1k`).
- `classify/evaluate.py` (shared with M5): `predict` (ids, true indices, float32 logits, mixed precision on GPU),
  `classification_metrics` (accuracy, **macro-F1 as the primary metric**, balanced accuracy, per-class recall; recall
  of a class with no true images is `None`, not 0), `per_variety_macro_f1` (using Paddy Doctor's variety metadata),
  and `preds_frame` (the per-image prediction format of the spec's section 6: `image_id`, `y_true`, `y_pred`, one
  `logit_<class>` float32 per training class, `domain`, `split`).
- `classify/train.py`, fully config-driven (`configs/cls_effb0.yaml`):

```yaml
model: efficientnet_b0      img_size: 224       epochs: 15        batch_size: 64
optimizer: {name: adamw, lr: 3.0e-4, weight_decay: 0.01}
scheduler: {name: cosine, warmup_epochs: 1}       # linear warm-up, then cosine to zero, per iteration
loss: {name: ce, label_smoothing: 0.1}            amp: true
augmentation: baseline      seed: 0               pretrained: true     num_workers: 2
early_stopping: {metric: val_macro_f1, patience: 4}
acceptance: {min_macro_f1: 0.90, min_class_recall: 0.80}
```

  - Loop: train one epoch, validate on S/val (macro-F1, accuracy), save checkpoints, early stopping on val macro-F1
    (patience 4), reload the best weights, evaluate on S/val and S/test.
  - Outputs in the run directory: `metrics.json` (val and test macro-F1, accuracy, balanced accuracy, per-class recall,
    per-variety macro-F1 on test, per-epoch history with loss, val metrics, learning rate and seconds, best epoch,
    device, git commit, class list, and an **`acceptance` block**), `preds_S_test.parquet`, `ckpt.pt` (best weights plus
    class names, config and epoch: what later modules load), `ckpt_last.pt` (model, optimizer, scheduler, scaler,
    history and early-stopping state), `config.yaml`, `git_commit.txt`, `log.txt`.
  - `--resume RUN_DIR` continues an interrupted run in its own directory (the epochs already done are not redone);
    `--seed` overrides the config; `--no-pretrained` allows offline debugging; `--debug` uses at most 200 images per
    split, 1 epoch, the CPU and no workers.
  - Mixed precision (`torch.autocast`, `GradScaler`) is used on GPU only. The classifier's `img_size` comes from the
    classifier config and the transform resize keeps the 256/224 ratio.
  - Seeds: Python, NumPy and torch are seeded from the config, the data-loader shuffle and the augmentation RNG too.
  - The acceptance rule from the spec: "clearly above 90% macro-F1 and no single class below 80% recall"; thresholds
    are config values. A run below this should trigger debugging of the data pipeline before moving on.

**Tests (M4, 10).** LR schedule (warm-up, monotone decay, near zero at the end, no-warm-up start), early stopper (best
epoch and patience), metrics on known predictions, absent-class recall is `None`, prediction-file contract,
per-variety macro-F1, a full tiny training run that writes every output with 10 logits per image and loadable
checkpoint, resume does not retrain finished epochs, same seed gives the same training losses, and a command-line
`--debug` run through the environment-variable paths (one epoch, config with the seed recorded, commit file written).

### 8.2 The debug run on the laptop (CPU, real data)

`python -m paddyguard.classify.train --config configs/cls_effb0.yaml --seed 0 --debug`: 200 images per split, 1 epoch,
pretrained weights downloaded successfully. Result: val macro-F1 0.067, test macro-F1 0.069, loss 3.55, as expected
after about three training steps; acceptance correctly reported "not passed". It proves the pipeline runs end to end.

### 8.3 Kaggle: what was prepared and what happened

- The laptop has no GPU, so training runs on Kaggle. Because there is no GitHub repository, the code is shipped as a
  Kaggle dataset instead of a `git clone`.
- `scripts/package_for_kaggle.sh` builds two zips in `data/kaggle_upload/`: `paddyguard-processed.zip` (746 MB: the
  manifest and `processed/`) and `paddyguard-code.zip` (54 KB: `src`, `configs`, `pyproject.toml`,
  `requirements.txt`). Kaggle unzips uploads automatically.
- The developer had already uploaded the three raw datasets (RiceSeg, Sethy, Dhan) as private Kaggle datasets; they are
  **not needed for training** (the processed images supersede them) and were kept as a backup. Paddy Doctor was not
  among the uploads shown (it is a competition dataset, and the processed images cover it).
- `notebooks/kaggle_run.ipynb` is thin: it finds the two attached datasets by searching `/kaggle/input` (robust to
  Kaggle's mount path), copies the code, installs `timm`, `albumentations>=2.0` and `imagehash`, sets `PYTHONPATH`,
  `PADDYGUARD_DATA` and `PADDYGUARD_OUT=/kaggle/working`, then has cells for a 1-minute `--debug` check, the real run, a
  results summary, and a resume example. `docs/kaggle_guide.md` has the step-by-step instructions for the developer
  (phone verification for GPU and Internet, uploading the two datasets, importing the notebook, Add Input, settings
  GPU on and Internet on, running the cells, Save Version, downloading `runs`, troubleshooting).
- **Problem 1 (setup cell):** `pip install -e .` failed in the Kaggle notebook ("pip subprocess to install build
  dependencies did not run successfully"): pip's isolated build step could not fetch its build tool. Fix: the package
  does not need installing; the notebook now sets `PYTHONPATH` to the copied `src` folder, and prints
  `paddyguard OK` plus the `timm` and `albumentations` versions. Verified by importing the package from a clean
  extraction of the code zip with no installation. Recorded in the guide's troubleshooting section.
- **Quick check on Kaggle** (200 images, 1 epoch): the pretrained weights downloaded (Internet on) and the result
  matched the laptop debug run almost exactly (loss 3.5526 vs 3.5492; val macro-F1 0.0668 vs 0.0672), confirming the
  pipeline behaves the same there. This quick check runs on the CPU by design, so the GPU path was first exercised by
  the real run.
- **Real run:** one interactive session on a Kaggle T4 GPU (the "GPU T4 x2" option; one GPU is used), seed 0, baseline
  augmentation, 15 epochs, about 31 seconds per epoch on average (the first epoch took 54 seconds). The "Save Version" the
  developer started afterwards (a repeat of the whole notebook, queued for a GPU) was unnecessary and was to be
  cancelled; outputs were instead downloaded from the interactive session by zipping the `runs` folder (without
  `ckpt_last.pt`).
- **Housekeeping on the laptop:** the download extracted as `runs/runs/...` with two run folders; the real one
  (`cls_effb0-0-20261019-123812`) was moved to `runs/`, the 1-epoch quick-check folder (`...123547`) was deleted, and the
  zip was moved to `data/archives/kaggle_runs_seed0.zip`.
- The `runs/` folder also contains 16 `data-0-<timestamp>` folders from the data-preparation scripts (see section 4); they
  are logs, not training runs.

### 8.4 Training curve of the real run (seed 0)

| Epoch | Train loss | Val macro-F1 | Val accuracy | Seconds |
| --- | --- | --- | --- | --- |
| 1 | 2.2076 | 0.6073 | 0.6849 | 53.9 |
| 2 | 0.9831 | 0.7879 | 0.8223 | 22.0 |
| 3 | 0.7598 | 0.8271 | 0.8570 | 23.8 |
| 4 | 0.6817 | 0.8773 | 0.8916 | 22.9 |
| 5 | 0.6401 | 0.8860 | 0.9067 | 23.0 |
| 6 | 0.6180 | 0.9039 | 0.9219 | 22.1 |
| 7 | 0.6014 | 0.9097 | 0.9294 | 25.1 |
| 8 | 0.5892 | 0.9185 | 0.9332 | 22.5 |
| 9 | 0.5804 | 0.9179 | 0.9338 | 35.9 |
| 10 | 0.5736 | 0.9214 | 0.9351 | 43.9 |
| 11 | 0.5669 | 0.9275 | 0.9395 | 40.9 |
| 12 | 0.5639 | 0.9246 | 0.9382 | 37.7 |
| 13 | 0.5613 | 0.9266 | 0.9389 | 36.7 |
| **14** | 0.5609 | **0.9322** | 0.9433 | 34.6 |
| 15 | 0.5596 | 0.9259 | 0.9370 | 26.1 |

Best epoch 14 (val macro-F1 0.9322, balanced accuracy 0.9363, accuracy 0.9433). Early stopping did not trigger (15
epochs). The loss falls smoothly and the validation score rises and plateaus around 0.92 to 0.93; the training loss
stays near 0.56 because of label smoothing (0.1).

### 8.5 Test results (Paddy Doctor test split, 1,486 images, best checkpoint)

| Metric | Value |
| --- | --- |
| Accuracy | 0.926 |
| **Macro-F1 (primary)** | **0.9202** |
| Balanced accuracy | 0.9163 |
| Acceptance (macro-F1 > 0.90 and every class recall >= 0.80) | **passed** (lowest recall: brown spot 0.840) |

Per-class recall on the test split (lowest to highest): brown_spot 0.840, hispa 0.858, bacterial_leaf_blight 0.863,
tungro 0.894, bacterial_leaf_streak 0.907, bacterial_panicle_blight 0.927, normal 0.949, downy_mildew 0.953, blast 0.978,
dead_heart 0.995.

Largest confusions (true to predicted, counts): hispa to tungro 14, brown_spot to bacterial_leaf_blight 11, normal to
hispa 8, hispa to blast 7, bacterial_leaf_blight to downy_mildew 6, hispa to normal 6.

Per-variety macro-F1 and the number of test images per variety: ADT45 0.919 (996), KarnatakaPonni 0.762 (187),
Ponni 0.991 (98), Onthanel 0.905 (57), AtchayaPonni 0.574 (55), Zonal 0.850 (48), AndraPonni 1.000 (27), IR20 1.000
(8), RR 1.000 (5), Surya 0.250 (5). The scores for IR20, RR and Surya rest on 5 to 8 images and are noise; the real
per-variety weak spots are **AtchayaPonni (0.57, 55 images)** and **KarnatakaPonni (0.76, 187 images)**.

Validation split (1,587 images) at the best epoch, per-class recall: bacterial_leaf_blight 0.889, blast 0.942,
brown_spot 0.933, tungro 0.940, normal 0.981, bacterial_leaf_streak 0.980, bacterial_panicle_blight 0.927, dead_heart
0.992, downy_mildew 0.897, hispa 0.881.

### 8.6 Verification of the downloaded run

- `preds_S_test.parquet` has 1,486 rows, its image ids equal the manifest's S/test ids exactly, the logit columns
  are the 10 classes in the training order and float32; recomputing macro-F1 (0.9202) and balanced accuracy (0.9163)
  from the file reproduces `metrics.json`.
- `ckpt.pt` loads, holds the right 10 class names, and is from epoch 14.
- `metrics.json` records device `cuda`, best epoch 14 of 15, and commit `unknown` (the code went up as a zip).

### 8.7 Interpretation

- The classifier is a working in-domain baseline and meets the spec's acceptance rule, but not with a large margin.
  The Paddy Doctor paper's 97.5% F1 (ResNet34) is for the 13-class version and with a split that is not
  duplicate-aware; this run uses 10 classes and group-aware splits, so the numbers are not directly comparable.
- The weakest classes (brown spot, hispa, bacterial leaf blight, tungro) are the disease classes shared with the
  targets and the ones whose errors will matter most in M5; the strong-augmentation variants in M5 may help.
- One run, one seed: no confidence interval or seed spread yet (M5 plans 3 seeds and bootstrap CIs over test images).

---

## 9. Chronological decision log

1. Read the spec and the guide; renamed the spec to `ARCHITECTURE.md`; moved the guide into `docs/`.
2. Reorganised the raw data (section 3.4); extracted the nested RiceSeg zip; kept all original zips.
3. Built the repository skeleton, packaging, configs (spec values only), `.gitignore`.
4. Coverage analysis guide versus modules: no missing module; seven spec corrections; Sethy native-resolution finding.
5. Dhan: field-only main target, `T2-white` secondary, 40/60 split, calibration sizes by the 65% rule, low-n rule,
   fallback, `T1` as the main benchmark and T2 as corroboration.
6. Deployable classes: classify over all 10 classes, advise on five, refer otherwise.
7. No assistant traces in the repository (ignore rules; no attribution lines); spec rewording pending.
8. M0 implemented; local virtual environment; macOS hidden `.pth` workaround (`PYTHONPATH=src`).
9. M1 implemented; run on the real data; mask answers (lesions only, black background, threshold 128); licences CC BY
   4.0 for all four; overlay-sampling bug fixed; masks answers stored in the config.
10. M2 implemented; per-dataset thresholds chosen from histograms and pair sheets (S 8, T1 14, T2 4, leakage 8);
    drop policy (T2 duplicates dropped, T1 duplicates dropped only in `seg_test`); splits fixed; cache built with
    Sethy native and EXIF handling; spec updated with measured values.
11. M3 implemented; Dhan label mapping confirmed by the developer from a montage; Sethy and Paddy Doctor mappings
    written from folder names; transform magnitudes declared design choices.
12. M4 implemented; Kaggle packaged (two datasets, thin notebook, guide); the pip install problem fixed with
    `PYTHONPATH`; quick check matched the laptop; seed 0 trained; acceptance passed; results verified and recorded.
13. The developer paused M5 so that the review can present M0 to M4; this file was written.

---

## 10. Problems found and how they were fixed

| Problem | Where it showed up | Fix |
| --- | --- | --- |
| Raw downloads nested and one zip not extracted; trailing-space folder names; `.DS_Store` files | Initial data inspection | Reorganised (section 3.4) |
| Spec cached all images at short side 256, which would downscale 300 px Sethy images and ignored the masks | Coverage analysis | Sethy native resolution; masks cached beside the images |
| Mask overlay sample biased to the first classes (sorted before taking 20) | Viewing the contact sheet | Random order kept; test added |
| Re-running inspection erased hand-written answers | Design review | Answers stored in `mask_encoding` in the config |
| macOS hides `.pth` files in the venv; Python ignores them | `import paddyguard` failed after a while | `PYTHONPATH=src` for commands; pytest `pythonpath` |
| A single duplicate threshold (8) merged 33 different Dhan leaves and under-grouped Sethy | First full dedup run | Per-dataset thresholds chosen from histograms and boundary pairs |
| Plan to drop all Sethy duplicates would have cut the segmenter's training data by about two thirds | Reviewing the first dedup result | Keep T1 duplicates as group members; drop extras only inside `seg_test` |
| About 7% of Sethy `.jpg` files are really PNGs; Dhan photos have EXIF rotation tags | Sampling file formats | Cache re-encodes non-JPEG; EXIF rotation applied when hashing and resizing |
| Debug image ids differ from real ones, so a debug cache could be reused wrongly | Reviewing the debug run | Debug cache goes to `data/processed_debug/` |
| Re-running the manifest builder would wipe hashes, groups and splits | Design review | Guard: refuses without `--force`; `dedup` and `splits` also refuse to redo splits |
| `assign_all_splits` required config for datasets absent from the data | A failing test | Only configured datasets with rows are split |
| Worker processes cannot re-import a stdin script on macOS (looked like a hang) | First loader smoke test | Re-ran from a script file; not a loader bug |
| `pip install -e .` failed on Kaggle | Kaggle setup cell | `PYTHONPATH`, no install |
| Spec text and example disagreed on the T2 calibration-size rule | Reviewing the rule | Rule loosened to 65% of the pool, in four places |

---

## 11. Findings that matter for the review and the report

1. **Sethy is heavily duplicated:** 5,174 exact-hash pairs; 5,932 images collapse to 1,847 groups. Any benchmark that
   ignores this inflates its numbers.
2. **No leakage** from Paddy Doctor into Sethy or Dhan at the leakage threshold; the nearest cross-dataset pairs are
   unrelated photos. Paddy Doctor itself has 10,407 images in 6,141 groups (mostly near-identical shots of the same
   plot), so a duplicate-aware split is necessary there too.
3. A single perceptual-hash threshold does not work across datasets: plain-background leaf photos (Dhan) need a much
   tighter threshold than field photos or Sethy's zoomed copies; thresholds were chosen per dataset and documented.
4. The RiceSeg masks are binary (white = lesion, black background, threshold 128) and contain **no leaf class**, so a
   leaf mask must be derived for severity (M6). Lesions cover about 5% of mask pixels.
5. The Dhan shared-class test set is small: brown spot has only 28 test images and the shared-class calibration pool is
   79, so T2 calibration sizes are limited to {25, 50} and T2 results are corroboration with wide CIs.
6. The Sethy test split has only 280 images (48 tungro), so severity and decision results will need bootstrap CIs.
7. In-domain classifier: macro-F1 0.9202 on Paddy Doctor test (10 classes, group-aware split); the weakest classes are
   brown spot, hispa, bacterial leaf blight and tungro; the weakest well-populated varieties are AtchayaPonni and
   KarnatakaPonni.
8. **Observation (2026-10-19, not yet quantified): many Sethy photos do not have a plain background** (field backgrounds, a leaf held in a hand, grey sheets), contrary to the guide's description. M6's colour-distance leaf mask and M7's "plain to field" wording must be checked against this at the start of M6 (validate the leaf mask on 30 images and count plain versus busy backgrounds).
9. Mask threshold sensitivity (300 Sethy masks): mean lesion share 6.2% at threshold 10, 5.7% at 60, 5.5% at 128, 5.3% at 200.
10. Only the ~5% blast severity threshold has been found for TNAU and it is unconfirmed; all other severity band edges are
   design choices.

Limits to state in the report: the duplicate-threshold review was done visually by the implementer and not by the
developer; pairs just above the Sethy threshold were not inspected; the white-background Dhan label mapping was done by
name only; GPU training is not bit-exact; one seed so far; the knowledge base does not exist yet (nothing is
expert-verified).

---

## 12. Open items and what is not done

- **M5 (paused):** cross-dataset evaluation (needs the saved checkpoint; can run on the laptop), MixStyle, the
  ablation grid (3 variants x 3 seeds = 9 runs; 8 still to run on Kaggle, about 4 hours of GPU), bootstrap CIs.
- M6 to M12: severity segmentation, simulated shift, reliability/conformal, advisory knowledge base and engine, decision
  evaluation, demo app, reporting.
- Re-do the Paddy Doctor seed-0 numbers with seeds 1 and 2 and the other variants before claiming any robustness
  conclusions.
- `git init` has not been run, so run directories record `unknown` as the commit.
- `ARCHITECTURE.md` still mentions the assistant tooling in six places (developer to decide on rewording).
- Dataset licences: CC BY 4.0 reported for all four; read the Paddy Doctor competition rules page too.
- The app's conformal gate must be calibrated in the 10-class space (decision pending for M11).
- Knowledge-base values must be typed in by the developer from TNAU and CIBRC (31.03.2024 edition); the guide's
  non-code checklist items remain (Amato et al. citation details, SOFS-Rice removal, a Google Scholar search for prior
  "Paddy Doctor" with "Sethy" work, optional DataPort question, optional expert glance at the knowledge base).
- `docs/` holds two diagram files added by the developer (`PaddyGaurdArchDia.drawio`, `PaddyGuard.drawio.png`).

---

### The M5 plan that was agreed (paused, to resume after the review)

M5 = cross-dataset evaluation and robustness ablations. It splits into three pieces:

1. **Cross-dataset evaluation (laptop only, no Kaggle).** For a saved checkpoint: S/test on all classes, T1 on the 4 shared
   classes (**representatives only**), T2 field on the 3 shared classes, and T2-white as a separate zero-shot row; metrics
   macro-F1, balanced accuracy, per-class recall, confusion matrices, bootstrap 95% CIs over images (1,000 resamples,
   stratified by class), per-image probabilities saved as `preds_<domain>.parquet`; the gap is the in-domain macro-F1 **on the
   same shared classes** minus the target macro-F1, plus the per-variety breakdown on S/test. Output
   `reports/results/cross_dataset.csv`. The seed-0 checkpoint already exists, so this can run immediately (a few minutes on CPU).
   `classify/evaluate.py` already has the prediction and metric helpers; `labelmap.restrict_logits` does the class restriction.
2. **Ablation grid (needs Kaggle GPU, no extra work from the developer).** EfficientNet-B0 x {`baseline`, `strong`,
   `strong_mixstyle`} x seeds {0, 1, 2} = 9 runs, 1 done, 8 to go (about 4 GPU hours; about 31 s per epoch). Needs the MixStyle
   module (`classify/mixstyle.py`, forward hooks after `blocks[1]` and `blocks[2]`, active only in training; the spec has a test
   for it) and `classify/ablations.py` (generates the configs and the run list). Plan: one notebook version that runs all 8
   runs, each followed by the cross-dataset evaluation, so one "Save and Run All" is enough; the code zip must be rebuilt with
   `./scripts/package_for_kaggle.sh` and re-uploaded as a new version of `paddyguard-code` first (the processed dataset does
   not change). If GPU quota is tight, start with `strong` x 3 seeds (about 1.5 hours); the spec's cut order drops the MixStyle
   variant first. Optional extras only if time allows: MobileNetV3, ConvNeXt-Tiny, class-balanced focal loss, few-shot (E8).
3. **Tables and statistics (laptop only).** Mean and standard deviation over seeds, bootstrap CIs over test images,
   paired seed-wise differences between variants as a plain table, **no p-values** with 3 seeds; Grad-CAM only as a
   qualitative figure.

Remember when resuming: T2 results are low-n for brown spot (28 test images); T1 evaluations must use representatives only;
the classifier output order is `LabelMap.source_classes`; run directories created on Kaggle record the commit as `unknown`.

## 13. How to reproduce everything

Environment: Python 3.12, the packages in `requirements.txt`; use the project venv `.venv` and set `PYTHONPATH=src` for
commands. Data lives under `data/` (or wherever `PADDYGUARD_DATA` points).

```
# checks
.venv/bin/python -m pytest -q
.venv/bin/ruff check .

# M1
PYTHONPATH=src .venv/bin/python -m paddyguard.data.download            # layout check and class counts
PYTHONPATH=src .venv/bin/python -m paddyguard.data.manifest            # manifest.csv and dataset card
PYTHONPATH=src .venv/bin/python -m paddyguard.data.inspect_masks       # overlays and mask note

# M2 (in this order; each also takes --debug)
PYTHONPATH=src .venv/bin/python -m paddyguard.data.dedup
PYTHONPATH=src .venv/bin/python -m paddyguard.data.splits
PYTHONPATH=src .venv/bin/python -m paddyguard.data.cache

# M3
PYTHONPATH=src .venv/bin/python -m paddyguard.data.labelmap            # fills the manifest label column

# M4 (laptop smoke test; the real run is on Kaggle)
PYTHONPATH=src .venv/bin/python -m paddyguard.classify.train --config configs/cls_effb0.yaml --seed 0 --debug

# Kaggle packaging
./scripts/package_for_kaggle.sh      # then follow docs/kaggle_guide.md
```

Re-running `manifest`, `dedup`, `splits` after the splits exist is refused without `--force`; do not re-split.

Generated reports (regenerable from the code): `reports/dataset_card.md`, `reports/mask_inspection.md`,
`reports/dedup_report.md`, `reports/splits_report.md`, `reports/figures/mask_overlays/`,
`reports/figures/dedup_boundary_pairs.jpg`; the human-written running record is `reports/experiment_log.md`.

---

## 14. Appendix: file inventory, test inventory, numbers

### 14.1 Source files written (lines of code)

| File | Lines |
| --- | --- |
| `utils/seed.py`, `paths.py`, `io.py`, `logging.py` | 17, 40, 96, 32 |
| `data/download.py`, `manifest.py`, `dataset_card.py`, `inspect_masks.py` | 95, 296, 76, 205 |
| `data/dedup.py`, `splits.py`, `cache.py` | 376, 188, 125 |
| `data/labelmap.py`, `transforms.py`, `datasets.py` | 121, 79, 146 |
| `classify/models.py`, `evaluate.py`, `train.py` | 11, 85, 259 |

Other files: `configs/{data,labelmap,transforms,cls_effb0,seg_unet_r34,shift,reliability,severity_bands,advisory}.yaml`,
`scripts/package_for_kaggle.sh`, `notebooks/kaggle_run.ipynb`, `docs/{dataset_description.md,kaggle_guide.md,
photo_vs_mask_example.jpg}`, `pyproject.toml`, `requirements.txt`, `.gitignore`, `ARCHITECTURE.md`.

### 14.2 Tests (106 in total)

| File | Tests | File | Tests |
| --- | --- | --- | --- |
| `test_seed` | 2 | `test_dedup` | 8 |
| `test_paths` | 3 | `test_splits` | 11 |
| `test_io` | 9 | `test_cache` | 2 |
| `test_logging` | 2 | `test_labelmap` | 13 |
| `test_import` | 1 | `test_transforms` | 8 |
| `test_download` | 3 | `test_datasets` | 11 |
| `test_manifest` | 14 | `test_classify` | 10 |
| `test_dataset_card` | 3 | `test_inspect_masks` | 6 |

### 14.3 Key numbers in one place

| Item | Value |
| --- | --- |
| Manifest rows | 17,445 (S 10,407; T1 5,932; T2 1,106) |
| Kept after dedup | S 10,407; T1 5,281 (of which 1,847 representatives); T2 1,099 |
| Duplicate groups | S 6,141; T1 1,847; T2 1,099 |
| Thresholds (Hamming) | S 8; T1 14; T2 4; leakage 8 |
| Leakage into targets | 0 and 0 |
| Sethy exact-hash pairs | 5,174 |
| S split | 7,334 / 1,587 / 1,486 |
| T1 split | 4,074 / 927 / 280 |
| T2 field | cal_pool 135 / test 201 (shared classes 79 / 119) |
| T2 white zero-shot | 763 (shared classes 401) |
| T2 calibration sizes | [25, 50] |
| Cached images | 16,787 (791 MB) |
| Kaggle upload sizes | processed 746 MB, code 54 KB |
| Classifier | EfficientNet-B0, 4.02 M parameters, pretrained, 224 px |
| Training | 15 epochs, batch 64, AdamW 3e-4, cosine with 1 warm-up epoch, label smoothing 0.1, mixed precision |
| Time | about 31 s per epoch on a Kaggle T4 |
| Best epoch / val macro-F1 | 14 / 0.9322 |
| Test macro-F1 / balanced accuracy / accuracy | 0.9202 / 0.9163 / 0.926 |
| Lowest test class recall | brown_spot 0.840 |
