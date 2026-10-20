# Experiment log

Date, run id, what changed, one-line result. Becomes the methods and results chapters.

## 2026-10-19: T2 (Dhan-Shomadhan) design decision
- Main `T2` target = the 337 field-background images; split by group, stratified by class, into 40% `cal_pool` / 60% `test` after dedup. Calibration sizes for `T2`: {25, 50, 100} (`T1` keeps {25, 50, 100, 150}).
- The 769 white-background images are kept as a zero-shot-only secondary row, `T2-white` (no calibration pool).
- Reason: Paddy Doctor is field photography, so field-versus-field is the cleaner measure of dataset shift; the white-background images resemble Sethy's plain-background leaves and add little as a second target. A 30% pool (about 100 images) would make n=100 draws nearly identical.
- The project guide said "use the field-background images for the main test" while ARCHITECTURE.md M2 split all of `T2` 30/70; this reconciles them. ARCHITECTURE.md M2/M8 and `configs/{data,reliability}.yaml` updated.
- Caveat to state in the review: the `T2` test set is small, so report bootstrap 95% CIs.
- Open check before splitting: per-class counts of the field subset (stop if any class has fewer than about 30 test images).

## 2026-10-19: T2 split procedure (follow-up)
- Build the `T2` split only after dedup, from real counts. Print per-class counts of the three shared classes in the field subset first. Only those 199 pre-dedup images can calibrate the shared-space gate; Leaf Scaled and Sheath Blight cannot.
- `T2` calibration sizes = values in {25, 50, 100} that are at most 65% of the pool, so draws never leave less than about 35% of the pool unused (loosened from 60%/40% on 2026-10-19: a pool of ~80 then allows {25, 50}; 100 needs a pool of about 154).
- Brown spot is expected to have about 29 test images pre-dedup. Keep it, report per-class numbers with bootstrap CIs, flag low-n in every `T2` table.
- Fallback if field counts become unusable: all shared-class Dhan images (field and white), stratified by background, field and white reported separately.
- Review framing: `T1` (Sethy, 5,932) is the main benchmark; `T2` is small-sample corroboration with wide CIs.
- Mapping confirmed by the developer on 2026-10-19 (montage of 5 images each from `Browon Spot`, `Rice Blast`, `Rice Turgro`); `labelmap.yaml` T2 written.

## 2026-10-19: M2 run (duplicates, splits, cache)
- Thresholds (Hamming, even distances only): S 8, T1 14, T2 4; leakage threshold 8. First attempt with a single threshold of 8 merged 33 different T2 leaves into one group and under-grouped T1; pair sheets reviewed visually during implementation (see `dedup.review_note` in `configs/data.yaml`).
- Result: S 10,407 images in 6,141 groups (largest 51); T1 5,932 images in 1,847 groups (largest 25); T2 1,106 in 1,099 groups. No cross-dataset leakage found (T1 and T2 vs Paddy Doctor: 0 images within the leakage threshold). Sethy is heavily duplicated: 5,174 exact-hash pairs.
- Splits (seed 42): S train/val/test 7,334/1,587/1,486 (the S split is by group, so the sizes are not exactly 70/15/15); T1 seg_train 4,074 / seg_val 927 / seg_test 280 (651 extra duplicate members dropped, from seg_test only); T2 field: cal_pool 135, test 201 (shared classes only: pool 79, test 119); T2-white 763 zero_shot; 7 T2 duplicates dropped.
- The pre-agreed T2 rules applied: allowed calibration sizes for T2 = [25, 50]; brown spot has 28 test images, flagged LOW-N (kept, bootstrap CIs, low-n flag in every T2 table).
- Cache: 16,787 images in `data/processed/` (S and T2 short side 256; T1 native with single-channel PNG masks; about 7% of Sethy files named .jpg are really PNGs and were re-encoded; Dhan EXIF rotation applied).

## 2026-10-19: M3 (label map, loaders, transforms)
- `label` column filled for all 17,445 manifest rows: S 10,407/10,407, T1 5,932/5,932, T2 606/1,106 (Leaf Scaled and Sheath Blight have no Paddy Doctor counterpart and stay empty).
- Loaders checked on every domain/split with real data: batches of (32, 3, 224, 224), class indices within the requested label space; T1 mask lesion share 4.9%; the training loader rejects anything but S/train.

## 2026-10-19: M4 (classifier training script)
- Implemented `classify/{models,evaluate,train}.py`; 106 tests pass. CPU `--debug` run on the real data (200 images per split, 1 epoch, EfficientNet-B0, pretrained): val macro-F1 0.067, test macro-F1 0.069, as expected for 3 training steps; it only proves the pipeline runs.
- Next: run `cls_effb0` (seed 0) on Kaggle and check the acceptance block (macro-F1 > 90%, no class recall < 80%).

## 2026-10-19: E1 in-domain baseline, EfficientNet-B0 baseline augmentation, seed 0 (Kaggle T4)
- Run `runs/cls_effb0-0-20261019-123812` (15 epochs, about 31 s per epoch, best epoch 14, pretrained ImageNet weights, mixed precision). The code was uploaded as a zip, so `git_commit` is `unknown`.
- S/test (1,486 images): accuracy 0.926, **macro-F1 0.9202**, balanced accuracy 0.9163. Acceptance check passed (macro-F1 > 0.90, lowest class recall 0.84 for brown_spot >= 0.80), but only narrowly.
- Per-class recall (low to high): brown_spot 0.840, hispa 0.858, bacterial_leaf_blight 0.863, tungro 0.894, bacterial_leaf_streak 0.907, bacterial_panicle_blight 0.927, normal 0.949, downy_mildew 0.953, blast 0.978, dead_heart 0.995. Largest confusions: hispa -> tungro (14), brown_spot -> bacterial_leaf_blight (11), normal -> hispa (8), hispa -> blast (7).
- Per-variety macro-F1 varies from 0.25 (Surya) to 1.0; test images per variety: ADT45 996, KarnatakaPonni 187, Ponni 98, Onthanel 57, AtchayaPonni 55, Zonal 48, AndraPonni 27, IR20 8, RR 5, Surya 5. Surya (0.25), RR and IR20 rest on 5-8 images and are noise; AtchayaPonni (0.574, 55 images) and KarnatakaPonni (0.762, 187 images) are the real per-variety weak spots.
- The Paddy Doctor paper reports F1 97.5% (ResNet34, 13 classes, no duplicate-aware split); our group-aware split is stricter and the class set differs, so the numbers are not directly comparable.
- Next: seeds 1 and 2, the `strong` and `strong_mixstyle` variants (M5), then the cross-dataset evaluation.

