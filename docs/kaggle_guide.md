# Running the training on Kaggle (step by step)

The laptop has no GPU, so training runs on a Kaggle notebook. The notebook is thin: it installs the
package and calls `python -m paddyguard.classify.train`.

## One-time setup

1. **Verify your Kaggle account with a phone number** (Settings, Phone verification). Kaggle only
   gives GPU and Internet access to verified accounts.
2. **Build the two upload files** on the laptop: `./scripts/package_for_kaggle.sh`. It writes
   `data/kaggle_upload/paddyguard-processed.zip` (about 750 MB: the manifest and the resized
   images) and `data/kaggle_upload/paddyguard-code.zip` (the package and configs).
3. **Upload them as two private datasets** (Datasets, New Dataset, drag the zip in, set the title,
   keep it Private, Create). Kaggle unzips them itself. Use the titles `paddyguard-processed` and
   `paddyguard-code`. Wait until each one finishes processing.
   The three raw datasets you already uploaded (RiceSeg, Sethy, Dhan-Shomadhan) are not needed for
   training; the processed images already contain what the code reads. Keep them as a backup.
4. **Create the notebook:** Code, New Notebook, File, Import Notebook, upload
   `notebooks/kaggle_run.ipynb`.
5. In the notebook's right-hand panel: **Add Input** (Your Datasets) and add both datasets;
   **Settings**: Accelerator = GPU (T4 or P100), Internet = On.

## Every run

1. Run the setup cells, then the **quick check** cell. It takes about a minute and must finish
   without an error.
2. Run the **real run** cell (EfficientNet-B0, 15 epochs, roughly 20 to 30 minutes).
3. Run the **results** cell. It prints the test macro-F1 and whether the acceptance check passed
   (above 90% macro-F1 and no class below 80% recall).
4. **Keep the outputs:** use Save Version, then Save and Run All (Commit). Files written to
   `/kaggle/working` are lost when an interactive session ends; a committed version keeps them.
   Then open the version's Output tab and download the `runs` folder.
5. Put the downloaded run folder (for example `cls_effb0-0-2026...`) into `runs/` in the project.
   It contains `metrics.json`, `preds_S_test.parquet` and `ckpt.pt`, which are what later modules
   read.

For the other seeds, change `--seed 0` to `--seed 1` and `--seed 2`.

## If something changes

- **Code or configs changed:** run `./scripts/package_for_kaggle.sh` again and upload the new zip as a
  new version of the matching dataset (Dataset page, New Version). Re-open the notebook so it picks
  up the latest version.
- **Session stopped mid-run:** run the setup cells again, then the resume cell with your run folder
  name.
- **GPU quota:** Kaggle gives a weekly GPU allowance (about 30 hours at the time of writing; check
  your profile). One classifier run is well under an hour.

## Troubleshooting

- **`pip subprocess to install build dependencies did not run successfully`:** this came from
  `pip install -e .`, which the notebook no longer uses. The setup cell now sets `PYTHONPATH` instead.
  Replace the setup cell's contents with the version in `notebooks/kaggle_run.ipynb`.
- **`timm` or `albumentations` cannot be installed:** the notebook's Internet setting is off. Turn it on
  in the right-hand Settings panel (it needs the phone-verified account).

