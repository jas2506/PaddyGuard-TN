#!/usr/bin/env bash
# Builds the two zip files to upload to Kaggle as private datasets:
#   data/kaggle_upload/paddyguard-processed.zip  manifest + resized images (what training reads)
#   data/kaggle_upload/paddyguard-code.zip       the package and configs (what the notebook installs)
# Re-run and upload a new dataset version whenever the code or the manifest changes.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/data/kaggle_upload"
mkdir -p "$out"
rm -f "$out/paddyguard-processed.zip" "$out/paddyguard-code.zip"

( cd "$root/data" && zip -qr "$out/paddyguard-processed.zip" manifests/manifest.csv processed )
( cd "$root" && zip -qr "$out/paddyguard-code.zip" src configs pyproject.toml requirements.txt \
    -x '*/__pycache__/*' '*.pyc' '*.egg-info/*' )

ls -lh "$out"/*.zip
