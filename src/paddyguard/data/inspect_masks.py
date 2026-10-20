"""Week-1 discovery for the RiceSeg masks (M1, blocking for M6).

The mask encoding is not documented, so this script only reports facts (modes, pixel values,
histograms) and saves photo | mask | blend overlays. It does not decide what the values mean:
the developer answers the questions in the note after looking at the overlays.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from paddyguard.utils import io, paths

MAX_PIXEL = 255
BLEND = 0.5
QUESTIONS = [
    (
        "Does a mask mark only lesions, or leaf and lesion with different values?",
        lambda e: f"{e['marks']} (lesion value {e['lesion_value']})",
    ),
    (
        "Is the background pure black or pure white?",
        lambda e: f"pixel value {e['background_value']}",
    ),
    (
        "Which threshold cleans the JPEG noise?",
        lambda e: f"{e['threshold']} (pixel >= {e['threshold']} counts as lesion)",
    ),
]


def mask_stats(arr: np.ndarray, round_to: int, hist_bins: int) -> dict[str, Any]:
    """Facts about one mask array (H x W grey, or H x W x 3 colour)."""
    colour = arr.ndim == 3
    mode = "RGB" if colour else "L"
    channels_equal = None
    if colour:
        channels_equal = bool(
            (arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all()
        )
    grey = arr[..., 0] if colour and channels_equal else arr
    values = arr.reshape(-1, 3) if colour and not channels_equal else grey.reshape(-1, 1)
    rounded = (np.round(values / round_to) * round_to).astype(int)
    keys = [tuple(v) if len(v) > 1 else int(v[0]) for v in rounded.tolist()]
    intensity = grey if channels_equal or not colour else arr.mean(axis=2)
    histogram, _ = np.histogram(intensity, bins=hist_bins, range=(0, MAX_PIXEL + 1))
    return {
        "mode": mode,
        "channels_equal": channels_equal,
        "rounded_values": dict(Counter(keys)),
        "histogram": histogram.tolist(),
        "n_pixels": int(values.shape[0]),
    }


def make_overlay(photo: Image.Image, mask: Image.Image) -> Image.Image:
    """photo | mask | 50% blend, side by side. Makes no assumption about what the values mean."""
    photo = photo.convert("RGB")
    mask = mask.convert("RGB").resize(photo.size, Image.NEAREST)
    blend = Image.blend(photo, mask, BLEND)
    sheet = Image.new("RGB", (3 * photo.width, photo.height))
    for i, im in enumerate((photo, mask, blend)):
        sheet.paste(im, (i * photo.width, 0))
    return sheet


def _sample_rows(manifest: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    pool = manifest[manifest["mask_path"].notna()]
    # keep the random order: overlays are the first n_overlays rows, so they must mix classes
    return pool.sample(n=min(n, len(pool)), random_state=seed)


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im)


def _aggregate(stats: list[dict[str, Any]], cfg: Mapping[str, Any]) -> str:
    n = len(stats)
    modes = Counter(s["mode"] for s in stats)
    equal = Counter(s["channels_equal"] for s in stats if s["mode"] == "RGB")
    per_mask = [len(s["rounded_values"]) for s in stats]
    total = Counter()
    for s in stats:
        total.update(s["rounded_values"])
    pixels = sum(total.values())
    hist = np.sum([s["histogram"] for s in stats], axis=0)
    edges = np.linspace(0, MAX_PIXEL + 1, cfg["hist_bins"] + 1).astype(int)
    lines = [
        f"- Masks examined: {n}",
        f"- Image modes: {dict(modes)}"
        + (f"; RGB masks with identical channels: {dict(equal)}" if equal else ""),
        (
            f"- Distinct rounded values per mask (rounded to a multiple of {cfg['round_to']}): "
            f"min {min(per_mask)}, median {int(np.median(per_mask))}, max {max(per_mask)}"
        ),
        "",
        f"### Most common rounded values, all {n} masks together",
        "",
        "| Rounded value | Share of pixels |",
        "|---|---|",
    ]
    lines += [f"| {k} | {100 * v / pixels:.2f}% |" for k, v in total.most_common(cfg["top_values"])]
    lines += [
        "",
        "### Pixel-intensity histogram (grey value or channel mean)",
        "",
        "| Intensity range | Share of pixels |",
        "|---|---|",
    ]
    lines += [
        f"| {edges[i]}–{edges[i + 1] - 1} | {100 * h / pixels:.2f}% |" for i, h in enumerate(hist)
    ]
    return "\n".join(lines)


def _answer(answer_for: Any, encoding: Mapping[str, Any] | None) -> str:
    if encoding is None:
        return "TO BE FILLED by the developer after viewing the overlays"
    return f"{answer_for(encoding)} (given by the developer, from `mask_encoding` in configs/data.yaml)"


def write_note(
    stats: list[dict[str, Any]],
    overlay_dir: Path,
    cfg: Mapping[str, Any],
    note_path: Path,
    encoding: Mapping[str, Any] | None = None,
) -> None:
    questions = "\n".join(
        f"{i}. {q}\n   - Answer: {_answer(answer_for, encoding)}"
        for i, (q, answer_for) in enumerate(QUESTIONS, 1)
    )
    heading = "Answers" if encoding else "Questions (M6 must not start until these are answered)"
    text = (
        "# RiceSeg mask inspection\n\n"
        "Generated by `python -m paddyguard.data.inspect_masks`. The numbers below are facts read from "
        "the files; what the values *mean* is not decided by the script.\n\n"
        f"Overlays (photo | mask | 50% blend): `{overlay_dir}`\n\n"
        f"## Pixel facts\n\n{_aggregate(stats, cfg)}\n\n"
        f"## {heading}\n\n{questions}\n"
    )
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(text)


def run_inspection(
    manifest: pd.DataFrame,
    data_root: Path,
    cfg: Mapping[str, Any],
    seed: int,
    overlay_dir: Path,
    note_path: Path,
    encoding: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Examine n_masks random masks, save n_overlays overlays, write the note."""
    rows = _sample_rows(manifest, cfg["n_masks"], seed)
    stats = [
        mask_stats(_load_mask(data_root / m), cfg["round_to"], cfg["hist_bins"])
        for m in rows["mask_path"]
    ]
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for r in list(rows.itertuples())[: cfg["n_overlays"]]:
        with Image.open(data_root / r.path) as photo, Image.open(data_root / r.mask_path) as mask:
            make_overlay(photo, mask).save(overlay_dir / f"{r.image_id}.png")
    write_note(stats, overlay_dir, cfg, note_path, encoding)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(paths.configs_dir() / "data.yaml"))
    parser.add_argument(
        "--debug", action="store_true", help="use manifest_debug.csv and *_debug outputs"
    )
    args = parser.parse_args(argv)
    cfg = io.load_yaml(args.config)
    suffix = "_debug" if args.debug else ""
    data_root = paths.data_root()
    manifest_file = data_root / cfg["manifest_path"]
    manifest = pd.read_csv(
        manifest_file.with_name(manifest_file.stem + suffix + manifest_file.suffix)
    )
    mi = cfg["mask_inspect"]
    overlay_dir = paths.reports_root() / (mi["overlay_dir"] + suffix)
    note = paths.reports_root() / mi["note_path"]
    note = note.with_name(note.stem + suffix + note.suffix)
    run_inspection(
        manifest, data_root, mi, cfg["seed"], overlay_dir, note, cfg.get("mask_encoding")
    )
    print(f"overlays: {overlay_dir}\nnote: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
