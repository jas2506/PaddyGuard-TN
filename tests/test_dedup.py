import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from paddyguard.data import dedup


def _smooth_image(seed: int, size: tuple[int, int] = (96, 72)) -> Image.Image:
    """Low-frequency random picture: distinct seeds give clearly different perceptual hashes."""
    coarse = np.random.default_rng(seed).integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(coarse).resize(size, Image.BICUBIC)


def test_popcount_matches_python() -> None:
    x = np.array([0, 1, 255, 2**63, 2**64 - 1], dtype=np.uint64)
    assert dedup.popcount(x).tolist() == [int(v).bit_count() for v in x]


def test_near_pairs_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**63, 6, dtype=np.uint64)
    hashes = np.concatenate([base, base ^ np.uint64(0b111), base ^ np.uint64(0xFFFF)])  # near + far
    got = {tuple(r) for r in dedup.near_pairs(hashes, max_dist=4, chunk_rows=4).tolist()}
    want = {
        (i, j, (int(hashes[i]) ^ int(hashes[j])).bit_count())
        for i, j in itertools.combinations(range(len(hashes)), 2)
        if (int(hashes[i]) ^ int(hashes[j])).bit_count() <= 4
    }
    assert got == want


def test_cross_pairs() -> None:
    a = np.array([0b0000, 0b1111_0000], dtype=np.uint64)
    b = np.array([0b0001, 0b1111_1111], dtype=np.uint64)
    got = {tuple(r) for r in dedup.cross_pairs(a, b, max_dist=1).tolist()}
    assert got == {(0, 0, 1)}


def test_cluster_links_chains_and_leaves_isolates() -> None:
    labels = dedup.cluster(6, np.array([[0, 1, 1], [1, 2, 1], [4, 5, 2]]))
    assert labels[0] == labels[1] == labels[2]
    assert labels[4] == labels[5] and labels[3] not in (labels[0], labels[4])
    assert len(set(labels.tolist())) == 3


def test_planted_exact_and_resized_duplicates_are_found(tmp_path: Path) -> None:
    originals = [_smooth_image(s) for s in range(5)]
    paths = []
    for i, im in enumerate(originals):
        p = tmp_path / f"o{i}.jpg"
        im.save(p, quality=95)
        paths.append(p)
    (tmp_path / "exact.jpg").write_bytes(paths[0].read_bytes())  # exact duplicate of image 0
    originals[1].resize((60, 45)).save(
        tmp_path / "resized.jpg", quality=80
    )  # resized duplicate of 1
    paths += [tmp_path / "exact.jpg", tmp_path / "resized.jpg"]
    hashes = dedup.hex_to_uint64([dedup.phash_hex(p, draft_size=256, hash_size=8) for p in paths])
    pairs = {(int(i), int(j)) for i, j, _ in dedup.near_pairs(hashes, max_dist=8, chunk_rows=4)}
    assert (0, 5) in pairs and (1, 6) in pairs
    assert pairs == {(0, 5), (1, 6)}  # nothing else is called a duplicate


def test_phash_respects_exif_orientation(tmp_path: Path) -> None:
    im = _smooth_image(3, (96, 48))
    plain, rotated = tmp_path / "plain.jpg", tmp_path / "rot.jpg"
    im.save(plain, quality=95)
    exif = Image.Exif()
    exif[274] = 6  # "rotate 90° clockwise to display"
    im.rotate(90, expand=True).save(rotated, quality=95, exif=exif)  # stored sideways
    h = [dedup.phash_hex(p, draft_size=256, hash_size=8) for p in (plain, rotated)]
    a, b = dedup.hex_to_uint64(h)
    assert dedup.popcount(np.array([a ^ b]))[0] <= 8


def _manifest(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    """(image_id, dataset, hash-as-int, label_raw)"""
    return pd.DataFrame(
        {
            "image_id": [r[0] for r in rows],
            "dataset": [r[1] for r in rows],
            "label_raw": [r[3] for r in rows],
            "phash": [f"{r[2]:016x}" for r in rows],
            "group_id": None,
            "split": None,
        }
    )


def test_groups_drops_and_leakage() -> None:
    far = 0xFFFF_0000_0000_0000
    df = _manifest(
        [
            ("S_000000", "S", 0x0, "a"),
            ("S_000001", "S", 0x1, "a"),  # duplicate of S_000000: same group, both kept
            ("S_000002", "S", far, "a"),
            ("T1_000000", "T1", 0x00F0_F0F0_0000_0000, "x"),
            (
                "T1_000001",
                "T1",
                0x00F0_F0F0_0000_0001,
                "x",
            ),  # duplicate of T1_000000: kept, same group
            ("T1_000002", "T1", 0x2, "x"),  # within threshold of S_000000: leakage
            ("T2_000000", "T2", 0x0F0F_0F0F_0F0F_0F0F, "y"),
            ("T2_000001", "T2", 0x0F0F_0F0F_0F0F_0F0E, "y"),  # duplicate of T2_000000: dropped
        ]
    )
    out, summary = dedup.assign_groups(
        df, thresholds={"S": 2, "T1": 2, "T2": 2}, leakage_threshold=2, hist_max=4, chunk_rows=8
    )
    g = out.set_index("image_id")
    assert (
        g.loc["S_000000", "group_id"]
        == g.loc["S_000001", "group_id"]
        != g.loc["S_000002", "group_id"]
    )
    assert g.loc[["S_000000", "S_000001", "S_000002"], "split"].isna().all()
    assert g.loc["T1_000000", "group_id"] == g.loc["T1_000001", "group_id"] == "T1_000000"
    assert pd.isna(
        g.loc["T1_000001", "split"]
    )  # T1 duplicates stay (the splitter handles seg_test)
    assert g.loc["T2_000001", "split"] == dedup.DROPPED_DUPLICATE
    assert dedup.is_representative(out).tolist() == [
        True,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
    ]
    assert g.loc["T1_000002", "split"] == dedup.DROPPED_LEAKAGE
    assert pd.isna(g.loc["T1_000000", "split"]) and pd.isna(g.loc["T2_000000", "split"])
    assert summary["leakage"] == {"T1": 1, "T2": 0}


def test_boundary_pairs_come_from_both_sides_of_the_threshold() -> None:
    pairs = np.array([[0, 1, d] for d in range(13)])
    chosen = dedup.boundary_pairs(pairs, threshold=8, n=6, seed=0)
    d = sorted(chosen[:, 2].tolist())
    assert d == [6, 7, 8, 9, 10, 11]  # 3 closest at or below 8, 3 closest above 8
