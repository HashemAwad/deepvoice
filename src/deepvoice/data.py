"""ASVspoof 2019 LA protocol loading. Label 1 = fake (spoof), 0 = real (bonafide)."""

from pathlib import Path

import pandas as pd

COLUMNS = ["speaker", "utt", "env", "attack", "key"]

# Names checked against the real LA.zip contents (2026-11-05). Always iterate over these protocol
# files, never glob the flac folders: dev/eval folders hold extra files (ASV enrollment audio)
# that are not in the CM protocols (eval: 71,933 files vs 71,237 protocol rows).
SPLITS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt"),
}


def read_protocol(path):
    """Parse one protocol file: `speaker utt env attack key`, key in {bonafide, spoof}."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    bad = set(df["key"]) - {"bonafide", "spoof"}
    if bad:
        raise ValueError(f"unexpected key values in {path}: {bad}")
    df["label"] = (df["key"] == "spoof").astype(int)
    df["attack"] = df["attack"].replace("-", "bonafide")
    return df


def load_split(root, split):
    """Protocol dataframe for train/dev/eval with an added `path` column to each .flac."""
    folder, proto = SPLITS[split]
    root = Path(root)
    df = read_protocol(root / "ASVspoof2019_LA_cm_protocols" / proto)
    df["path"] = [str(root / folder / "flac" / f"{u}.flac") for u in df["utt"]]
    df["split"] = split
    return df


def load_protocols(root, splits=("train", "dev", "eval")):
    """One table for all splits: file, speaker, attack, label, split, path.

    Reads protocol text only (no audio), so listing eval here does not count as using it.
    """
    df = pd.concat([load_split(root, s) for s in splits], ignore_index=True)
    df = df.rename(columns={"utt": "file"})
    return df[["file", "speaker", "attack", "label", "split", "path"]]


def summarize(df):
    """Per split: real/fake counts, total, number of speakers and the attack IDs present."""
    rows = []
    for split, g in df.groupby("split", sort=False):
        attacks = sorted(set(g["attack"]) - {"bonafide"})
        rows.append(
            {
                "split": split,
                "real": int((g["label"] == 0).sum()),
                "fake": int((g["label"] == 1).sum()),
                "total": len(g),
                "speakers": g["speaker"].nunique(),
                "attacks": f"{attacks[0]}-{attacks[-1]} ({len(attacks)})" if attacks else "",
            }
        )
    return pd.DataFrame(rows)
