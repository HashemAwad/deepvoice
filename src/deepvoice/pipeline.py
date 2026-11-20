"""Build every cache the notebooks read, in order. Each step loads its cache if it exists, so
re-running is cheap; from scratch it takes about 3 hours on an Apple M5 (mostly XLS-R).

    uv run --extra dl python -m deepvoice.pipeline
"""

from pathlib import Path

import pandas as pd

from deepvoice.data import load_itw, load_protocols
from deepvoice.features import feature_tables
from deepvoice.silence import silence_table


def build_all(root="data", xlsr=True):
    root = Path(root)
    proto_file = root / "protocols.parquet"
    if not proto_file.exists():
        load_protocols(root / "LA").to_parquet(proto_file)
    p = pd.read_parquet(proto_file)
    sets = {s: p[p.split == s].reset_index(drop=True) for s in ["train", "dev", "eval"]}
    sets["itw"] = load_itw(root / "release_in_the_wild").reset_index(drop=True)
    feats = root / "features"
    for s, df in sets.items():
        silence_table(df, feats / f"silence_{s}.parquet")
        feature_tables(df, s, root=feats)
    for change in ["trim", "gain-12", "gain+6"]:  # E5 test-time changes
        for s in ["eval", "itw"]:
            feature_tables(sets[s], f"{s}_{change}", root=feats, transform=change)
    feature_tables(sets["train"], "train_augment", root=feats, transform="augment")  # E5 mitigation
    if xlsr:
        from deepvoice import xlsr as xl

        model = None
        jobs = [(s, None) for s in sets] + [("eval_trim", "trim"), ("itw_trim", "trim")]
        for name, change in jobs:
            if not xl.cache_path(name, feats).exists():
                model = model or xl.load_model()
            xl.extract_cached(sets[name.split("_")[0]], name, root=feats, model=model,
                              transform=change)


if __name__ == "__main__":
    build_all()
    print("all caches ready")
