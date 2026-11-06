import pytest

from deepvoice.data import SPLITS, load_protocols, read_protocol, summarize

SAMPLE = """LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1271820 - - bonafide
LA_0079 LA_T_1272637 - A01 spoof
LA_0092 LA_T_2334921 - A02 spoof
"""


def test_read_protocol(tmp_path):
    p = tmp_path / "proto.txt"
    p.write_text(SAMPLE)
    df = read_protocol(p)
    assert list(df["label"]) == [0, 0, 1, 1]
    assert list(df["attack"]) == ["bonafide", "bonafide", "A01", "A02"]
    assert df["speaker"].nunique() == 2


def test_read_protocol_rejects_unknown_key(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("LA_0079 LA_T_1 - - maybe\n")
    with pytest.raises(ValueError):
        read_protocol(p)


def test_load_protocols_and_summarize(tmp_path):
    proto_dir = tmp_path / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir()
    (proto_dir / SPLITS["train"][1]).write_text(SAMPLE)
    (proto_dir / SPLITS["dev"][1]).write_text("LA_0069 LA_D_1 - - bonafide\nLA_0069 LA_D_2 - A03 spoof\n")
    df = load_protocols(tmp_path, splits=("train", "dev"))
    assert list(df.columns) == ["file", "speaker", "attack", "label", "split", "path"]
    assert df["path"].iloc[0].endswith("ASVspoof2019_LA_train/flac/LA_T_1138215.flac")
    s = summarize(df).set_index("split")
    assert s.loc["train", ["real", "fake", "total", "speakers"]].tolist() == [2, 2, 4, 2]
    assert s.loc["train", "attacks"] == "A01-A02 (2)"
    assert s.loc["dev", "speakers"] == 1


def test_load_itw(tmp_path):
    from deepvoice.data import load_itw

    (tmp_path / "meta.csv").write_text("file,speaker,label\n0.wav,A B,spoof\n1.wav,A B,bona-fide\n")
    df = load_itw(tmp_path)
    assert list(df["file"]) == ["0", "1"] and list(df["label"]) == [1, 0]
    assert list(df["attack"]) == ["unknown", "bonafide"] and set(df["split"]) == {"itw"}
    assert df["path"].iloc[0].endswith("0.wav")
