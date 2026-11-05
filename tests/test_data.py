import pytest

from deepvoice.data import read_protocol

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
