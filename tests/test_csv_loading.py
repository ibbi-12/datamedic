"""Tests for defensive CSV loading against unseen/malformed files.

profile_csv runs before any LLM call and sits outside the self-healing retry
loop, so a raw pandas exception here is terminal for the whole job. These
cases are real failure modes for CSVs an agent has never seen before.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.nodes import CSVLoadError, load_csv_robust, profile_csv


def test_loads_non_utf8_encoding(tmp_path):
    path = tmp_path / "latin1.csv"
    path.write_bytes("name,city\nJos\xe9,S\xe3o Paulo\n".encode("latin-1"))
    df = load_csv_robust(str(path))
    assert df.shape == (1, 2)
    assert df["name"].iloc[0] == "José"


def test_loads_utf8_bom(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbfa,b\n1,2\n")
    df = load_csv_robust(str(path))
    assert list(df.columns) == ["a", "b"]


def test_recovers_ragged_rows(tmp_path):
    path = tmp_path / "ragged.csv"
    path.write_text("a,b,c\n1,2,3\n4,5\n6,7,8,9\n")
    df = load_csv_robust(str(path))
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) >= 1  # bad lines skipped, not a crash


def test_sniffs_semicolon_delimiter(tmp_path):
    path = tmp_path / "semi.csv"
    path.write_text("a;b;c\n1;2;3\n4;5;6\n")
    df = load_csv_robust(str(path))
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (2, 3)


def test_sniffs_tab_delimiter(tmp_path):
    path = tmp_path / "tabs.csv"
    path.write_text("a\tb\tc\n1\t2\t3\n")
    df = load_csv_robust(str(path))
    assert list(df.columns) == ["a", "b", "c"]


def test_single_column_csv_not_mangled_by_sniffing(tmp_path):
    """csv.Sniffer can't find a delimiter in genuinely single-column data —
    it must not be allowed to corrupt an otherwise-fine parse."""
    path = tmp_path / "single.csv"
    path.write_text("a\n1\n2\n3\n")
    df = load_csv_robust(str(path))
    assert list(df.columns) == ["a"]
    assert df.shape == (3, 1)


def test_empty_file_raises_clean_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(CSVLoadError):
        load_csv_robust(str(path))


def test_profile_csv_node_normalizes_file_on_disk(tmp_path):
    """After profile_csv runs, the on-disk file must be plain comma/UTF-8 —
    the generated code always does a bare pd.read_csv() on it."""
    path = tmp_path / "data.csv"
    path.write_bytes("a;b\n1;2\n3;4\n".encode("utf-8"))
    state = {"csv_path": str(path)}
    out = profile_csv(state)
    assert "shape: 2 rows x 2 columns" in out["csv_profile"]
    reread = pd.read_csv(path)  # plain default read, as the generated code would do
    assert list(reread.columns) == ["a", "b"]
    assert reread.shape == (2, 2)
