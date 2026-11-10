import csv
import os
from pathlib import Path

import pytest

from app.glossary import (
    SENTINEL_PREFIX,
    GlossaryFormatError,
    GlossaryNotFound,
    GlossaryStore,
    Match,
    load_glossary_file,
    placeholder_for,
    restore_placeholders,
)

ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "glossary" / "deep-learning.csv"
GLOSSARY = load_glossary_file(CSV_PATH)

REQUIRED = {
    "back-propagation": "反向传播",
    "gradient descent": "梯度下降",
    "activation function": "激活函数",
    "loss function": "损失函数",
    "overfitting": "过拟合",
    "convolutional neural network": "卷积神经网络",
    "recurrent neural network": "循环神经网络",
    "transformer": "Transformer",
    "attention": "注意力",
    "batch normalization": "批量规范化",
    "dropout": "暂退法",
    "learning rate": "学习率",
    "epoch": "迭代周期",
    "optimizer": "优化器",
}

KEPT_LATIN = {"Adam", "ReLU", "Transformer", "Softmax", "Sigmoid", "Tanh"}


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def test_course_file_is_shared_csv_shape():
    raw = CSV_PATH.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.splitlines()[0] == b"en,zh,note"
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert 60 <= len(rows) <= 100
    assert len(rows) == len(GLOSSARY.terms)
    keys = [row["en"].casefold() for row in rows]
    assert len(keys) == len(set(keys))
    by_en = {term.en: term for term in GLOSSARY.terms}
    for row in rows:
        assert row["en"]
        assert row["zh"]
        assert row["zh"] in KEPT_LATIN or _has_cjk(row["zh"])
        assert by_en[row["en"]].zh == row["zh"]
        assert by_en[row["en"]].note == row["note"]
    for en, zh in REQUIRED.items():
        assert by_en[en].zh == zh
    assert by_en["dropout"].note


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("self-attention", ["self-attention"]),
        ("encoder-decoder", ["encoder-decoder"]),
        ("weight decay", ["weight decay"]),
        ("learning rate schedule", ["learning rate schedule"]),
        ("convolutional neural network", ["convolutional neural network"]),
        ("max pooling", ["max pooling"]),
        ("vanishing gradient", ["vanishing gradient"]),
        ("stochastic gradient descent", ["stochastic gradient descent"]),
        ("batch normalization", ["batch normalization"]),
    ],
)
def test_longest_phrase_wins(text: str, expected: list[str]):
    _protected, matches = GLOSSARY.protect(text)
    assert [match.en for match in matches] == expected


def test_case_plural_and_word_boundaries():
    protected, matches = GLOSSARY.protect("CNNs use Dropout across two epochs and biases.")
    assert [match.en for match in matches] == ["CNN", "dropout", "epoch", "bias"]
    assert [match.matched for match in matches] == ["CNNs", "Dropout", "epochs", "biases"]
    assert "CNN" not in protected
    assert "epochal" == GLOSSARY.protect("epochal")[0]
    assert GLOSSARY.protect("attentional")[1] == []
    assert GLOSSARY.protect("The relational plot.")[1] == []
    _protected, spaced = GLOSSARY.protect("batch  normalization")
    assert [match.en for match in spaced] == ["batch normalization"]


@pytest.mark.parametrize(
    "spelling",
    ["backpropagation", "back propagation", "back-propagation", "BackPropagations"],
)
def test_hyphen_space_and_closed_spellings(spelling: str):
    _protected, matches = GLOSSARY.protect(spelling)
    assert [(match.en, match.zh) for match in matches] == [("back-propagation", "反向传播")]


def test_plural_es_and_spaced_hyphen_terms():
    assert [match.en for match in GLOSSARY.protect("mini-batches")[1]] == ["mini-batch"]
    assert [match.en for match in GLOSSARY.protect("mini batches")[1]] == ["mini-batch"]
    assert [match.en for match in GLOSSARY.protect("fine tuning")[1]] == ["fine-tuning"]
    assert [match.en for match in GLOSSARY.protect("fine-tunings")[1]] == ["fine-tuning"]
    assert [match.en for match in GLOSSARY.protect("multi head attention")[1]] == ["multi-head attention"]
    assert [match.en for match in GLOSSARY.protect("cross entropy")[1]] == ["cross-entropy"]
    assert [match.en for match in GLOSSARY.protect("pretraining")[1]] == ["pre-training"]
    assert [match.en for match in GLOSSARY.protect("backprop")[1]] == ["backprop"]
    assert [match.en for match in GLOSSARY.protect("loss functions")[1]] == ["loss function"]


def test_loader_rejects_bad_rows(tmp_path: Path):
    missing = tmp_path / "bad-header.csv"
    missing.write_text("en,zh\nfoo,甲\n", encoding="utf-8")
    with pytest.raises(GlossaryFormatError, match="header"):
        load_glossary_file(missing)

    duplicate = tmp_path / "dup.csv"
    duplicate.write_text("en,zh,note\nFoo,甲,\nfoo,乙,\n", encoding="utf-8")
    with pytest.raises(GlossaryFormatError, match="duplicate"):
        load_glossary_file(duplicate)

    sentinel = tmp_path / "sentinel.csv"
    sentinel.write_text(f"en,zh,note\nfoo{placeholder_for(0)},甲,\n", encoding="utf-8")
    with pytest.raises(GlossaryFormatError, match="sentinel"):
        load_glossary_file(sentinel)


def test_loader_accepts_bom_crlf_and_extra_columns(tmp_path: Path):
    path = tmp_path / "course.csv"
    path.write_bytes(
        "\ufeffen,zh,note,alias\r\nfoo,甲,\"a, b\",extra\r\n\r\n".encode("utf-8")
    )
    glossary = load_glossary_file(path)
    assert glossary.name == "course"
    assert glossary.terms[0].zh == "甲"
    assert glossary.terms[0].note == "a, b"


def test_store_reloads_and_keeps_last_good_file(tmp_path: Path):
    path = tmp_path / "demo.csv"
    path.write_text("en,zh,note\nfoo,甲,\n", encoding="utf-8")
    store = GlossaryStore(tmp_path)
    assert store.get("Demo").protect("foo")[1][0].zh == "甲"

    path.write_text("en,zh,note\nfoo,乙,\n", encoding="utf-8")
    later = path.stat().st_mtime_ns + 1_000_000
    os.utime(path, ns=(later, later))
    assert store.get("demo").protect("foo")[1][0].zh == "乙"

    path.write_text("en,zh\nfoo,丙\n", encoding="utf-8")
    broken = path.stat().st_mtime_ns + 1_000_000
    os.utime(path, ns=(broken, broken))
    assert store.get("demo").protect("foo")[1][0].zh == "乙"

    with pytest.raises(GlossaryNotFound) as exc:
        store.get("missing")
    assert exc.value.available == ["demo"]


def test_existing_sentinel_index_is_not_reused():
    sentinel = placeholder_for(0)
    protected, matches = GLOSSARY.protect(f"{sentinel} uses dropout.")
    assert matches[0].en == "dropout"
    assert matches[0].placeholder != sentinel
    assert sentinel in protected


def test_restore_accepts_marian_z_transliteration():
    first = Match(0, 4, placeholder_for(0), "dropout", "暂退法", "", "dropout")
    second = Match(5, 9, placeholder_for(1), "pooling", "汇聚", "", "pooling")
    restored, missing = restore_placeholders("兹XQ0层依ZXQ1而变化。", [first, second])
    assert missing == []
    assert restored == "暂退法层依汇聚而变化。"
    assert SENTINEL_PREFIX not in restored
    assert "兹XQ" not in restored


def test_first_load_of_a_broken_directory_fails(tmp_path: Path):
    path = tmp_path / "demo.csv"
    path.write_text("nope\n", encoding="utf-8")
    store = GlossaryStore(tmp_path)
    with pytest.raises(GlossaryFormatError):
        store.get("demo")
