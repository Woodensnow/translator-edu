"""Glossary CSV loading. Uses the fixture file, never glossary/deep-learning.csv."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from asr.glossary import (
    DEFAULT_GLOSSARY_PATH,
    GlossaryFormatError,
    GlossaryPrompt,
    build_hotwords,
    build_initial_prompt,
    english_terms,
    glossary_path_from_env,
    load_english_terms,
    load_glossary,
)
from support import FIXTURE


def test_fixture_uses_english_header_and_empty_note():
    entries = load_glossary(FIXTURE)
    assert [(item.en, item.zh, item.note) for item in entries] == [
        ("backpropagation", "反向传播", "core training term"),
        ("gradient descent", "梯度下降", ""),
        ("activation function", "激活函数", "includes ReLU and sigmoid"),
    ]
    assert load_english_terms(FIXTURE) == [
        "backpropagation",
        "gradient descent",
        "activation function",
    ]


def test_prompt_and_hotwords_use_english_column_only():
    terms = load_english_terms(FIXTURE)
    prompt = build_initial_prompt(terms)
    hotwords = build_hotwords(terms)
    for term in terms:
        assert term in prompt
        assert term in hotwords
    for excluded in ("反向传播", "梯度下降", "激活函数", "ReLU", "sigmoid"):
        assert excluded not in prompt
        assert excluded not in hotwords


def test_chinese_header_bom_and_blank_note(tmp_path):
    path = tmp_path / "course.csv"
    path.write_bytes("\ufeff英文,中文,备注\r\nneural network,神经网络,\r\n".encode("utf-8"))
    entries = load_glossary(path)
    assert len(entries) == 1
    assert entries[0].en == "neural network"
    assert entries[0].zh == "神经网络"
    assert entries[0].note == ""
    prompt = build_initial_prompt(english_terms(entries))
    assert "neural network" in prompt
    assert "神经网络" not in prompt


def test_english_header_is_case_insensitive_and_notes_may_contain_commas(tmp_path):
    path = tmp_path / "course.csv"
    path.write_text(
        'EN, ZH, NOTE\ntransformer,变换器,"attention, encoder"\n',
        encoding="utf-8",
    )
    entries = load_glossary(path)
    assert entries[0].en == "transformer"
    assert entries[0].note == "attention, encoder"
    prompt = build_initial_prompt(english_terms(entries))
    assert "transformer" in prompt
    assert "encoder" not in prompt
    assert "变换器" not in prompt


def test_missing_note_column_is_empty(tmp_path):
    path = tmp_path / "course.csv"
    path.write_text("en,zh\nbackpropagation,反向传播\n", encoding="utf-8")
    entries = load_glossary(path)
    assert entries == load_glossary(path)
    assert entries[0].note == ""
    assert "反向传播" not in build_initial_prompt(english_terms(entries))


def test_skips_blank_english_and_duplicate_terms(tmp_path):
    path = tmp_path / "course.csv"
    path.write_text(
        "en,zh,note\n,神经网络,skip\nbackpropagation,反向传播,a\nbackpropagation,反向传播二,b\n",
        encoding="utf-8",
    )
    assert load_english_terms(path) == ["backpropagation"]


def test_missing_file_warns_and_returns_no_terms(tmp_path, caplog):
    missing = tmp_path / "deep-learning.csv"
    with caplog.at_level(logging.WARNING):
        assert load_glossary(missing) == []
    assert "ASR_GLOSSARY_PATH" in caplog.text
    assert "glossary/deep-learning.csv" in caplog.text
    assert str(missing) in caplog.text


def test_bad_header_raises(tmp_path):
    path = tmp_path / "course.csv"
    path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(GlossaryFormatError, match="en,zh,note"):
        load_glossary(path)


def test_default_path_and_override():
    assert DEFAULT_GLOSSARY_PATH == "glossary/deep-learning.csv"
    assert glossary_path_from_env({}) == Path("glossary/deep-learning.csv")
    assert glossary_path_from_env({"ASR_GLOSSARY_PATH": "glossary/nlp.csv"}) == Path("glossary/nlp.csv")


def test_prompt_reloads_when_the_csv_changes(tmp_path):
    path = tmp_path / "course.csv"
    path.write_text("en,zh,note\nbackpropagation,反向传播,\n", encoding="utf-8")
    prompts = GlossaryPrompt(path)
    assert "backpropagation" in prompts.prompt_for("")
    assert "反向传播" not in prompts.prompt_for("")
    path.write_text("en,zh,note\ngradient descent,梯度下降,\n", encoding="utf-8")
    os.utime(path, (path.stat().st_atime + 10, path.stat().st_mtime + 10))
    refreshed = prompts.prompt_for("")
    assert "gradient descent" in refreshed
    assert "backpropagation" not in refreshed
    assert "梯度下降" not in refreshed
    assert prompts.current_hotwords() == "gradient descent"


def test_prompt_truncates_to_budget_and_keeps_first_terms():
    terms = [f"term{i:03d}" + ("x" * 24) for i in range(40)]
    prompt = build_initial_prompt(terms, previous="P" * 800, max_chars=180)
    assert len(prompt) <= 180
    assert "term000" in prompt
    hotwords = build_hotwords(terms, max_chars=80)
    assert hotwords is not None
    assert len(hotwords) <= 80
    assert build_hotwords([]) is None
    assert build_initial_prompt([]) == "English technical lecture."
