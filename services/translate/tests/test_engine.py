from pathlib import Path

from app.engine import split_for_translation, translate_texts
from app.glossary import SENTINEL_PREFIX, load_glossary_file, placeholder_for

ROOT = Path(__file__).resolve().parents[3]
GLOSSARY = load_glossary_file(ROOT / "glossary" / "deep-learning.csv")


class Echo:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def translate_many(self, texts: list[str]) -> list[str]:
        self.batches.append(list(texts))
        return list(texts)


class Drop:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def translate_many(self, texts: list[str]) -> list[str]:
        self.batches.append(list(texts))
        return ["XXX" for _ in texts]


def test_mock_path_substitutes_terms_and_keeps_other_words():
    echo = Echo()
    result = translate_texts(
        ["Backpropagation uses gradient descent to train the neural network."],
        GLOSSARY,
        echo,
    )[0]
    assert echo.batches[0] == [
        f"{placeholder_for(0)} uses {placeholder_for(1)} to train the {placeholder_for(2)}."
    ]
    assert result.translation == "反向传播 uses 梯度下降 to train the 神经网络."
    assert [term.en for term in result.terms] == [
        "back-propagation",
        "gradient descent",
        "neural network",
    ]
    assert result.terms[0].matched == "Backpropagation"
    assert SENTINEL_PREFIX not in result.translation


def test_plain_english_is_unchanged_when_no_term_matches():
    result = translate_texts(["Hello from the classroom."], GLOSSARY, Echo())[0]
    assert result.translation == "Hello from the classroom."
    assert result.terms == ()


def test_weights_biases_and_hyphenated_phrase():
    result = translate_texts(
        ["The weights and biases use self-attention."],
        GLOSSARY,
        Echo(),
    )[0]
    assert result.translation == "The 权重 and 偏置 use 自注意力."
    assert [term.en for term in result.terms] == ["weight", "bias", "self-attention"]


def test_dropped_sentinels_still_insert_chinese():
    drop = Drop()
    result = translate_texts(["Backpropagation updates weights."], GLOSSARY, drop)[0]
    assert "反向传播" in result.translation
    assert "权重" in result.translation
    assert SENTINEL_PREFIX not in result.translation
    assert "backprop" not in drop.batches[0][0].casefold()
    assert len(drop.batches) >= 2
    for piece in drop.batches[1]:
        assert "backpropagation" not in piece.casefold()
        assert "weight" not in piece.casefold()


def test_long_text_is_split_without_losing_terms():
    text = "Backpropagation is useful. " * 30
    echo = Echo()
    result = translate_texts([text], GLOSSARY, echo, max_unit_chars=40)[0]
    assert len(echo.batches) == 1
    assert len(echo.batches[0]) > 1
    joined = "".join(echo.batches[0])
    assert SENTINEL_PREFIX in joined
    assert "backpropagation" not in joined.casefold()
    assert result.translation == text.replace("Backpropagation", "反向传播")


def test_split_is_lossless_and_keeps_sentinels():
    body = "word " * 80
    sentinel = placeholder_for(0)
    text = body + sentinel + body
    parts = split_for_translation(text, 48)
    assert "".join(parts) == text
    assert sum(part.count(sentinel) for part in parts) == 1
    for part in parts:
        if SENTINEL_PREFIX in part:
            assert sentinel in part
    assert split_for_translation("short text", 48) == ["short text"]
