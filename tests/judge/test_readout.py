import json
import math
from pathlib import Path

import pytest

from vjaudit.judge.readout import (
    margin_from_top, p_yes, readout_first_token, readout_vcg, token_class, vcg_verdict_index,
)

FX = Path("tests/fixtures/openai")


def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def load(name):
    return json.loads((FX / name).read_text())["content"]


@pytest.mark.parametrize("tok,cls", [(" Yes", "yes"), ("'yes", "yes"), ('"no"', "no"), ("No:", "no"), ("y", None), ("yess", None)])
def test_token_class(tok, cls):
    assert token_class(tok) == cls


def test_margin_aggregates_variants():
    m = margin_from_top([(" Yes", -0.1), ("yes", -2.3), ("No", -2.5)])
    assert m == pytest.approx(lse([-0.1, -2.3]) - (-2.5), abs=1e-9)


def test_margin_none_when_class_absent():
    assert margin_from_top([("Yes", -0.1), ("maybe", -2.0)]) is None


def test_p_yes_is_sigmoid():
    assert p_yes(0.0) == 0.5 and p_yes(2.0) == pytest.approx(1 / (1 + math.exp(-2)))


def test_vcg_index_on_standard_dict():
    assert vcg_verdict_index(["{'", "pred", "':", " '", "yes", "',", " '", "score"]) == 4


def test_vcg_index_double_quotes():
    assert vcg_verdict_index(['{"', "pred", '":', ' "', "no", '",']) == 4


def test_vcg_index_split_token_returns_none():
    assert vcg_verdict_index(["{'", "pred", "':", " '", "y", "es", "',"]) is None


def test_vcg_index_merged_quote_token():
    assert vcg_verdict_index(["{'", "pred", "':", " 'yes", "',"]) == 3


def test_readout_vcg_standard():
    margin, top = readout_vcg(load("vcg_standard.json"))
    assert margin == pytest.approx(lse([-0.1, -4.0]) - (-2.5))
    assert top[0] == ["yes", -0.1] and len(top) <= 5


def test_readout_vcg_split_gives_none():
    assert readout_vcg(load("vcg_split.json")) == (None, [])


def test_first_token_readout_for_rub():
    margin, top = readout_first_token(load("rub_first_token.json"))
    assert margin == pytest.approx(lse([-0.3, -3.2]) - (-1.4))
    assert [t for t, _ in top] == ["Yes", "No", " yes"]
