import pytest
from pydantic import ValidationError

from tests.helpers.factories import make_generation, make_item, make_judgecall


def test_valid_records_construct():
    assert make_item().item_id == "msvd:test:0"
    assert make_generation().style == "short"
    assert make_judgecall().parse_status == "ok"


def test_schema_rejects_unknown_parse_status():
    with pytest.raises(ValidationError):
        make_judgecall(parse_status="maybe")


def test_generation_style_allows_mc_but_not_other():
    assert make_generation(style="mc").style == "mc"
    with pytest.raises(ValidationError):
        make_generation(style="long")


def test_vcg_score_accepts_float():
    assert make_judgecall(vcg_score=4.8).vcg_score == 4.8


def test_item_human_split_values():
    assert make_item(human_split="dev").human_split == "dev"
    with pytest.raises(ValidationError):
        make_item(human_split="train")


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        make_item(colour="red")
