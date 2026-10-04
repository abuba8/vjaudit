import pytest

from vjaudit.judge.parse import parse_vcg, parse_yes_no


@pytest.mark.parametrize("text,expected", [
    ("{'pred': 'yes', 'score': 4}", (True, 4.0, "ok")),
    ('{"pred": "no", "score": 1}', (False, 1.0, "ok")),
    ("```python\n{'pred': 'yes', 'score': 4.8}\n```", (True, 4.8, "ok")),
    ("pred: yes score: 3", (True, 3.0, "ok")),
    ("I think so", (None, None, "unparseable")),
    ("{'pred': 'No', 'score': '2'}", (False, 2.0, "ok")),
    ("{'pred': 'yes'}", (True, None, "ok")),
    ("{'pred': \"yes\", 'score': 5, 'note': 'it\\'s {fine}'}", (True, 5.0, "ok")),
])
def test_parse_vcg_cases(text, expected):
    assert parse_vcg(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Yes", (True, "ok")), (" yes.", (True, "ok")), ("NO", (False, "ok")), ("Maybe", (None, "off_vocab")),
    ("", (None, "off_vocab")),
])
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) == expected
