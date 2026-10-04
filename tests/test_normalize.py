import pytest

from vjaudit.normalize import normalize_answer

CASES = [
    ("A man.", "man"), ("the man", "man"), ("Man", "man"), ("The Man!", "man"),
    ("an apple", "apple"), ("The The", "the"),
    ("playing guitar", "playing guitar"), ("playing a guitar", "playing a guitar"),
    ("man's hat", "man s hat"), ("  Two\n dogs ", "two dogs"), ("Ｄｏｇ", "dog"),
    ("", ""), ("...", "..."), ("a", "a"),
    ("Sorry, I can't tell from the video.", "sorry i can t tell from the video"),
    ("yes", "yes"), ("No.", "no"), ("3", "3"), ("three", "three"),
    ("cat or dog", "cat or dog"), ("a cat", "cat"), ("a cat or a dog", "cat or a dog"),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_normalize_cases(raw, expected):
    assert normalize_answer(raw) == expected


def test_case_table_has_22_rows():
    assert len(CASES) == 22


def test_never_maps_nonempty_to_empty():
    for raw in ["!", "?!", "the", "  .  "]:
        assert normalize_answer(raw) != ""
