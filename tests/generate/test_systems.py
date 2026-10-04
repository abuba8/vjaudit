import pytest

from vjaudit.generate.systems import expand_mc_systems, expand_systems, parse_mc_letter


def test_expand_systems_yields_11_ids(gens):
    specs = expand_systems(gens, "msvd")
    ids = [s.system_id for s in specs]
    assert len(ids) == 11 == len(set(ids))
    assert {"qwen3vl8b", "qwen3vl8b-sentence", "qwen3vl8b-blind", "qwen25vl7b-blind", "internvl35_8b-blind",
            "qwen3vl8b-f1", "qwen3vl8b-f4", "qwen3vl2b", "llavanextvideo7b"} <= set(ids)
    by = {s.system_id: s for s in specs}
    assert by["qwen3vl8b-sentence"].style == "sentence" and by["qwen3vl8b-blind"].blind
    assert by["qwen3vl8b-blind"].n_frames == 0 and by["qwen3vl8b-f4"].n_frames == 4
    assert by["qwen3vl8b"].n_frames == 16
    assert {s.system_id: s.n_frames for s in expand_systems(gens, "anetqa")}["qwen3vl8b"] == 32


def test_expand_mc_systems_yields_5(gens):
    specs = expand_mc_systems(gens)
    assert len(specs) == 5 and all(s.style == "mc" and s.system_id.endswith("-mc") for s in specs)


@pytest.mark.parametrize("text,idx", [("B", 1), ("(C) the man", 2), ("The answer is D.", 3), ("none", None),
                                      ("A.", 0), ("E) eats", 4), ("Because he fell", None), ("", None)])
def test_parse_mc_letter(text, idx):
    assert parse_mc_letter(text) == idx
