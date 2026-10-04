import hashlib
import re

from vjaudit.hashing import canonical_json, config_hash, git_sha, sha256_hex, short_id


def test_canonical_json_is_key_order_invariant():
    assert canonical_json({"b": 1, "a": [2, {"d": 3, "c": 4}]}) == canonical_json(
        {"a": [2, {"c": 4, "d": 3}], "b": 1}
    )


def test_canonical_json_keeps_unicode():
    assert canonical_json({"q": "Ｄｏｇ é"}) == '{"q":"Ｄｏｇ é"}'


def test_short_id_matches_contract_formula():
    assert short_id("msvd:test:1", "qwen3vl8b", "abc") == hashlib.sha1(b"msvd:test:1|qwen3vl8b|abc").hexdigest()[:16]


def test_config_hash_changes_with_any_field():
    assert config_hash({"n_frames": 16}) != config_hash({"n_frames": 32})


def test_config_hash_is_sha256_of_canonical_json():
    obj = {"b": [1, 2], "a": "x"}
    assert config_hash(obj) == sha256_hex(canonical_json(obj))
    assert re.fullmatch(r"[0-9a-f]{64}", config_hash(obj))


def test_git_sha_outside_repo_is_nogit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert git_sha() == "nogit"
