import shutil

from vjaudit.config import GeneratorsFile, load_yaml
from vjaudit.generate.pin import pin_revisions


def test_pin_revisions_rewrites_only_revision_lines(tmp_path):
    p = tmp_path / "generators.yaml"
    shutil.copy("configs/profiles/server/generators.yaml", p)
    before = p.read_text()
    got = pin_revisions(p, resolver=lambda repo: "sha-" + repo.split("/")[1].lower())
    after = p.read_text()
    gens = load_yaml(p, GeneratorsFile).generators
    assert got["qwen3vl8b"] == "sha-qwen3-vl-8b-instruct"
    assert all(g.revision.startswith("sha-") for g in gens)
    assert before.count("\n") == after.count("\n") and "# Generators" in after  # comments kept
