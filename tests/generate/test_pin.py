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


def test_pin_judges_only_entries_with_revision(tmp_path):
    from vjaudit.config import JudgesFile
    p = tmp_path / "judges.yaml"
    shutil.copy("configs/profiles/mac/judges.yaml", p)
    before = p.read_text()
    got = pin_revisions(p, resolver=lambda repo: "sha-" + repo.split("/")[1].lower())
    judges = {j.id: j for j in load_yaml(p, JudgesFile).judges}
    assert got == {"qwen3-4b": "sha-qwen3-4b-instruct-2507", "qwen25-1.5b": "sha-qwen2.5-1.5b-instruct"}
    assert judges["qwen3-4b"].revision == "sha-qwen3-4b-instruct-2507"
    gpt_block = before[before.index("- id: gpt-3.5-turbo-0125"):]
    assert gpt_block in p.read_text()  # the API judge is untouched
