"""Judge templates: T-VCG (Video-ChatGPT protocol, verbatim) and T-RUB (BlackboxNLP rubric, zero-shot)."""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from vjaudit.paths import CONFIGS

JUDGE_DIR = CONFIGS / "prompts" / "judge"
VCG_CONTENT_SHA256 = "6a709644a1df2e2736bad758dd2b98888ea2441ab18916fc3be85d8ef0420125"
FINAL_YES_NO = "Answer with a single word: Yes or No."
_SLOT = re.compile(r"\{(question|answer|pred)\}")


@dataclass(frozen=True)
class Template:
    id: str
    parts: tuple[str, ...]
    content_hash: str


def _read(name: str) -> str:
    return (JUDGE_DIR / name).read_text(encoding="utf-8")


def load_template(template_id: str) -> Template:
    if template_id == "t-vcg":
        parts = (_read("t_vcg_system.txt"), _read("t_vcg_user.txt"))
        digest = hashlib.sha256(parts[0].encode() + b"\x00" + parts[1].encode()).hexdigest()
    elif template_id == "t-rub":
        parts = (_read("t_rub.txt"),)
        digest = hashlib.sha256(parts[0].encode()).hexdigest()
    else:
        raise KeyError(template_id)
    return Template(template_id, parts, digest)


def template_source(template_id: str) -> dict:
    return yaml.safe_load((JUDGE_DIR / "source.yaml").read_text(encoding="utf-8"))[template_id]


def fill(text: str, question: str, answer: str, pred: str) -> str:
    """Single-pass slot filling: values are never re-scanned, so braces in them stay literal."""
    values = {"question": question, "answer": answer, "pred": pred}
    return _SLOT.sub(lambda m: values[m.group(1)], text)


def render_vcg(question: str, answer: str, pred: str) -> list[dict]:
    system, user = load_template("t-vcg").parts
    return [{"role": "system", "content": system}, {"role": "user", "content": fill(user, question, answer, pred)}]


def render_rub(question: str, answer: str, pred: str) -> list[dict]:
    (text,) = load_template("t-rub").parts
    return [{"role": "user", "content": fill(text, question, answer, pred)}]


def protocol_id(template: Template, decoding_tag: str) -> str:
    return f"{template.id}@{template.content_hash[:8]}/{decoding_tag}"


# -- provenance -------------------------------------------------------------------
def _joined(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = ""
        for v in node.values:
            if isinstance(v, ast.Constant):
                out += v.value
            elif isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name):
                out += "{" + v.value.id + "}"
            else:
                raise ValueError("unsupported f-string part")
        return out
    raise ValueError(f"unsupported content node {type(node).__name__}")


def extract_vcg_messages(upstream_py: Path) -> dict[str, str]:
    """{"system": ..., "user": ...} from the upstream evaluation script's messages list."""
    found: dict[str, str] = {}
    for node in ast.walk(ast.parse(Path(upstream_py).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Dict):
            d = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
            if "role" in d and "content" in d and isinstance(d["role"], ast.Constant):
                found.setdefault(d["role"].value, _joined(d["content"]))
    return found


def verify_vcg_source(upstream_files: list[Path]) -> list[str]:
    """Differences between our T-VCG files and each upstream script; [] means identical."""
    system, user = load_template("t-vcg").parts
    diffs = []
    for f in upstream_files:
        msgs = extract_vcg_messages(f)
        if msgs.get("system") != system:
            diffs.append(f"{Path(f).name}: system message differs")
        if msgs.get("user") != user:
            diffs.append(f"{Path(f).name}: user message differs")
    return diffs


def vendor_rub(prior_template: Path, var_map: dict[str, str], out: Path = JUDGE_DIR / "t_rub.txt") -> str:
    """Render the prior repo's Jinja judge template in zero-shot form with our slots, append the
    final Yes/No instruction if missing, and write it to ``out``."""
    from jinja2 import Environment

    env = Environment(keep_trailing_newline=True)
    text = env.from_string(Path(prior_template).read_text(encoding="utf-8")).render(
        **{k: "{" + v + "}" for k, v in var_map.items()})
    text = text.rstrip()
    if not text.endswith(FINAL_YES_NO):
        text += "\n" + FINAL_YES_NO
    Path(out).write_text(text, encoding="utf-8")
    return text
