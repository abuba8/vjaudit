"""Sections 4–6 of the notebook must behave exactly like the tested package."""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.factories import make_item
from tests.test_e2e import _video
from vjaudit.data.availability import check_availability
from vjaudit.data.loaders import load_benchmark
from vjaudit.data.sampling import assign_nested, split_dev_test

RAW = Path("tests/fixtures/raw").resolve()
DIRS = {"msvd": "msvd", "anetqa": "anetqa", "nextoe": "nextqa"}


def data_tree(tmp_path: Path) -> Path:
    """tmp/data/<msvd|anetqa|nextqa>/annotations = the fixture question files."""
    root = tmp_path / "data"
    for d in DIRS.values():
        shutil.copytree(RAW / d, root / d / "annotations")
    return root


def _cmp(item):
    return (item["item_id"], item["question"], list(item["answers_ref"]), item["qtype"], item["video_id"],
            None if item["mc_options"] is None else list(item["mc_options"]), item["mc_answer_idx"])


@pytest.mark.parametrize("bench", ["msvd", "anetqa", "nextoe"])
def test_loaders_match_package(nb_lib, bench):
    loader = nb_lib["LOADERS"][bench]
    ours = [_cmp(r) for r in loader(RAW / DIRS[bench])]
    theirs = [_cmp(i.model_dump()) for i in load_benchmark(bench, RAW / DIRS[bench])]
    assert ours == theirs and len(ours) > 0


def test_load_benchmarks_builds_one_table(nb_lib, tmp_path):
    df = nb_lib["load_benchmarks"](data_tree(tmp_path))
    assert list(df.columns[:4]) == ["item_id", "benchmark", "split", "video_id"]
    assert set(df["benchmark"]) == {"msvd", "anetqa", "nextoe"} and df["item_id"].is_unique


def test_missing_data_root_message(nb_lib, tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        nb_lib["load_benchmarks"](tmp_path / "nope")


def test_check_videos_matches_package(nb_lib, tmp_path):
    root = data_tree(tmp_path)
    df = nb_lib["load_benchmarks"](root)
    msvd = df[df.benchmark == "msvd"]
    vids = root / "msvd" / "videos"
    first = msvd.iloc[0]["video_id"]
    _video(vids / f"{first}.avi")
    (vids / "broken.avi").write_bytes(b"not a video")
    ours = nb_lib["check_videos"](msvd, root)
    items = load_benchmark("msvd", root / "msvd" / "annotations")
    theirs = check_availability(items, vids, report_path=tmp_path / "r.md", exts=[".mp4", ".avi", ".mkv", ".webm"])
    assert list(ours["available"]) == [i.available for i in theirs]
    assert list(ours["video_path"]) == [i.video_path for i in theirs]
    assert set(ours.loc[~ours.available, "problem"]) == {"missing"}


def _pool():
    rows = []
    for n in range(60):
        rows.append(make_item(n, item_id=f"anetqa:test:{n:03d}", benchmark="anetqa",
                              qtype=["color", "motion", "number"][n % 3], available=n % 7 != 0))
    for n in range(80):
        rows.append(make_item(n, item_id=f"msvd:test:{n:03d}", benchmark="msvd",
                              qtype=["what", "who", "how", "when", "where"][n % 5] if n < 70 else "other"))
    return rows


def test_sampling_matches_package(nb_lib):
    items = _pool()
    theirs = split_dev_test(assign_nested(items, (40, 20, 10), 42, 15), 0.4, 42)
    pool = pd.DataFrame([i.model_dump() for i in items])
    ours = nb_lib["split_dev_test"](nb_lib["assign_nested"](pool, (40, 20, 10), 42, 15), 0.4, 42)
    for col in ("in_main", "in_annotated", "in_vid_subset", "human_split", "vid_rank"):
        a = {i.item_id: getattr(i, col) for i in theirs}
        b = {r["item_id"]: (None if pd.isna(r[col]) else r[col]) for r in ours.to_dict("records")}
        assert a == b, col


# --- final review: crash safety and a fast, resumable video check ------------------------------------------
def test_store_repairs_a_cut_off_last_line(nb_lib, tmp_path):
    st = nb_lib["Store"](tmp_path)
    st.append("t", [{"k": "a"}, {"k": "b"}], key="k")
    with open(tmp_path / "t.jsonl", "a") as f:
        f.write('{"_key": "c", "k": "c", "x"')          # the kernel died while writing row c
    st2 = nb_lib["Store"](tmp_path)
    assert list(st2.load("t").k) == ["a", "b"]
    st2.append("t", [{"k": "c"}, {"k": "d"}], key="k")
    assert list(nb_lib["Store"](tmp_path).load("t").k) == ["a", "b", "c", "d"]


def test_store_names_file_and_line_of_a_broken_row(nb_lib, tmp_path):
    (tmp_path / "t.jsonl").write_text('{"_key": "a", "k": "a"}\nnot json\n{"_key": "b", "k": "b"}\n')
    with pytest.raises(ValueError, match=r"t\.jsonl.*line 2"):
        nb_lib["Store"](tmp_path).load("t")


def _with_empty_videos(root: Path, questions: pd.DataFrame) -> None:
    for bench, folder in DIRS.items():
        for v in questions.loc[questions.benchmark == bench, "video_id"]:
            p = root / folder / "videos" / f"{v}.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()


def _settings(nb_lib, root, tmp_path, dry):
    return nb_lib["resolve_settings"](dict(PROFILE="mac", DRY_RUN=dry, DATA_ROOT=str(root), RUN="annotated"),
                                      tmp_path / "repo")


def test_video_check_saves_in_chunks_and_resumes(nb_lib, tmp_path):
    root = data_tree(tmp_path)
    questions = nb_lib["load_benchmarks"](root)
    _with_empty_videos(root, questions)
    S = _settings(nb_lib, root, tmp_path, dry=False)
    probed = []

    def dies_after_5(path):
        if len(probed) == 5:
            raise KeyboardInterrupt
        probed.append(path)
        return True

    with pytest.raises(KeyboardInterrupt):
        nb_lib["update_pool"](nb_lib["Store"](S["runs"]), questions, S, probe=dies_after_5, chunk=2)
    saved = len(nb_lib["Store"](S["runs"]).load("pool"))
    assert 4 <= saved < len(questions)
    n = nb_lib["update_pool"](nb_lib["Store"](S["runs"]), questions, S, probe=lambda p: True, chunk=2)
    pool = nb_lib["Store"](S["runs"]).load("pool")
    assert n == len(questions) - saved and len(pool) == len(questions) and pool.item_id.is_unique
    assert pool.available.all()
    assert nb_lib["update_pool"](nb_lib["Store"](S["runs"]), questions, S, probe=lambda p: True) == 0


def test_dry_run_checks_only_a_few_questions_per_benchmark(nb_lib, tmp_path):
    root = data_tree(tmp_path)
    questions = nb_lib["load_benchmarks"](root)
    _with_empty_videos(root, questions)
    assert _settings(nb_lib, root, tmp_path, dry=False)["check_limit"] is None
    S = _settings(nb_lib, root, tmp_path, dry=True)
    assert S["check_limit"] == 20
    S["check_limit"] = 2
    store = nb_lib["Store"](S["runs"])
    assert nb_lib["update_pool"](store, questions, S, probe=lambda p: True) == 6
    assert store.load("pool").groupby("benchmark").size().to_dict() == {"anetqa": 2, "msvd": 2, "nextoe": 2}
    assert nb_lib["update_pool"](store, questions, S, probe=lambda p: True) == 0
