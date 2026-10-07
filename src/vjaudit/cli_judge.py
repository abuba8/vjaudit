"""`vjaudit.cli judge ...` commands (templates provenance; GPT-3.5 capture is added in Task 13)."""
from __future__ import annotations

from pathlib import Path

import typer
import yaml

from .cli_common import PROFILE_HELP, open_profile, profile_doc
from .judge.templates import JUDGE_DIR, vendor_rub, verify_vcg_source
from .paths import DEFAULT_PATHS_FILE

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Judging: template provenance and capture.")
SOURCE = JUDGE_DIR / "source.yaml"


def _update_source(path: Path, template_id: str, **fields) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data[template_id].update(fields)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@app.command("verify-vcg")
def verify_vcg_cmd(upstream: list[Path] = typer.Option(..., help="upstream Video-ChatGPT eval script(s); first = ActivityNet"),
                   commit: str = typer.Option(..., help="upstream commit SHA the files came from"),
                   source: Path = typer.Option(SOURCE)) -> None:
    """Check t_vcg_*.txt against the upstream script(s); mark verified when the first file matches."""
    diffs = verify_vcg_source(upstream)
    for d in diffs:
        typer.echo(d)
    main_ok = not any(d.startswith(Path(upstream[0]).name + ":") for d in diffs)
    if main_ok:
        _update_source(source, "t-vcg", upstream_commit=commit, verified=True)
        typer.echo(f"t-vcg verified against {upstream[0]} @ {commit}")
    raise typer.Exit(0 if main_ok else 1)


@app.command("vendor-rub")
def vendor_rub_cmd(prior: Path = typer.Option(..., help="prior repo src/Prompts/LLM_as_judge_base.j2"),
                   var_map: str = typer.Option(..., help="prior Jinja var → slot, e.g. q=question,gt=answer,gen=pred"),
                   commit: str = typer.Option(...), out: Path = typer.Option(JUDGE_DIR / "t_rub.txt"),
                   source: Path = typer.Option(SOURCE)) -> None:
    """Replace t_rub.txt with the prior repo's template (zero-shot form) and mark it verified."""
    mapping = dict(kv.split("=", 1) for kv in var_map.split(","))
    if set(mapping.values()) != {"question", "answer", "pred"}:
        raise typer.BadParameter("var-map must map onto exactly question, answer, pred")
    text = vendor_rub(prior, mapping, out)
    _update_source(source, "t-rub", upstream_commit=commit, verified=True)
    typer.echo(text)


def _groups(gens) -> dict[str, set[str]]:
    from .generate.systems import expand_systems

    ablation = {f"{g.id}-f{k}" for g in gens for k in g.frame_ablation}
    judged = {s.system_id for b in ("msvd", "anetqa", "nextoe") for s in expand_systems(gens, b)}
    return {"core": judged - ablation, "ablation": ablation}


@app.command("smoke")
def smoke_cmd(allow_unpinned: bool = typer.Option(False, help="allow UNPINNED judge revisions (quick tests only)"),
              paths: Path = typer.Option(DEFAULT_PATHS_FILE),
              profile: str = typer.Option(None, help=PROFILE_HELP)) -> None:
    """Load each local judge and grade one fixed example with T-VCG and T-RUB; print verdict, margin, top tokens."""
    from .judge import smoke
    from .judge.factory import make_client

    prof = open_profile(profile, paths)
    rows = smoke.smoke_judges(prof, client_factory=lambda cfg, d: make_client(cfg, d, allow_unpinned=allow_unpinned))
    if not rows:
        typer.echo("no local judges (backend: transformers) in this profile")
        return
    typer.echo("| judge | template | verdict | margin | seconds | top tokens |\n|---|---|---|---|---|---|")
    for r in rows:
        margin = "—" if r["margin"] is None else f"{r['margin']:.3f}"
        top = ", ".join(f"{t!r}:{lp:.2f}" for t, lp in r["top_tokens"][:3])
        typer.echo(f"| {r['judge']} | {r['template']} | {r['verdict']} | {margin} | {r['seconds']} | {top} |")
    if any(str(r["verdict"]).startswith("ERROR:") for r in rows):
        raise typer.Exit(1)


@app.command("capture")
def capture_cmd(stages: str = typer.Option("", help="comma list, e.g. M1,M2 (default: every non-optional stage, "
                                                    "in order)"),
                confirm: bool = typer.Option(False, help="actually run the judges (otherwise: dry run with estimates)"),
                allow_unverified: bool = typer.Option(False, help="run even if a template's provenance is unverified"),
                allow_unpinned: bool = typer.Option(False, help="allow UNPINNED local judge revisions (tests only)"),
                paths: Path = typer.Option(DEFAULT_PATHS_FILE),
                profile: str = typer.Option(None, help=PROFILE_HELP),
                report: Path = typer.Option(None, help="default: docs/capture_<profile>.md")) -> None:
    """Capture judge verdicts + logprobs in the profile's stage order (dry run unless --confirm)."""
    import asyncio

    from .judge.capture import build_jobs, load_stages, run_capture, select_stages, write_capture_report
    from .judge.factory import make_client
    from .judge.templates import render_rub, render_vcg, template_source
    from .schemas import Item

    prof = open_profile(profile, paths)
    report = report or profile_doc(prof, "capture")
    try:
        chosen = select_stages(load_stages(prof.grid), [s for s in stages.split(",") if s])
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Exit(2) from e
    for t in sorted({s.template for s in chosen}):
        if not template_source(t)["verified"] and not allow_unverified:
            typer.echo(f"{t} is not verified (configs/prompts/judge/source.yaml); see docs/GUIDE.md "
                       "'Verify the judge prompts', or pass --allow-unverified")
            raise typer.Exit(2)
    store = prof.store()
    items = [Item(**r) for r in store.iter("item")]
    groups = _groups(prof.generators)
    clients = {jid: make_client(prof.judge(jid), prof.api_cache, allow_unpinned=allow_unpinned)
               for jid in {s.judge_id for s in chosen}}
    jobs = build_jobs(store, chosen, items, groups)
    total = 0.0
    for st in chosen:
        sj = [j for j in jobs if j.stage.name == st.name]
        client = clients[st.judge_id]
        est = 0.0
        if sj and client.cfg.backend != "transformers":  # local judges cost nothing
            render = render_vcg if st.template == "t-vcg" else render_rub
            r = st.request
            reqs = [client.build_request(render(j.item.question, j.item.answers_ref[0], j.gen["output_text"]),
                                         max_tokens=r["max_tokens"], temperature=r.get("temperature"),
                                         logprobs=r["logprobs"], top_logprobs=r["top_logprobs"], n=r.get("n", 1))
                    for j in sj]
            est = client.estimate_cost(reqs)
        total += est
        typer.echo(f"{st.name}: {len(sj)} jobs, estimated ≤ {est:.2f} USD ({st.judge_id}, {st.template}/{st.decoding_tag})")
    typer.echo(f"TOTAL estimated ≤ {total:.2f} USD")
    if not confirm:
        typer.echo("dry run: nothing sent. Re-run with --confirm to run the judges.")
        return
    summary = asyncio.run(run_capture(store, clients, jobs))
    write_capture_report(store, chosen, items, groups, report)
    for e in summary.errors[:50]:
        typer.echo(f"FAILED {e}")
    typer.echo(f"captured {summary.done} calls; {summary.failed} failed (re-run to retry them); spent "
               f"{summary.spent_usd:.2f} USD; {summary.stopped_reason or 'completed'}; report: {report}")
