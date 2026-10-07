"""Shared CLI options: every command takes --profile and prints the active profile first."""
from __future__ import annotations

from pathlib import Path

import typer

from .profile import Profile, load_profile

PROFILE_HELP = "profile name (configs/profiles/<name>) or folder; default: $VJAUDIT_PROFILE or 'server'"


def open_profile(profile: str | None, paths: Path) -> Profile:
    """Load the profile and print 'profile: <name>' so a run without PROFILE=mac is obvious."""
    p = load_profile(profile, paths)
    typer.echo(f"profile: {p.name} (artifacts: {p.artifacts})")
    base = p.artifacts.parent
    old = [k for k in ("item_pool", "item", "generation", "judgecall") if (base / k).is_dir()]
    if old:
        typer.echo(f"WARNING: {base} holds records from before profiles existed ({', '.join(old)}); this command "
                   f"does not see them (it uses {p.artifacts}). If they belong to the server profile, move them: "
                   "see docs/GUIDE.md, 'Profiles: server and mac'.")
    return p


def profile_doc(p: Profile, stem: str) -> Path:
    """Per-profile report path, e.g. docs/samples_mac.md."""
    return Path("docs") / f"{stem}_{p.name}.md"
