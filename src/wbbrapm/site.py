"""Stage 6: render the static explorer site from processed outputs."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from . import io
from .config import (
    LOW_MINUTES_FLAG,
    MIN_MINUTES_DEFAULT_FILTER,
    PROCESSED_DIR,
    SITE_DIR,
    STANFORD_TEAM_NAME,
    TEMPLATE_DIR,
    SeasonPaths,
)

LINEUP_MIN_POSS = 50
PLAYER_JS_COLS = [
    "athlete_id", "name", "team", "team_logo", "position", "headshot",
    "games", "minutes", "orapm", "drapm", "rapm", "rapm_margin", "rank", "low_sample",
]


def season_label(season: int) -> str:
    return f"{season - 1}–{str(season)[2:]}"


def _available_seasons() -> list[int]:
    out = []
    for p in PROCESSED_DIR.glob("ratings_*.parquet"):
        m = re.fullmatch(r"ratings_(\d{4})\.parquet", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _team_summaries(ratings: pd.DataFrame, team_box: pd.DataFrame, teams_ref: pd.DataFrame) -> pd.DataFrame:
    r = ratings.dropna(subset=["team_id"]).copy()
    r["team_id"] = r["team_id"].astype(int)
    team_rapm = (
        r[r["minutes"] > 0]
        .groupby("team_id")
        .apply(lambda g: np.average(g["rapm"], weights=g["minutes"]), include_groups=False)
        .rename("team_rapm")
        .reset_index()
    )
    tb = team_box.copy()
    tb["margin"] = tb["team_score"] - tb["opponent_team_score"]
    perf = tb.groupby("team_id").agg(
        wins=("team_winner", "sum"),
        games=("game_id", "nunique"),
        net_margin=("margin", "mean"),
    ).reset_index()
    perf["record"] = perf["wins"].astype(int).astype(str) + "–" + (perf["games"] - perf["wins"]).astype(int).astype(str)

    ref = teams_ref[["team_id", "display_name", "logo"]].rename(columns={"display_name": "name"})
    out = team_rapm.merge(perf, on="team_id").merge(ref, on="team_id", how="left")
    out["name"] = out["name"].fillna("Unknown")
    out = out.sort_values("team_rapm", ascending=False).reset_index(drop=True)
    out["rapm_rank"] = out.index + 1
    return out


def _top_lineups(stints: pd.DataFrame, team_id: int, names: dict[int, str], top_n: int = 5) -> list[dict]:
    frames = []
    for side, opp in (("home", "away"), ("away", "home")):
        sub = stints[stints[f"{side}_team_id"] == team_id]
        if sub.empty:
            continue
        frames.append(pd.DataFrame({
            "lineup": sub[f"{side}_lineup"].map(lambda l: tuple(sorted(int(a) for a in l))),
            "seconds": sub["seconds"],
            "pf": sub[f"{side}_pts"], "pa": sub[f"{opp}_pts"],
            "own_poss": sub[f"{side}_poss"], "opp_poss": sub[f"{opp}_poss"],
        }))
    if not frames:
        return []
    g = pd.concat(frames).groupby("lineup").sum()
    g = g[(g["own_poss"] >= LINEUP_MIN_POSS) & (g["opp_poss"] >= LINEUP_MIN_POSS)]
    if g.empty:
        return []
    g["ortg"] = 100 * g["pf"] / g["own_poss"]
    g["drtg"] = 100 * g["pa"] / g["opp_poss"]
    g["net"] = g["ortg"] - g["drtg"]
    g["poss"] = (g["own_poss"] + g["opp_poss"]) / 2
    g = g.sort_values("net", ascending=False).head(top_n)
    return [
        {
            "names": sorted(names.get(a, str(a)).split()[-1] for a in lineup),
            "poss": row.poss, "minutes": row.seconds / 60,
            "ortg": row.ortg, "drtg": row.drtg, "net": row.net,
        }
        for lineup, row in g.iterrows()
    ]


def build_site(season: int) -> None:
    paths = SeasonPaths(season)
    ratings = pd.read_parquet(paths.ratings)
    stints = pd.read_parquet(paths.stints)
    team_box = io.load_team_box(paths)
    teams_ref = io.load_teams(paths)
    validation = json.loads(paths.validation_report.read_text())

    d1_ids = set(teams_ref["team_id"].dropna().astype(int))
    team_box = team_box[team_box["team_id"].isin(d1_ids)]

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)
    (SITE_DIR / "data").mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "teams").mkdir(parents=True, exist_ok=True)

    # --- data files (JS for the table, CSV for download) ---
    js_df = ratings[PLAYER_JS_COLS].copy()
    for c in ("orapm", "drapm", "rapm", "rapm_margin", "minutes"):
        js_df[c] = js_df[c].astype(float).round(2)
    js_df = js_df.where(js_df.notna(), None)
    payload = {"season": season, "players": js_df.to_dict(orient="records")}
    (SITE_DIR / "data" / f"ratings_{season}.js").write_text(
        "window.WBB_DATA = " + json.dumps(payload) + ";"
    )
    ratings.drop(columns=["team_color"], errors="ignore").to_csv(
        SITE_DIR / "data" / f"ratings_{season}.csv", index=False
    )

    seasons = [
        {"season": s, "label": season_label(s),
         "href": "index.html" if s == max(_available_seasons()) else f"index-{s}.html"}
        for s in _available_seasons()
    ]
    common = {
        "season": season,
        "season_label": season_label(season),
        "seasons": seasons,
        "min_minutes_default": MIN_MINUTES_DEFAULT_FILTER,
        "low_minutes_flag": LOW_MINUTES_FLAG,
        "lineup_min_poss": LINEUP_MIN_POSS,
    }

    # --- leaderboard ---
    team_options = ratings.dropna(subset=["team"]).drop_duplicates("team")[["team"]]
    team_options = team_options.sort_values("team").to_dict(orient="records")
    index_html = env.get_template("index.html.j2").render(
        page="index", rel="", teams=team_options, **common
    )
    is_latest = season == max(_available_seasons())
    (SITE_DIR / f"index-{season}.html").write_text(index_html)
    if is_latest:
        (SITE_DIR / "index.html").write_text(index_html)

    # --- team pages ---
    summaries = _team_summaries(ratings, team_box, teams_ref)
    n_teams = len(summaries)
    names = dict(zip(ratings["athlete_id"], ratings["name"]))
    roster_cols = ["rank", "name", "position", "headshot", "games", "minutes",
                   "orapm", "drapm", "rapm", "low_sample"]

    teamindex_html = env.get_template("teamindex.html.j2").render(
        page="teams", rel="", teams=summaries.to_dict(orient="records"), **common
    )
    (SITE_DIR / "teamindex.html").write_text(teamindex_html)

    team_tpl = env.get_template("team.html.j2")
    stanford_html = None
    for row in summaries.itertuples(index=False):
        tid = int(row.team_id)
        roster = ratings[ratings["team_id"] == tid].sort_values("rapm", ascending=False)
        team_ctx = {
            "team_id": tid, "name": row.name, "logo": row.logo, "record": row.record,
            "net_margin": row.net_margin, "team_rapm": row.team_rapm,
            "rapm_rank": int(row.rapm_rank), "n_teams": n_teams,
            "players": roster[roster_cols].to_dict(orient="records"),
            "lineups": _top_lineups(stints, tid, names),
        }
        html = team_tpl.render(page="team", rel="../", team=team_ctx, **common)
        (SITE_DIR / "teams" / f"{tid}.html").write_text(html)
        if STANFORD_TEAM_NAME.lower() in str(row.name).lower():
            stanford_html = team_tpl.render(page="stanford", rel="", team=team_ctx, **common)

    if stanford_html is not None:
        (SITE_DIR / "stanford.html").write_text(stanford_html)

    # --- about / validation ---
    class DotDict(dict):
        __getattr__ = dict.get

    v = json.loads(json.dumps(validation), object_hook=DotDict)
    about_html = env.get_template("about.html.j2").render(page="about", rel="", v=v, **common)
    (SITE_DIR / "about.html").write_text(about_html)
