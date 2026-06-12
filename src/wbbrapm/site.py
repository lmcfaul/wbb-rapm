"""Stage 6: render the static explorer site from processed outputs.

Pages are generated per season (index-<s>.html, teamindex-<s>.html,
stanford-<s>.html, about-<s>.html, teams/<s>/<id>.html) with root aliases
(index.html, ...) pointing at the latest season. A header dropdown switches
the current page to its equivalent in another season, and a cross-season
history file (data/history.js) powers per-player season-by-season ratings
in the leaderboard modal.
"""
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
    MODEL_SPECS,
    PROCESSED_DIR,
    SITE_DIR,
    STANFORD_TEAM_NAME,
    TEMPLATE_DIR,
    SeasonPaths,
)

LINEUP_MIN_POSS = 50
DECOMP_COLS = ["raw_off", "raw_def", "raw_net", "tm_off", "tm_def", "tm_net",
               "opp_off", "opp_def", "opp_net"]
SD_COLS = ["orapm_sd", "drapm_sd", "rapm_sd"]  # ridge-only credible-interval SDs
PLAYER_JS_COLS = [
    "athlete_id", "name", "team", "team_logo", "position", "headshot",
    "games", "minutes", "orapm", "drapm", "rapm", "rapm_margin", "rank", "low_sample",
] + DECOMP_COLS + SD_COLS
HISTORY_COLS = ["season", "team", "games", "minutes", "orapm", "drapm", "rapm", "rank"]


def season_label(season: int) -> str:
    return f"{season - 1}–{str(season)[2:]}"


def available_seasons() -> list[int]:
    out = []
    for p in PROCESSED_DIR.glob("ratings_*.parquet"):
        m = re.fullmatch(r"ratings_(\d{4})\.parquet", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _page_hrefs(season: int) -> dict[str, str]:
    """Root-relative hrefs for one season's pages."""
    return {
        "index": f"index-{season}.html",
        "teams": f"teamindex-{season}.html",
        "lineups": f"lineups-{season}.html",
        "stanford": f"stanford-{season}.html",
        "about": f"about-{season}.html",
    }


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


COMBO_MIN_POSS = {2: 100, 3: 75, 4: 50, 5: 50}  # offensive-possession floor per combo size


def build_lineup_combos(stints: pd.DataFrame, ratings: pd.DataFrame) -> dict:
    """Aggregate every 2- to 5-player same-team combination's on-court numbers.

    Returns {"teams": [...], "combos": {team_id: [[k, names, poss, min, ortg, drtg, net], ...]}}
    filtered by per-size possession floors to keep the data file manageable.
    """
    from itertools import combinations

    last_names = {
        int(a): str(n).split()[-1] for a, n in zip(ratings["athlete_id"], ratings["name"])
    }
    agg: dict[tuple, list] = {}  # (team_id, combo) -> [sec, pf, pa, own_poss, opp_poss]
    for s in stints.itertuples(index=False):
        sides = [
            (s.home_team_id, s.home_lineup, s.home_pts, s.away_pts, s.home_poss, s.away_poss),
            (s.away_team_id, s.away_lineup, s.away_pts, s.home_pts, s.away_poss, s.home_poss),
        ]
        for tid, lineup, pf, pa, own, opp in sides:
            five = sorted(int(a) for a in lineup)
            for k in (2, 3, 4, 5):
                for combo in combinations(five, k):
                    a = agg.setdefault((tid, combo), [0.0, 0.0, 0.0, 0.0, 0.0])
                    a[0] += s.seconds; a[1] += pf; a[2] += pa; a[3] += own; a[4] += opp

    team_names = (
        ratings.dropna(subset=["team_id", "team"])
        .drop_duplicates("team_id")[["team_id", "team"]]
        .astype({"team_id": int})
    )
    name_of_team = dict(zip(team_names["team_id"], team_names["team"]))

    combos: dict[int, list] = {}
    for (tid, combo), (sec, pf, pa, own, opp) in agg.items():
        k = len(combo)
        if own < COMBO_MIN_POSS[k] or opp < COMBO_MIN_POSS[k] or tid not in name_of_team:
            continue
        ortg = 100.0 * pf / own
        drtg = 100.0 * pa / opp
        names = " · ".join(sorted(last_names.get(a, str(a)) for a in combo))
        combos.setdefault(int(tid), []).append([
            k, names, round((own + opp) / 2), round(sec / 60),
            round(ortg, 1), round(drtg, 1), round(ortg - drtg, 1),
        ])
    for tid in combos:
        combos[tid].sort(key=lambda r: -r[6])
    teams = sorted(
        ({"id": t, "name": name_of_team[t]} for t in combos),
        key=lambda d: d["name"],
    )
    return {"teams": teams, "combos": combos}


def build_history() -> None:
    """data/history.js: athlete_id -> per-season rating records, all seasons."""
    frames = []
    for s in available_seasons():
        r = pd.read_parquet(SeasonPaths(s).ratings, columns=HISTORY_COLS + ["athlete_id"])
        frames.append(r)
    if not frames:
        return
    df = pd.concat(frames).sort_values(["athlete_id", "season"])
    for c in ("orapm", "drapm", "rapm", "minutes"):
        df[c] = df[c].astype(float).round(2)
    df = df.where(df.notna(), None)
    hist: dict[str, list] = {}
    for aid, grp in df.groupby("athlete_id"):
        hist[str(int(aid))] = grp[HISTORY_COLS].to_dict(orient="records")
    (SITE_DIR / "data").mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "data" / "history.js").write_text(
        "window.WBB_HISTORY = " + json.dumps(hist) + ";"
    )


def _season_switch(current: int, page: str, rel: str, team_pages: dict[int, set[int]] | None = None,
                   team_id: int | None = None) -> list[dict]:
    """Header dropdown options: this page's equivalent in every built season."""
    out = []
    for s in available_seasons():
        if page == "team" and team_id is not None:
            # fall back to that season's team index if the team wasn't rated
            if team_pages and team_id in team_pages.get(s, set()):
                href = f"{rel}teams/{s}/{team_id}.html"
            else:
                href = f"{rel}{_page_hrefs(s)['teams']}"
        else:
            href = rel + _page_hrefs(s).get(page, _page_hrefs(s)["index"])
        out.append({"label": season_label(s), "href": href, "current": s == current})
    return out


def _rated_team_ids() -> dict[int, set[int]]:
    out = {}
    for s in available_seasons():
        r = pd.read_parquet(SeasonPaths(s).ratings, columns=["team_id"])
        out[s] = set(r["team_id"].dropna().astype(int))
    return out


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
    (SITE_DIR / "teams" / str(season)).mkdir(parents=True, exist_ok=True)

    # --- data files (JS for the table, CSV for download), one set per model ---
    def write_ratings_files(df: pd.DataFrame, model: str) -> None:
        suffix = "" if model == "ridge" else f"_{model}"
        js_df = df[[c for c in PLAYER_JS_COLS if c in df.columns]].copy()
        for c in ["orapm", "drapm", "rapm", "rapm_margin", "minutes"] + DECOMP_COLS + SD_COLS:
            if c in js_df.columns:
                js_df[c] = js_df[c].astype(float).round(2)
        js_df = js_df.where(js_df.notna(), None)
        payload = {"season": season, "model": model, "players": js_df.to_dict(orient="records")}
        (SITE_DIR / "data" / f"ratings_{season}{suffix}.js").write_text(
            "window.WBB_DATA = " + json.dumps(payload) + ";"
        )
        df.drop(columns=["team_color"], errors="ignore").to_csv(
            SITE_DIR / "data" / f"ratings_{season}{suffix}.csv", index=False
        )

    write_ratings_files(ratings, "ridge")
    models_available = ["ridge"]
    for model in MODEL_SPECS:
        p = paths.ratings_for(model)
        if model != "ridge" and p.exists():
            write_ratings_files(pd.read_parquet(p), model)
            models_available.append(model)

    # season-phase trends for the player modal
    trends: dict[str, list] = {}
    phase_labels: list[str] = []
    if paths.phases.exists():
        ph = pd.read_parquet(paths.phases)
        labels = ph.drop_duplicates("phase").sort_values("phase")
        phase_labels = [f"{r.start} – {r.end}" for r in labels.itertuples(index=False)]
        has_sd = "rapm_sd" in ph.columns
        for aid, grp in ph.sort_values("phase").groupby("athlete_id"):
            trends[str(int(aid))] = [
                [int(r.phase), round(r.orapm, 2), round(r.drapm, 2),
                 round(r.rapm, 2), round(r.minutes, 1)]
                + ([round(r.rapm_sd, 2)] if has_sd else [])
                for r in grp.itertuples(index=False)
            ]
    (SITE_DIR / "data" / f"trends_{season}.js").write_text(
        "window.WBB_TRENDS = " + json.dumps({"labels": phase_labels, "players": trends}) + ";"
    )

    # 2- to 5-player lineup combinations for the explorer page
    lineup_data = build_lineup_combos(stints, ratings)
    (SITE_DIR / "data" / f"lineups_{season}.js").write_text(
        "window.WBB_LINEUPS = " + json.dumps(lineup_data) + ";"
    )

    is_latest = season == max(available_seasons())
    hrefs = _page_hrefs(season)
    team_pages = _rated_team_ids()

    lr = validation.get("lineups", {})
    coverage_note = None
    if lr.get("n_games") and lr.get("n_excluded", 0) / lr["n_games"] > 0.2 and lr.get("included_date_range"):
        d0, d1 = lr["included_date_range"]
        coverage_note = (
            f"Partial season: ESPN only recorded substitutions for "
            f"{lr['included_games']} of {lr['n_games']} games "
            f"({d0} to {d1}), so these ratings cover late-season play only."
        )

    common = {
        "season": season,
        "season_label": season_label(season),
        "min_minutes_default": MIN_MINUTES_DEFAULT_FILTER,
        "low_minutes_flag": LOW_MINUTES_FLAG,
        "lineup_min_poss": LINEUP_MIN_POSS,
        "coverage_note": coverage_note,
        "models": [{"key": m, "label": MODEL_SPECS[m]["label"]} for m in models_available],
    }

    def write(name: str, html: str, alias: str | None = None) -> None:
        (SITE_DIR / name).write_text(html)
        if is_latest and alias:
            (SITE_DIR / alias).write_text(html)

    # --- leaderboard ---
    team_options = ratings.dropna(subset=["team"]).drop_duplicates("team")[["team"]]
    team_options = team_options.sort_values("team").to_dict(orient="records")
    index_html = env.get_template("index.html.j2").render(
        page="index", rel="", nav=hrefs, teams=team_options,
        season_switch=_season_switch(season, "index", ""), **common
    )
    write(hrefs["index"], index_html, "index.html")

    # --- team pages ---
    summaries = _team_summaries(ratings, team_box, teams_ref)
    n_teams = len(summaries)
    names = dict(zip(ratings["athlete_id"], ratings["name"]))
    roster_cols = ["athlete_id", "rank", "name", "position", "headshot", "games", "minutes",
                   "orapm", "drapm", "rapm", "low_sample"] + \
                  [c for c in DECOMP_COLS if c in ratings.columns]

    teamindex_html = env.get_template("teamindex.html.j2").render(
        page="teams", rel="", nav=hrefs, teams=summaries.to_dict(orient="records"),
        season_switch=_season_switch(season, "teams", ""), **common
    )
    write(hrefs["teams"], teamindex_html, "teamindex.html")

    # --- lineup explorer ---
    stanford_team_id = next(
        (t["id"] for t in lineup_data["teams"] if t["name"] == STANFORD_TEAM_NAME), ""
    )
    lineups_html = env.get_template("lineups.html.j2").render(
        page="lineups", rel="", nav=hrefs, default_team=stanford_team_id,
        season_switch=_season_switch(season, "lineups", ""), **common
    )
    write(hrefs["lineups"], lineups_html, "lineups.html")

    team_tpl = env.get_template("team.html.j2")
    nav_up = {k: "../../" + v for k, v in hrefs.items()}
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
        html = team_tpl.render(
            page="team", rel="../../", nav=nav_up, team=team_ctx,
            season_switch=_season_switch(season, "team", "../../", team_pages, tid), **common
        )
        (SITE_DIR / "teams" / str(season) / f"{tid}.html").write_text(html)
        if STANFORD_TEAM_NAME.lower() in str(row.name).lower():
            stanford_html = team_tpl.render(
                page="stanford", rel="", nav=hrefs, team=team_ctx,
                season_switch=_season_switch(season, "stanford", ""), **common
            )
            write(hrefs["stanford"], stanford_html, "stanford.html")

    # --- about / validation ---
    class DotDict(dict):
        __getattr__ = dict.get

    v = json.loads(json.dumps(validation), object_hook=DotDict)
    about_html = env.get_template("about.html.j2").render(
        page="about", rel="", nav=hrefs, v=v,
        season_switch=_season_switch(season, "about", ""), **common
    )
    write(hrefs["about"], about_html, "about.html")


def build_all_sites() -> None:
    """Rebuild every built season's pages (so season dropdowns stay complete)
    plus the cross-season player history."""
    for s in available_seasons():
        build_site(s)
    build_history()
