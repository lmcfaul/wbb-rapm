#!/usr/bin/env python3
"""Run the full RAPM pipeline for one season.

Usage: python run.py --season 2026 [--skip-fetch] [--lam-margin X --lam-od Y] [--no-site]
(wehoop convention: 2026 == the 2025-26 season)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from wbbrapm import io, lineups, rapm, stints, validate  # noqa: E402
from wbbrapm.config import PROCESSED_DIR, SeasonPaths  # noqa: E402


def fetch(season: int) -> None:
    paths = SeasonPaths(season)
    if all(p.exists() for p in (paths.pbp, paths.player_box, paths.team_box, paths.teams)):
        print(f"[fetch] cached raw data found for {season}, skipping")
        return
    print(f"[fetch] pulling season {season} via wehoop ...")
    subprocess.run(
        ["Rscript", str(ROOT / "R" / "fetch_data.R"), "--season", str(season)],
        check=True, cwd=ROOT,
    )


def build_stints_for_season(season: int):
    paths = SeasonPaths(season)
    print("[load] reading raw extracts ...")
    pbp = io.load_pbp(paths)
    player_box = io.load_player_box(paths)

    print("[lineups] reconstructing on-court lineups ...")
    starters, rosters, names = lineups.build_game_inputs(player_box)
    box_by_game = dict(tuple(player_box.groupby("game_id")))

    all_stints = []
    game_maes: dict[int, float] = {}
    qa_counts = dict(toggle_errors=0, forced_on=0, overfull=0, bad_starters=0)
    t0 = time.time()
    n_games = pbp["game_id"].nunique()
    for k, (gid, gpbp) in enumerate(pbp.groupby("game_id", sort=False)):
        gid = int(gid)
        gl = lineups.reconstruct_game(gpbp, starters.get(gid, {}), rosters.get(gid, {}), names.get(gid, {}))
        if not gl.ok:
            qa_counts["bad_starters"] += 1
            game_maes[gid] = float("nan")
            continue
        qa_counts["toggle_errors"] += gl.n_toggle_errors
        qa_counts["forced_on"] += gl.n_forced_on
        qa_counts["overfull"] += gl.n_overfull
        game_maes[gid] = validate.game_minute_mae(gl.player_seconds, box_by_game[gid])
        df = stints.build_stints(gpbp, gl)
        if not df.empty:
            df["minute_mae"] = game_maes[gid]
            df["player_seconds"] = [gl.player_seconds] + [None] * (len(df) - 1)
            all_stints.append(df)
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{n_games} games ({time.time() - t0:.0f}s)")

    report = validate.reconciliation_report(game_maes)
    report["qa_counts"] = qa_counts
    print(f"[lineups] minute MAE mean={report['mean_mae_minutes']:.3f} "
          f"median={report['median_mae_minutes']:.3f} p95={report['p95_mae_minutes']:.3f}; "
          f"excluding {report['n_excluded']}/{report['n_games']} games")

    stint_df = pd.concat(all_stints, ignore_index=True)
    paths.lineup_report.parent.mkdir(parents=True, exist_ok=True)
    paths.lineup_report.write_text(json.dumps(report, indent=2))
    return stint_df, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--lam-margin", type=float, default=None)
    ap.add_argument("--lam-od", type=float, default=None)
    ap.add_argument("--no-site", action="store_true")
    args = ap.parse_args()
    season = args.season
    paths = SeasonPaths(season)

    if not args.skip_fetch:
        fetch(season)

    stint_df, lineup_report = build_stints_for_season(season)

    # exclude games that failed minute reconciliation
    good = stint_df["minute_mae"] <= lineup_report["threshold"]
    player_seconds: dict[int, float] = {}
    for d in stint_df.loc[good & stint_df["player_seconds"].notna(), "player_seconds"]:
        for a, s in d.items():
            player_seconds[a] = player_seconds.get(a, 0.0) + s
    model_stints = stint_df[good].drop(columns=["player_seconds"]).reset_index(drop=True)
    print(f"[stints] {len(model_stints)} stints from {model_stints['game_id'].nunique()} games")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model_stints.to_parquet(paths.stints)

    print("[rapm] building design matrices + fitting ...")
    player_box = io.load_player_box(paths)
    played = player_box[~player_box["did_not_play"].fillna(False)]
    athlete_team = (
        played.groupby("athlete_id")["team_id"].agg(lambda s: s.mode().iloc[0]).astype(int).to_dict()
    )
    d1_ids = io.d1_team_ids(paths)
    col_of, non_d1_col, d1_players = rapm.build_player_index(model_stints, athlete_team, d1_ids)
    print(f"  {len(d1_players)} D1 players, non-D1 collapsed to 1 column")

    coefs = rapm.fit_models(model_stints, col_of, non_d1_col,
                            lam_margin=args.lam_margin, lam_od=args.lam_od)
    ratings = rapm.assemble_ratings(coefs, d1_players, player_seconds, player_box, season)
    ratings.to_parquet(paths.ratings)
    print(f"[rapm] wrote {paths.ratings}")

    print("[validate] team aggregation sanity check ...")
    team_box = io.load_team_box(paths)
    d1_team_box = team_box[team_box["team_id"].isin(d1_ids)]
    sanity = validate.team_sanity_check(ratings, d1_team_box)
    report = {
        "lineups": lineup_report,
        "model": {k: coefs[k] for k in ("hca_margin", "hca_od", "lam_margin", "lam_od", "n_stints")},
        "cv_scores": coefs.get("cv_scores", {}),
        "team_sanity": sanity,
    }
    paths.validation_report.write_text(json.dumps(report, indent=2, default=float))
    print(f"  corr(team RAPM, net margin) = {sanity['pearson_rapm_vs_net_margin']:.3f} "
          f"| corr(team RAPM, win%) = {sanity['pearson_rapm_vs_win_pct']:.3f} "
          f"| passed = {sanity['passed']}")

    if not args.no_site:
        from wbbrapm import site
        site.build_site(season)
        print("[site] rebuilt static site")


if __name__ == "__main__":
    main()
