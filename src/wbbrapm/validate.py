"""Stage 5: QA checks.

1. Lineup-minute reconciliation: reconstructed on-court seconds vs box-score
   minutes, per game. Games with mean absolute error above the threshold are
   excluded from the model.
2. Team aggregation sanity check: minute-weighted team RAPM should correlate
   positively with team net rating / win pct from the team box scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MINUTE_MAE_THRESHOLD


def game_minute_mae(player_seconds: dict[int, float], box: pd.DataFrame) -> float:
    """Mean abs error (minutes/player) between reconstruction and box score."""
    played = box[~box["did_not_play"].fillna(False) & box["minutes"].notna()]
    if played.empty:
        return float("nan")
    errs = [
        abs(player_seconds.get(int(r.athlete_id), 0.0) / 60.0 - float(r.minutes))
        for r in played.itertuples(index=False)
    ]
    return float(np.mean(errs))


def reconciliation_report(game_maes: dict[int, float], threshold=MINUTE_MAE_THRESHOLD) -> dict:
    maes = np.array([m for m in game_maes.values() if np.isfinite(m)])
    bad = [g for g, m in game_maes.items() if not np.isfinite(m) or m > threshold]
    return {
        "n_games": len(game_maes),
        "mean_mae_minutes": float(maes.mean()) if len(maes) else None,
        "median_mae_minutes": float(np.median(maes)) if len(maes) else None,
        "p95_mae_minutes": float(np.percentile(maes, 95)) if len(maes) else None,
        "threshold": threshold,
        "n_excluded": len(bad),
        "excluded_game_ids": sorted(bad),
    }


def team_sanity_check(ratings: pd.DataFrame, team_box: pd.DataFrame) -> dict:
    """Minute-weighted team RAPM vs team net rating and win pct."""
    r = ratings.dropna(subset=["team_id"]).copy()
    r["w"] = r["minutes"].clip(lower=0)
    team_rapm = (
        r.groupby("team_id")
        .apply(lambda g: np.average(g["rapm"], weights=g["w"]) if g["w"].sum() > 0 else np.nan,
               include_groups=False)
        .rename("team_rapm")
        .reset_index()
    )

    tb = team_box.copy()
    tb["margin"] = tb["team_score"] - tb["opponent_team_score"]
    tb["win"] = tb["team_winner"].astype(bool)
    team_perf = tb.groupby("team_id").agg(
        net_margin=("margin", "mean"),
        win_pct=("win", "mean"),
        games=("game_id", "nunique"),
        team_name=("team_short_display_name", "last"),
    ).reset_index()

    m = team_rapm.merge(team_perf, on="team_id").dropna(subset=["team_rapm", "net_margin"])
    pearson_margin = float(np.corrcoef(m["team_rapm"], m["net_margin"])[0, 1])
    pearson_win = float(np.corrcoef(m["team_rapm"], m["win_pct"])[0, 1])
    spearman_margin = float(m["team_rapm"].corr(m["net_margin"], method="spearman"))
    return {
        "n_teams": len(m),
        "pearson_rapm_vs_net_margin": pearson_margin,
        "pearson_rapm_vs_win_pct": pearson_win,
        "spearman_rapm_vs_net_margin": spearman_margin,
        "passed": pearson_margin > 0.5,
        "teams": m.sort_values("team_rapm", ascending=False)[
            ["team_id", "team_name", "team_rapm", "net_margin", "win_pct", "games"]
        ].to_dict(orient="records"),
    }
