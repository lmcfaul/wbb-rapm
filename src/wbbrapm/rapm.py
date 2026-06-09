"""Stage 4: ridge-regression RAPM (overall margin model + O/D split).

Every stint contributes rows whose columns are the 10 on-court players, so a
player's coefficient is automatically adjusted for the quality of teammates
and opponents she shared the floor with. The ridge penalty (lambda, chosen by
game-grouped cross-validation) shrinks low-minute players toward 0, which is
what keeps thin samples off the extremes of the leaderboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .config import CV_FOLDS, LAMBDA_GRID, LOW_MINUTES_FLAG, NON_D1_PLAYER_ID


def build_player_index(stints: pd.DataFrame, athlete_team: dict[int, int], d1_ids: set[int]):
    """Athlete -> column map; non-D1 players collapse into one shared column."""
    players: set[int] = set()
    for col in ("home_lineup", "away_lineup"):
        for lineup in stints[col]:
            players.update(int(a) for a in lineup)
    d1_players = sorted(a for a in players if athlete_team.get(a) in d1_ids)
    col_of = {a: i for i, a in enumerate(d1_players)}
    non_d1_col = len(d1_players)  # everyone else shares this column
    return col_of, non_d1_col, d1_players


def _lineup_cols(lineup, col_of, non_d1_col):
    return [col_of.get(int(a), non_d1_col) for a in lineup]


def build_margin_system(stints: pd.DataFrame, col_of, non_d1_col):
    """One row per stint: +1 home players, -1 away players, y = margin/100poss."""
    n_cols = non_d1_col + 1
    rows, cols, vals, ys, ws, groups = [], [], [], [], [], []
    r = 0
    for s in stints.itertuples(index=False):
        w = (s.home_poss + s.away_poss) / 2.0
        if w <= 0:
            continue
        for c in _lineup_cols(s.home_lineup, col_of, non_d1_col):
            rows.append(r); cols.append(c); vals.append(1.0)
        for c in _lineup_cols(s.away_lineup, col_of, non_d1_col):
            rows.append(r); cols.append(c); vals.append(-1.0)
        ys.append(100.0 * (s.home_pts - s.away_pts) / w)
        ws.append(w)
        groups.append(s.game_id)
        r += 1
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n_cols))
    return X, np.array(ys), np.array(ws), np.array(groups)


def build_od_system(stints: pd.DataFrame, col_of, non_d1_col):
    """Two rows per stint (each side's offense). Columns: [O block | D block | home-offense].

    O coefficient: points per 100 the player adds on offense.
    D coefficient: points per 100 the player *allows* on defense (lower = better).
    """
    n_players = non_d1_col + 1
    n_cols = 2 * n_players + 1
    rows, cols, vals, ys, ws, groups = [], [], [], [], [], []
    r = 0
    for s in stints.itertuples(index=False):
        sides = [
            (s.home_lineup, s.away_lineup, s.home_pts, s.home_poss, 1.0),
            (s.away_lineup, s.home_lineup, s.away_pts, s.away_poss, 0.0),
        ]
        for off, deff, pts, poss, is_home in sides:
            if poss <= 0:
                continue
            for c in _lineup_cols(off, col_of, non_d1_col):
                rows.append(r); cols.append(c); vals.append(1.0)
            for c in _lineup_cols(deff, col_of, non_d1_col):
                rows.append(r); cols.append(n_players + c); vals.append(1.0)
            if is_home:
                rows.append(r); cols.append(2 * n_players); vals.append(1.0)
            ys.append(100.0 * pts / poss)
            ws.append(poss)
            groups.append(s.game_id)
            r += 1
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n_cols))
    return X, np.array(ys), np.array(ws), np.array(groups)


def _fit(X, y, w, lam) -> np.ndarray:
    model = Ridge(alpha=lam, fit_intercept=True, solver="sparse_cg")
    model.fit(X, y, sample_weight=w)
    return model


def cv_lambda(X, y, w, groups, grid=LAMBDA_GRID, n_splits=CV_FOLDS) -> tuple[float, dict]:
    """Pick lambda by weighted MSE under GroupKFold grouped by game_id."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = {lam: 0.0 for lam in grid}
    for train, test in gkf.split(X, y, groups):
        for lam in grid:
            m = _fit(X[train], y[train], w[train], lam)
            resid = y[test] - m.predict(X[test])
            scores[lam] += float(np.average(resid**2, weights=w[test]))
    best = min(scores, key=scores.get)
    return best, scores


def fit_models(stints: pd.DataFrame, col_of, non_d1_col, lam_margin=None, lam_od=None, verbose=True):
    """Fit both models; returns per-player coefficient arrays + chosen lambdas."""
    Xm, ym, wm, gm = build_margin_system(stints, col_of, non_d1_col)
    Xo, yo, wo, go = build_od_system(stints, col_of, non_d1_col)

    cv_scores = {}
    if lam_margin is None:
        lam_margin, cv_scores["margin"] = cv_lambda(Xm, ym, wm, gm)
    if lam_od is None:
        lam_od, cv_scores["od"] = cv_lambda(Xo, yo, wo, go)
    if verbose:
        print(f"  lambda (margin) = {lam_margin}, lambda (O/D) = {lam_od}")

    margin_model = _fit(Xm, ym, wm, lam_margin)
    od_model = _fit(Xo, yo, wo, lam_od)

    n_players = non_d1_col + 1
    coefs = {
        "rapm_margin": margin_model.coef_[:n_players],
        "orapm": od_model.coef_[:n_players],
        # negate: D coefficient is points *allowed*, so lower is better
        "drapm": -od_model.coef_[n_players: 2 * n_players],
        "hca_margin": float(margin_model.intercept_),
        "hca_od": float(od_model.coef_[2 * n_players]),
        "lam_margin": lam_margin,
        "lam_od": lam_od,
        "cv_scores": cv_scores,
        "n_stints": Xm.shape[0],
    }
    return coefs


def assemble_ratings(
    coefs: dict,
    d1_players: list[int],
    player_seconds: dict[int, float],
    player_box: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Join coefficients with identity/minutes metadata for output + site."""
    played = player_box[~player_box["did_not_play"].fillna(False)]
    meta = (
        played.sort_values("game_date")
        .groupby("athlete_id")
        .agg(
            name=("athlete_display_name", "last"),
            team_id=("team_id", "last"),
            team=("team_short_display_name", "last"),
            team_logo=("team_logo", "last"),
            team_color=("team_color", "last"),
            position=("athlete_position_abbreviation", "last"),
            headshot=("athlete_headshot_href", "last"),
            games=("game_id", "nunique"),
            box_minutes=("minutes", "sum"),
        )
        .reset_index()
    )

    idx = {a: i for i, a in enumerate(d1_players)}
    df = pd.DataFrame({"athlete_id": d1_players})
    df["rapm_margin"] = [coefs["rapm_margin"][idx[a]] for a in d1_players]
    df["orapm"] = [coefs["orapm"][idx[a]] for a in d1_players]
    df["drapm"] = [coefs["drapm"][idx[a]] for a in d1_players]
    df["rapm"] = df["orapm"] + df["drapm"]
    df["minutes"] = [player_seconds.get(a, 0.0) / 60.0 for a in d1_players]
    df = df.merge(meta, on="athlete_id", how="left")
    df["season"] = season
    df["low_sample"] = df["minutes"] < LOW_MINUTES_FLAG
    df = df.sort_values("rapm", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df
