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
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GroupKFold

from .config import CV_FOLDS, LAMBDA_GRID, LOW_MINUTES_FLAG, MODEL_SPECS, NON_D1_PLAYER_ID


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


def _make_estimator(kind: str, lam: float | None = None, system: str = "margin"):
    """lam only applies to ridge; lasso/enet use their calibrated alphas,
    with a lighter alpha on the O/D system (different response scale)."""
    spec = MODEL_SPECS[kind]
    if kind == "ridge":
        return Ridge(alpha=lam, fit_intercept=True, solver="sparse_cg")
    alpha = spec["alpha_od"] if system == "od" and "alpha_od" in spec else spec["alpha"]
    if kind == "lasso":
        return Lasso(alpha=alpha, fit_intercept=True, max_iter=3000, tol=1e-3)
    if kind == "enet":
        return ElasticNet(alpha=alpha, l1_ratio=spec["l1_ratio"],
                          fit_intercept=True, max_iter=3000, tol=1e-3)
    raise ValueError(f"unknown model kind {kind!r}")


def _fit(X, y, w, lam, kind: str = "ridge", system: str = "margin"):
    model = _make_estimator(kind, lam, system)
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


def _package_coefs(margin_model, od_model, n_players, lam_margin, lam_od, n_stints, cv_scores):
    return {
        "rapm_margin": margin_model.coef_[:n_players],
        "orapm": od_model.coef_[:n_players],
        # negate: D coefficient is points *allowed*, so lower is better
        "drapm": -od_model.coef_[n_players: 2 * n_players],
        "hca_margin": float(margin_model.intercept_),
        "hca_od": float(od_model.coef_[2 * n_players]),
        "lam_margin": lam_margin,
        "lam_od": lam_od,
        "cv_scores": cv_scores,
        "n_stints": n_stints,
    }


def fit_models(stints: pd.DataFrame, col_of, non_d1_col, lam_margin=None, lam_od=None,
               verbose=True, model="ridge"):
    """Fit both systems with one regression family; returns coefficient arrays."""
    return fit_models_multi(stints, col_of, non_d1_col, lam_margin, lam_od,
                            verbose=verbose, models=[model])[model]


def fit_models_multi(stints: pd.DataFrame, col_of, non_d1_col, lam_margin=None, lam_od=None,
                     verbose=True, models=None):
    """Fit every requested regression family on shared design matrices.

    Returns {model_kind: coefs}. Ridge lambdas come from CV (or the args);
    lasso / elastic net use the calibrated alphas in MODEL_SPECS.
    """
    models = models or list(MODEL_SPECS)
    Xm, ym, wm, gm = build_margin_system(stints, col_of, non_d1_col)
    Xo, yo, wo, go = build_od_system(stints, col_of, non_d1_col)

    cv_scores = {}
    if "ridge" in models:
        if lam_margin is None:
            lam_margin, cv_scores["margin"] = cv_lambda(Xm, ym, wm, gm)
        if lam_od is None:
            lam_od, cv_scores["od"] = cv_lambda(Xo, yo, wo, go)
        if verbose:
            print(f"  lambda (margin) = {lam_margin}, lambda (O/D) = {lam_od}")

    out = {}
    n_players = non_d1_col + 1
    for kind in models:
        mm = _fit(Xm, ym, wm, lam_margin, kind, system="margin")
        om = _fit(Xo, yo, wo, lam_od, kind, system="od")
        out[kind] = _package_coefs(mm, om, n_players, lam_margin, lam_od,
                                   Xm.shape[0], cv_scores if kind == "ridge" else {})
        if verbose and kind != "ridge":
            nz = int((out[kind]["rapm_margin"] != 0).sum())
            print(f"  {kind}: {nz}/{n_players - 1} players nonzero (margin model)")
    return out


def fit_phase_ratings(
    stints: pd.DataFrame,
    col_of,
    non_d1_col,
    lam_od: float,
    game_dates: dict[int, str],
    n_phases: int | None = None,
) -> pd.DataFrame:
    """Refit the O/D model on date-based slices of the season (default halves).

    Games are sorted by date and split into n_phases equal-count groups; each
    phase gets its own ridge fit (same lambda as the full model, so phase
    ratings are comparable but noisier - a fraction of the data shrinks
    harder). Returns one row per (athlete, phase) with phase minutes for the
    UI to grey out thin samples.
    """
    from .config import N_PHASES

    if n_phases is None:
        n_phases = N_PHASES
    games = sorted(
        (g for g in stints["game_id"].unique() if g in game_dates),
        key=lambda g: game_dates[g],
    )
    rows = []
    for phase, gs in enumerate(np.array_split(np.array(games), n_phases)):
        sub = stints[stints["game_id"].isin(set(gs))]
        Xo, yo, wo, _ = build_od_system(sub, col_of, non_d1_col)
        m = _fit(Xo, yo, wo, lam_od)
        n_players = non_d1_col + 1
        o = m.coef_[:n_players]
        d = -m.coef_[n_players: 2 * n_players]

        seconds: dict[int, float] = {}
        for s in sub.itertuples(index=False):
            for lineup in (s.home_lineup, s.away_lineup):
                for a in lineup:
                    seconds[int(a)] = seconds.get(int(a), 0.0) + s.seconds

        start, end = game_dates[gs[0]][:10], game_dates[gs[-1]][:10]
        for a, c in col_of.items():
            rows.append({
                "athlete_id": a, "phase": phase, "start": start, "end": end,
                "orapm": float(o[c]), "drapm": float(d[c]),
                "rapm": float(o[c] + d[c]),
                "minutes": seconds.get(a, 0.0) / 60.0,
            })
    return pd.DataFrame(rows)


def decompose_ratings(stints: pd.DataFrame, col_of, non_d1_col, coefs) -> dict[str, np.ndarray]:
    """Split each player's rating into raw on-court production plus the
    adjustments the model applies for teammates and competition.

    Using the O/D model (offense points per 100 ~ intercept + sum O_off + sum D_def):

        ORAPM_i ~ raw_off_i + tm_off_i + opp_off_i
        DRAPM_i ~ raw_def_i + tm_def_i + opp_def_i

    raw_*: the player's on-court rating vs league average; tm_*: minus the
    (possession-weighted) quality of teammates she shared the floor with;
    opp_*: plus the quality of the opposition she faced. Approximate, not an
    exact identity - ridge shrinkage and home-court are not redistributed.
    """
    n_players = non_d1_col + 1
    O = np.asarray(coefs["orapm"], dtype=float)
    D = -np.asarray(coefs["drapm"], dtype=float)  # back to points-allowed sign

    off_poss = np.zeros(n_players); off_pts = np.zeros(n_players)
    def_poss = np.zeros(n_players); def_pts = np.zeros(n_players)
    tm_o = np.zeros(n_players); opp_d = np.zeros(n_players)
    tm_d = np.zeros(n_players); opp_o = np.zeros(n_players)
    tot_pts = tot_poss = 0.0

    for s in stints.itertuples(index=False):
        sides = [
            (s.home_lineup, s.away_lineup, s.home_pts, s.home_poss),
            (s.away_lineup, s.home_lineup, s.away_pts, s.away_poss),
        ]
        for off, deff, pts, poss in sides:
            if poss <= 0:
                continue
            c_off = _lineup_cols(off, col_of, non_d1_col)
            c_def = _lineup_cols(deff, col_of, non_d1_col)
            sum_o = sum(O[c] for c in c_off)
            sum_d = sum(D[c] for c in c_def)
            tot_pts += pts; tot_poss += poss
            for c in c_off:
                off_poss[c] += poss; off_pts[c] += pts
                tm_o[c] += poss * (sum_o - O[c])
                opp_d[c] += poss * sum_d
            for c in c_def:
                def_poss[c] += poss; def_pts[c] += pts
                tm_d[c] += poss * (sum_d - D[c])
                opp_o[c] += poss * sum_o

    league_avg = 100.0 * tot_pts / tot_poss
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_off = 100.0 * off_pts / off_poss - league_avg
        raw_def = league_avg - 100.0 * def_pts / def_poss
        out = {
            "raw_off": raw_off,
            "raw_def": raw_def,
            "raw_net": raw_off + raw_def,
            "tm_off": -tm_o / off_poss,
            "tm_def": tm_d / def_poss,
            "opp_off": -opp_d / off_poss,
            "opp_def": opp_o / def_poss,
        }
    out["tm_net"] = out["tm_off"] + out["tm_def"]
    out["opp_net"] = out["opp_off"] + out["opp_def"]
    return out


DECOMP_COLS = ["raw_off", "raw_def", "raw_net", "tm_off", "tm_def", "tm_net",
               "opp_off", "opp_def", "opp_net"]


def assemble_ratings(
    coefs: dict,
    d1_players: list[int],
    player_seconds: dict[int, float],
    player_box: pd.DataFrame,
    season: int,
    decomp: dict[str, np.ndarray] | None = None,
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
    if decomp is not None:
        for k in DECOMP_COLS:
            df[k] = [decomp[k][idx[a]] for a in d1_players]
    df["minutes"] = [player_seconds.get(a, 0.0) / 60.0 for a in d1_players]
    df = df.merge(meta, on="athlete_id", how="left")
    df["season"] = season
    df["low_sample"] = df["minutes"] < LOW_MINUTES_FLAG
    df = df.sort_values("rapm", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df
