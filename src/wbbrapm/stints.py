"""Stage 3: collapse lineup-annotated pbp into constant-lineup stints.

A stint is a maximal run of events (within one period) where all 10 on-court
players are unchanged. Each stint records both lineups, elapsed seconds,
points by side, and possession estimates via the standard box proxy
FGA - OREB + TO + 0.44*FTA counted from the events inside the stint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FTA_POSSESSION_WEIGHT
from .lineups import GameLineups

FT_TYPE = "MadeFreeThrow"  # covers makes AND misses; scoring_play distinguishes
FGA_TYPES = {"JumpShot", "LayUpShot", "TipShot", "DunkShot", "Shot"}
TO_TYPES = {"Lost Ball Turnover"}
OREB_TYPES = {"Offensive Rebound"}


def build_stints(pbp: pd.DataFrame, gl: GameLineups) -> pd.DataFrame:
    """pbp: one game's rows in the same order used for reconstruction."""
    n = len(pbp)
    if n == 0 or not gl.ok:
        return pd.DataFrame()

    home_id = int(pbp["home_team_id"].iloc[0])
    away_id = int(pbp["away_team_id"].iloc[0])

    periods = pbp["period_number"].to_numpy()
    types = pbp["type_text"].to_numpy()
    team_ids = pbp["team_id"].to_numpy(dtype=float)
    scoring = pbp["scoring_play"].fillna(False).to_numpy(dtype=bool)
    score_value = pbp["score_value"].fillna(0).to_numpy(dtype=float)

    # Mid-substitution the on-court sets briefly hold 4 or 6 players (a paired
    # "subbing out"/"subbing in" arrives as two events, sometimes with free
    # throws in between). Attribute those transitional moments to the next
    # settled 5v5 lineup so every stint is a true 5-on-5.
    hl = list(gl.home_lineup)
    al = list(gl.away_lineup)
    valid = [len(hl[i]) == 5 and len(al[i]) == 5 for i in range(n)]
    if not any(valid):
        return pd.DataFrame()
    nh = na = None
    for i in range(n - 1, -1, -1):  # backward fill from the next valid state
        if valid[i]:
            nh, na = hl[i], al[i]
        else:
            hl[i], al[i] = nh, na
    ph = pa = None
    for i in range(n):  # a trailing invalid run falls back to the last valid state
        if hl[i] is not None:
            ph, pa = hl[i], al[i]
        else:
            hl[i], al[i] = ph, pa

    # stint id increments whenever either lineup or the period changes
    stint_id = np.zeros(n, dtype=int)
    sid = 0
    for i in range(1, n):
        if hl[i] != hl[i - 1] or al[i] != al[i - 1] or periods[i] != periods[i - 1]:
            sid += 1
        stint_id[i] = sid

    # stint s is on the clock from its first event (the sub that created it)
    # until the first event of stint s+1; chaining the boundaries this way
    # partitions the full game clock with no gaps or double counting. When the
    # next stint starts in a new period, the boundary is the period break, not
    # that stint's first event (which may come seconds into the period).
    from .lineups import period_start_seconds

    first_idx = np.searchsorted(stint_id, np.arange(sid + 1), side="left")
    end_t = np.append(gl.elapsed[first_idx[1:]], gl.elapsed[n - 1])
    for s in range(sid):
        nxt = first_idx[s + 1]
        if periods[nxt] != periods[first_idx[s]]:
            end_t[s] = min(end_t[s], period_start_seconds(int(periods[nxt])))
    start_t = np.concatenate([[0.0], end_t[:-1]])

    rows = []
    for s in range(sid + 1):
        idx = np.where(stint_id == s)[0]
        first = idx[0]
        # lineup arrays hold post-event state, so the stint's lineup is the
        # one in effect at its first event
        h_lineup = hl[first]
        a_lineup = al[first]
        seconds = float(end_t[s] - start_t[s])

        stats = {home_id: dict(pts=0.0, fga=0, fta=0, to=0, oreb=0),
                 away_id: dict(pts=0.0, fga=0, fta=0, to=0, oreb=0)}
        for i in idx:
            tid = team_ids[i]
            if not np.isfinite(tid) or int(tid) not in stats:
                continue
            st = stats[int(tid)]
            typ = types[i]
            if scoring[i]:
                st["pts"] += score_value[i]
            if typ in FGA_TYPES:
                st["fga"] += 1
            elif typ == FT_TYPE:
                st["fta"] += 1
            elif typ in TO_TYPES:
                st["to"] += 1
            elif typ in OREB_TYPES:
                st["oreb"] += 1

        h, a = stats[home_id], stats[away_id]
        rows.append({
            "game_id": gl.game_id,
            "stint": s,
            "period": int(periods[first]),
            "seconds": seconds,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_lineup": list(h_lineup),
            "away_lineup": list(a_lineup),
            "home_pts": h["pts"],
            "away_pts": a["pts"],
            "home_poss": _poss(h),
            "away_poss": _poss(a),
        })
    return pd.DataFrame(rows)


def _poss(s: dict) -> float:
    return max(s["fga"] - s["oreb"] + s["to"] + FTA_POSSESSION_WEIGHT * s["fta"], 0.0)
