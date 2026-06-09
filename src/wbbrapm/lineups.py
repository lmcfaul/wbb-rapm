"""Stage 2: reconstruct the 10 on-court players for every pbp event.

ESPN WBB substitution text is explicitly directional ("X subbing in for TEAM" /
"X subbing out for TEAM"), so the primary signal is parsed from text. Two
safety nets correct residual errors:

1. any player who records a stat event is forced on court (you cannot shoot,
   rebound, foul, steal, or turn the ball over from the bench);
2. reconstructed seconds are reconciled against box-score minutes in
   validate.py, and games above the error threshold are excluded.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import OT_SECONDS, QUARTER_SECONDS

SUB_TYPE = "Substitution"
# Event types that never place athlete_id_1 on the floor for their own team.
NON_PLAYER_TYPES = {
    SUB_TYPE,
    "End Period",
    "End Game",
    "OfficialTVTimeOut",
    "ShortTimeOut",
    "RegularTimeOut",
}

_IN_RE = re.compile(r"subbing in", re.IGNORECASE)
_OUT_RE = re.compile(r"subbing out", re.IGNORECASE)


def period_start_seconds(period: int) -> int:
    """Elapsed game seconds at the start of a (1-indexed) period."""
    if period <= 4:
        return (period - 1) * QUARTER_SECONDS
    return 4 * QUARTER_SECONDS + (period - 5) * OT_SECONDS


def period_length(period: int) -> int:
    return QUARTER_SECONDS if period <= 4 else OT_SECONDS


@dataclass
class GameLineups:
    """Per-event lineup annotation plus QA counters for one game."""

    game_id: int
    # arrays aligned with the game's pbp rows
    elapsed: np.ndarray
    home_lineup: list[tuple[int, ...]]
    away_lineup: list[tuple[int, ...]]
    player_seconds: dict[int, float]
    n_toggle_errors: int = 0
    n_forced_on: int = 0
    n_overfull: int = 0
    ok: bool = True
    reason: str = ""


@dataclass
class _TeamState:
    on: set[int] = field(default_factory=set)
    toggle_errors: int = 0
    forced_on: int = 0
    overfull: int = 0
    # who most recently entered, used to break ties when trimming a 6th player
    last_change: dict[int, int] = field(default_factory=dict)

    def sub(self, athlete: int, direction: str, seq: int) -> None:
        if direction == "in":
            if athlete in self.on:
                self.toggle_errors += 1
            self.on.add(athlete)
        else:
            if athlete not in self.on:
                self.toggle_errors += 1
            self.on.discard(athlete)
        self.last_change[athlete] = seq

    def force_on(self, athlete: int, seq: int) -> None:
        if athlete not in self.on:
            self.on.add(athlete)
            self.forced_on += 1
            self.last_change[athlete] = seq

    def trim(self, seq: int) -> None:
        """If more than 5 are on court, drop the stalest entrants."""
        while len(self.on) > 5:
            self.overfull += 1
            stalest = min(self.on, key=lambda a: self.last_change.get(a, -1))
            self.on.discard(stalest)


def _sub_direction(text: str) -> str | None:
    if _IN_RE.search(text):
        return "in"
    if _OUT_RE.search(text):
        return "out"
    return None


def _match_name(text: str, names_to_id: dict[str, int]) -> int | None:
    """Recover a missing athlete_id on a sub row from the leading player name."""
    m = re.match(r"^(.*?)\s+subbing\s+(?:in|out)\s+for\s+", text, re.IGNORECASE)
    if not m:
        return None
    return names_to_id.get(m.group(1).strip().lower())


def reconstruct_game(
    pbp: pd.DataFrame,
    starters: dict[int, set[int]],
    roster: dict[int, int],
    names_to_id: dict[int, dict[str, int]],
) -> GameLineups:
    """Walk one game's pbp chronologically, maintaining both on-court sets.

    pbp: this game's rows, already in chronological order.
    starters: team_id -> set of 5 starter athlete_ids.
    roster: athlete_id -> team_id for everyone who played in this game.
    names_to_id: team_id -> {lowercased display name -> athlete_id}.
    """
    game_id = int(pbp["game_id"].iloc[0])
    home_id = int(pbp["home_team_id"].iloc[0])
    away_id = int(pbp["away_team_id"].iloc[0])

    if set(starters) != {home_id, away_id} or any(len(s) != 5 for s in starters.values()):
        return GameLineups(game_id, np.array([]), [], [], {}, ok=False, reason="bad starters")

    state = {home_id: _TeamState(set(starters[home_id])), away_id: _TeamState(set(starters[away_id]))}

    n = len(pbp)
    elapsed = np.empty(n, dtype=float)
    home_lineups: list[tuple[int, ...]] = [()] * n
    away_lineups: list[tuple[int, ...]] = [()] * n
    player_seconds: dict[int, float] = defaultdict(float)

    periods = pbp["period_number"].to_numpy()
    qsr = pbp["start_quarter_seconds_remaining"].to_numpy(dtype=float)
    types = pbp["type_text"].to_numpy()
    texts = pbp["text"].astype(str).to_numpy()
    team_ids = pbp["team_id"].to_numpy(dtype=float)
    a1 = pbp["athlete_id_1"].to_numpy(dtype=float)
    a2 = pbp["athlete_id_2"].to_numpy(dtype=float)

    prev_t = 0.0
    prev_home: tuple[int, ...] = tuple(sorted(state[home_id].on))
    prev_away: tuple[int, ...] = tuple(sorted(state[away_id].on))

    for i in range(n):
        p = int(periods[i])
        rem = qsr[i] if np.isfinite(qsr[i]) else period_length(p)
        rem = min(max(rem, 0.0), period_length(p))
        t = period_start_seconds(p) + (period_length(p) - rem)
        t = max(t, prev_t)  # clock noise can make time appear to run backwards
        elapsed[i] = t

        # accrue seconds for whoever was on court since the previous event
        dt = t - prev_t
        if dt > 0:
            for a in prev_home:
                player_seconds[a] += dt
            for a in prev_away:
                player_seconds[a] += dt
        prev_t = t

        typ = types[i]
        if typ == SUB_TYPE:
            direction = _sub_direction(texts[i])
            tid = int(team_ids[i]) if np.isfinite(team_ids[i]) else None
            ath = int(a1[i]) if np.isfinite(a1[i]) else None
            if ath is None and tid in names_to_id:
                ath = _match_name(texts[i], names_to_id[tid])
            if ath is not None and tid is None:
                tid = roster.get(ath)
            if direction is not None and ath is not None and tid in state:
                state[tid].sub(ath, direction, i)
                state[tid].trim(i)
        elif typ not in NON_PLAYER_TYPES:
            for raw in (a1[i], a2[i]):
                if np.isfinite(raw):
                    ath = int(raw)
                    tid = roster.get(ath)
                    if tid in state:
                        state[tid].force_on(ath, i)
                        state[tid].trim(i)

        prev_home = tuple(sorted(state[home_id].on))
        prev_away = tuple(sorted(state[away_id].on))
        home_lineups[i] = prev_home
        away_lineups[i] = prev_away

    qa = [state[home_id], state[away_id]]
    return GameLineups(
        game_id=game_id,
        elapsed=elapsed,
        home_lineup=home_lineups,
        away_lineup=away_lineups,
        player_seconds=dict(player_seconds),
        n_toggle_errors=sum(s.toggle_errors for s in qa),
        n_forced_on=sum(s.forced_on for s in qa),
        n_overfull=sum(s.overfull for s in qa),
    )


def build_game_inputs(player_box: pd.DataFrame):
    """Precompute starters / rosters / name maps for every game."""
    played = player_box[~player_box["did_not_play"].fillna(False)]
    starters_df = played[played["starter"] == True]  # noqa: E712

    starters: dict[int, dict[int, set[int]]] = defaultdict(dict)
    for (gid, tid), grp in starters_df.groupby(["game_id", "team_id"]):
        starters[int(gid)][int(tid)] = set(grp["athlete_id"].astype(int))

    rosters: dict[int, dict[int, int]] = defaultdict(dict)
    names: dict[int, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for row in played[["game_id", "team_id", "athlete_id", "athlete_display_name"]].itertuples(index=False):
        gid, tid, aid = int(row.game_id), int(row.team_id), int(row.athlete_id)
        rosters[gid][aid] = tid
        names[gid][tid][str(row.athlete_display_name).lower()] = aid
    return starters, rosters, names
