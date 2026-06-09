"""Read cached wehoop extracts and derive the D1 team reference set."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SeasonPaths


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `Rscript R/fetch_data.R --season <year>` first"
        )
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=columns)
    else:
        df = pd.read_csv(path, low_memory=False)
        if columns:
            df = df[[c for c in columns if c in df.columns]]
    return df


PBP_COLUMNS = [
    "game_id",
    "sequence_number",
    "period_number",
    "clock_display_value",
    "type_id",
    "type_text",
    "text",
    "team_id",
    "athlete_id_1",
    "athlete_id_2",
    "scoring_play",
    "shooting_play",
    "score_value",
    "home_score",
    "away_score",
    "home_team_id",
    "away_team_id",
    "season",
    "game_play_number",
    "start_quarter_seconds_remaining",
    "start_game_seconds_remaining",
]


def load_pbp(paths: SeasonPaths) -> pd.DataFrame:
    df = _read(paths.pbp)
    cols = [c for c in PBP_COLUMNS if c in df.columns]
    df = df[cols].copy()
    for c in ("team_id", "athlete_id_1", "athlete_id_2", "home_team_id", "away_team_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df = df.sort_values(["game_id", "period_number", "game_play_number"], kind="stable")
    return df


def load_player_box(paths: SeasonPaths) -> pd.DataFrame:
    df = _read(paths.player_box)
    for c in ("athlete_id", "team_id", "game_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "minutes" in df.columns:
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    return df


def load_team_box(paths: SeasonPaths) -> pd.DataFrame:
    df = _read(paths.team_box)
    for c in ("team_id", "game_id", "opponent_team_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def load_teams(paths: SeasonPaths) -> pd.DataFrame:
    """espn_wbb_teams() output: the D1 reference set for the season."""
    df = _read(paths.teams)
    id_col = next((c for c in ("team_id", "id") if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"no team id column in {paths.teams}: {list(df.columns)}")
    df = df.rename(columns={id_col: "team_id"})
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    return df


def d1_team_ids(paths: SeasonPaths) -> set[int]:
    return set(load_teams(paths)["team_id"].dropna().astype(int))
