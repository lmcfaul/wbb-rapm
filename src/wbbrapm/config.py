"""Central configuration for the WBB RAPM pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SITE_DIR = ROOT / "site"
TEMPLATE_DIR = ROOT / "templates"

# Regulation quarter and overtime lengths (seconds). WBB plays 4x10min + 5min OTs.
QUARTER_SECONDS = 600
OT_SECONDS = 300

# Ridge penalty grid for cross-validation (per-100-possession scale).
LAMBDA_GRID = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
CV_FOLDS = 5

# The three regression families the site can toggle between. Ridge is the
# canonical RAPM (used for team aggregates, history, and trends); lasso and
# elastic net are alternative views fit on the same design matrices. Alphas
# were calibrated on 2025-26: lasso 0.02 keeps ~1k players nonzero, enet
# sits between lasso's sparsity and ridge's dense shrinkage.
MODEL_SPECS = {
    "ridge": {"label": "Ridge (RAPM)"},
    # the O/D system needs a lighter penalty than the margin system to keep
    # a comparable number of players nonzero (calibrated on 2025-26)
    "lasso": {"label": "Lasso (sparse)", "alpha": 0.02, "alpha_od": 0.01},
    "enet": {"label": "Elastic Net", "alpha": 0.05, "alpha_od": 0.02, "l1_ratio": 0.5},
}

# Season trend: number of date-based phases the season is split into.
N_PHASES = 2

# Free-throw possession weight in the standard possession proxy.
FTA_POSSESSION_WEIGHT = 0.44

# Lineup QA: exclude games whose mean absolute per-player minute
# reconciliation error exceeds this many minutes.
MINUTE_MAE_THRESHOLD = 1.5

# Games with fewer substitution events than this cannot be reconstructed
# (ESPN feeds before Feb 2025 have no subs at all) and are excluded outright,
# whatever their minute MAE says.
MIN_SUBS_PER_GAME = 10

# Players under this many total minutes get a "low sample" flag on the site
# (they stay in the model; ridge already shrinks them).
LOW_MINUTES_FLAG = 200

# Site display: hide players under this many minutes from the default view.
MIN_MINUTES_DEFAULT_FILTER = 100

NON_D1_PLAYER_ID = "NON_D1"

STANFORD_TEAM_NAME = "Stanford"


@dataclass
class SeasonPaths:
    """All input/output paths for one season."""

    season: int
    ext: str = "parquet"  # fetch_data.R falls back to csv.gz when arrow is missing

    def _raw(self, name: str) -> Path:
        for ext in (self.ext, "csv.gz"):
            p = RAW_DIR / f"{name}_{self.season}.{ext}"
            if p.exists():
                return p
        return RAW_DIR / f"{name}_{self.season}.{self.ext}"

    @property
    def pbp(self) -> Path:
        return self._raw("pbp")

    @property
    def player_box(self) -> Path:
        return self._raw("player_box")

    @property
    def team_box(self) -> Path:
        return self._raw("team_box")

    @property
    def teams(self) -> Path:
        return self._raw("teams")

    @property
    def stints(self) -> Path:
        return PROCESSED_DIR / f"stints_{self.season}.parquet"

    @property
    def ratings(self) -> Path:
        return PROCESSED_DIR / f"ratings_{self.season}.parquet"

    def ratings_for(self, model: str) -> Path:
        """ridge is the canonical ratings file; other models get a suffix."""
        if model == "ridge":
            return self.ratings
        return PROCESSED_DIR / f"ratings_{self.season}_{model}.parquet"

    @property
    def phases(self) -> Path:
        return PROCESSED_DIR / f"phases_{self.season}.parquet"

    @property
    def lineup_report(self) -> Path:
        return PROCESSED_DIR / f"lineup_qa_{self.season}.json"

    @property
    def validation_report(self) -> Path:
        return PROCESSED_DIR / f"validation_{self.season}.json"
