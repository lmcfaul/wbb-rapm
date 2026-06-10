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

    @property
    def phases(self) -> Path:
        return PROCESSED_DIR / f"phases_{self.season}.parquet"

    @property
    def lineup_report(self) -> Path:
        return PROCESSED_DIR / f"lineup_qa_{self.season}.json"

    @property
    def validation_report(self) -> Path:
        return PROCESSED_DIR / f"validation_{self.season}.json"
