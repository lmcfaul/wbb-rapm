# WBB RAPM — D1 Women's Basketball Player Ratings

Season-long, opponent- and teammate-adjusted player impact ratings (RAPM) for all of
Division I women's basketball, plus a fully static explorer site with a page for every
team (including a dedicated Stanford page).

## What the rating means

**RAPM** (regularized adjusted plus-minus) is each player's effect on point margin per
100 possessions. Every constant-lineup stint of every game becomes a ridge-regression
row whose predictors are the ten players on the floor, so each coefficient is
automatically adjusted for teammate and opponent quality. The ridge penalty λ (chosen
by game-grouped cross-validation) shrinks low-minute players toward average, so thin
samples can't top the leaderboard. A second two-rows-per-stint model splits the rating
into **ORAPM** (offense) and **DRAPM** (defense), with RAPM = ORAPM + DRAPM.

## Quick start

```bash
# R deps: wehoop, arrow, dplyr, readr      Python deps:
pip install -r requirements.txt

make season YEAR=2026        # 2026 == the 2025-26 season (wehoop convention)
open site/index.html
```

Re-run any prior season with `make season YEAR=2025` etc. Every artifact is
partitioned by season. Each page on the site has a season dropdown in the header
(leaderboard, team index, every team page, Stanford, about), and clicking a player
opens a modal with her season-by-season rating history (`site/data/history.js`).
Root pages (`index.html`, ...) are aliases for the latest season; every season also
has suffixed pages (`index-2025.html`, `teams/2025/<id>.html`, ...).

### Season availability

Lineup RAPM needs substitution events, and ESPN's WBB play-by-play only began
recording them mid-2024-25 (from Feb 12, 2025):

- **2026** (2025–26): full season — 5,943 of 6,011 games pass QA.
- **2025** (2024–25): partial — 1,362 games from Feb 12 to Apr 6, 2025 (late season +
  postseason). Pages carry a coverage warning.
- **2024 and earlier**: no substitution data exists; the pipeline refuses to produce
  ratings (games with <10 sub events are excluded outright, and a season with zero
  usable games aborts with an explanation).

## Pipeline

| Stage | Code | What it does |
|---|---|---|
| 1 Fetch | `R/fetch_data.R` | wehoop pbp / player box / team box / D1 teams → `data/raw/*.parquet` (cached) |
| 2 Lineups | `src/wbbrapm/lineups.py` | Reconstructs the 10 on-court players for every event from directional sub text ("X subbing in/out"), starters, and stat-forced corrections |
| 3 Stints | `src/wbbrapm/stints.py` | Constant-lineup 5v5 stints with points + possession proxy (FGA − OREB + TO + 0.44·FTA) |
| 4 RAPM | `src/wbbrapm/rapm.py` | Sparse ridge: overall margin model + O/D split; λ via GroupKFold CV; non-D1 opponents collapsed to one column |
| 5 Validate | `src/wbbrapm/validate.py` | Reconstructed minutes vs box minutes (bad games excluded); team-level RAPM vs net margin / win% |
| 6 Site | `src/wbbrapm/site.py` | Static HTML: leaderboard w/ search-sort-filter + player modal, team pages w/ top lineups, Stanford page, about/validation page, CSV export |

## Validation (2025–26)

- Lineup minute reconciliation MAE: **0.36 min/player** (68 of 6,011 games excluded at the 1.5-min threshold)
- Stint points reconcile exactly to final scores; stint seconds partition the game clock
- corr(team minute-weighted RAPM, team net margin) = **0.90**; vs win% = **0.84**
- Low-minute players (<100 min) max |RAPM| ≈ 5 vs ≈ 20 for high-minute stars — the ridge shrinkage working as intended

`make test` runs the unit suite (fixture-game lineup/stint reconstruction, star
recovery, and shrinkage behavior).

## Layout

```
R/fetch_data.R          data pull (R/wehoop)
src/wbbrapm/            python package: config, io, lineups, stints, rapm, validate, site
templates/              jinja2 templates for the static site
data/raw|processed/     season-partitioned caches and outputs (gitignored)
site/                   generated static site (host on GitHub Pages or open locally)
run.py                  one-season orchestrator
```

## Caveats

- ESPN pbp quality varies; games failing minute reconciliation are excluded and counted
  in `data/processed/lineup_qa_<season>.json`.
- Players who almost always share the floor are statistically hard to separate; ridge
  keeps them stable, but read their O/D splits with care.
- Possessions are estimated with the standard box proxy, not tracked possessions.
