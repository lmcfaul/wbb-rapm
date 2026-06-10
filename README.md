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
recording them mid-2024-25 (from Feb 12, 2025). **2025–26 is the first fully covered
season and the only one published** (5,943 of 6,011 games pass QA).

Alternative sources were investigated for earlier seasons and none pan out:
stats.ncaa.org has subs but blocks programmatic access (403); NCAA.com's data API
(data.ncaa.com) does not retain per-game play-by-play; NCAA LiveStats / Genius Sports
is credentialed-commercial; Her Hoop Stats / CBB Analytics have no pbp export;
Sports-Reference has no WBB pbp. 2024-25 (late-season-only ESPN coverage) was built
and then removed by request.

Guards: games with <10 sub events are excluded outright, and a season with zero
usable games aborts with an explanation. Future seasons: `make season YEAR=2027`.

## Pipeline

| Stage | Code | What it does |
|---|---|---|
| 1 Fetch | `R/fetch_data.R` | wehoop pbp / player box / team box / D1 teams → `data/raw/*.parquet` (cached) |
| 2 Lineups | `src/wbbrapm/lineups.py` | Reconstructs the 10 on-court players for every event from directional sub text ("X subbing in/out"), starters, and stat-forced corrections |
| 3 Stints | `src/wbbrapm/stints.py` | Constant-lineup 5v5 stints with points + possession proxy (FGA − OREB + TO + 0.44·FTA) |
| 4 RAPM | `src/wbbrapm/rapm.py` | Sparse ridge: overall margin model + O/D split; λ via GroupKFold CV; non-D1 opponents collapsed to one column |
| 5 Validate | `src/wbbrapm/validate.py` | Reconstructed minutes vs box minutes (bad games excluded); team-level RAPM vs net margin / win% |
| 6 Site | `src/wbbrapm/site.py` | Static HTML: leaderboard w/ search-sort-filter, player modal (rating breakdown, season-phase trend chart, multi-year history) on leaderboard *and* team pages, lineup explorer (2- to 5-player combos), team pages w/ top lineups, Stanford page, about/validation page, CSV export |

The player modal's **season trend** splits the season's games into three date-based
thirds and refits the O/D ridge model on each (same λ), showing Early/Mid/Late RAPM
with an inline chart; phases under 30 minutes are greyed as thin samples. The
**lineup explorer** aggregates every 2-, 3-, 4-, and 5-player same-team combination's
raw on-court numbers (possession floors of 100/75/50/50 by size) with team, size,
player, and possession filters.

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
