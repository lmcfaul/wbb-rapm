# Methodology — How the WBB RAPM Ratings Are Calculated

This document explains, in detail, how every number on the site is produced: where
the data comes from, which games are thrown out and why, how the regression works,
what a ridge regression actually *is*, and how to read each rating. It is meant to
be self-contained — you should not need to read the source to understand the
results, though every section points at the file that implements it.

If you just want the one-sentence version: **each player's rating (RAPM) is her
estimated effect on the score margin per 100 possessions, after statistically
removing the quality of her teammates and her opponents, with a penalty that pulls
players with little playing time toward zero so small samples can't dominate.**

---

## Table of contents

1. [The rating in one picture](#1-the-rating-in-one-picture)
2. [Stage 1 — Data source](#2-stage-1--data-source)
3. [Stage 2 — Reconstructing who was on the court](#3-stage-2--reconstructing-who-was-on-the-court)
4. [Stage 3 — Stints and possessions](#4-stage-3--stints-and-possessions)
5. [Stage 4 — The regression (RAPM, ORAPM, DRAPM)](#5-stage-4--the-regression-rapm-orapm-drapm)
6. [What a ridge regression *is*](#6-what-a-ridge-regression-is)
7. [Choosing the penalty λ by cross-validation](#7-choosing-the-penalty-λ-by-cross-validation)
8. [The three model families (ridge / lasso / elastic net)](#8-the-three-model-families-ridge--lasso--elastic-net)
9. [The rating breakdown (raw / teammate / opponent)](#9-the-rating-breakdown-raw--teammate--opponent)
10. [Season-phase trends](#10-season-phase-trends)
11. [Lineup explorer](#11-lineup-explorer)
12. [Which games are excluded, and why](#12-which-games-are-excluded-and-why)
13. [Validation — how we know it works](#13-validation--how-we-know-it-works)
14. [Caveats and known limitations](#14-caveats-and-known-limitations)
15. [Glossary](#15-glossary)

---

## 1. The rating in one picture

```
ESPN play-by-play
   │  (Stage 2) parse "X subbing in/out", force stat-recorders on court
   ▼
10 on-court players known at every event
   │  (Stage 3) cut into constant-lineup "stints"; count points + possessions
   ▼
one regression row per stint:  +1 for each home player, −1 for each away player
   │  (Stage 4) ridge regression with a penalty λ chosen by cross-validation
   ▼
one coefficient per player = RAPM (points per 100 possessions vs. an average D1 player)
```

Everything else on the site — the offense/defense split, the rating breakdown, the
season-half trend, the lineup explorer — is a variation on this same pipeline.

---

## 2. Stage 1 — Data source

**Code:** `R/fetch_data.R` (uses the R package [`wehoop`](https://wehoop.sportsdataverse.org/))

All raw data comes from **ESPN's women's college basketball feeds**, pulled through
`wehoop`. For each season we cache four tables to `data/raw/` (Parquet, or `.csv.gz`
if the `arrow` package is unavailable):

| Table | What it is | Used for |
|---|---|---|
| `pbp` | Play-by-play: every event (shots, rebounds, fouls, **substitutions**, timeouts) with a clock and the athlete(s) involved | Lineup reconstruction, stints |
| `player_box` | Per-player box score per game: minutes, starter flag, DNP flag, name, team, headshot | Starters, minute reconciliation, identity metadata |
| `team_box` | Per-team box score per game: final score, win flag | Validation (team RAPM vs. real results) |
| `teams` | ESPN's Division I team directory | Flagging which players/opponents are D1 |

Season numbering follows the `wehoop` convention: **`2026` means the 2025–26
season.** Data is fetched once and cached; re-running the pipeline reuses the cache
unless you pass `--refresh`.

### Why only 2025–26 is published

Lineup-based RAPM is impossible without **substitution events** in the play-by-play.
ESPN's WBB feed only began recording substitutions mid-way through 2024–25 (from
**Feb 12, 2025**). 2025–26 is therefore the first fully covered season and the only
one published. Earlier-season alternatives were investigated and none work
programmatically (stats.ncaa.org has subs but blocks automated access with HTTP 403;
data.ncaa.com doesn't retain per-game play-by-play; Genius Sports / NCAA LiveStats is
credentialed-commercial; Her Hoop Stats, CBB Analytics, and Sports-Reference have no
WBB play-by-play export). A partial 2024–25 build was made and then removed by
request.

---

## 3. Stage 2 — Reconstructing who was on the court

**Code:** `src/wbbrapm/lineups.py`

The play-by-play does **not** carry a "current lineup" field. We have to rebuild the
five-on-five state ourselves by walking each game's events in order. The algorithm:

1. **Start from the box-score starters.** Each team's five starters (from
   `player_box.starter`) seed the on-court set. If a team doesn't have exactly five
   identifiable starters, the game is marked unusable (`reason="bad starters"`).

2. **Apply substitutions from text.** ESPN's sub events are *directional*: the text
   reads `"<Player> subbing in for <TEAM>"` or `"<Player> subbing out for <TEAM>"`.
   We parse the direction and add/remove that player. If the sub row is missing its
   `athlete_id`, we recover it by matching the leading player name against the team
   roster.

3. **Safety net A — force stat-recorders on court.** You cannot shoot, rebound, foul,
   steal, assist, or turn the ball over from the bench. So any player who records a
   *non*-substitution stat event is forced into the on-court set if the text-parsing
   missed her entrance. (Substitutions, period-end, game-end, and timeouts are
   excluded from this rule — see `NON_PLAYER_TYPES`.)

4. **Safety net B — trim overfull lineups.** If errors ever push a team above five
   on the floor, we drop the "stalest" player (the one whose last substitution/event
   was longest ago), since a missed "subbing out" is the most likely cause.

5. **Accrue playing time.** Between consecutive events we add the elapsed seconds to
   everyone currently on the floor. This reconstructed playing time is later checked
   against the official box-score minutes (Stage 5) — the central quality gate.

### The game clock

Elapsed game-seconds are computed from `start_quarter_seconds_remaining` plus the
period offset. WBB plays **four 10-minute quarters (600s each)** and **5-minute
overtimes (300s each)**, encoded in `period_start_seconds()` / `period_length()`.
Clock noise that would make time appear to run backwards is clamped so elapsed time
is monotonic.

The result is a `GameLineups` object: for every event row, the exact set of 5 home
and 5 away players, plus per-player reconstructed seconds and QA counters
(toggle errors, forced-ons, overfulls).

---

## 4. Stage 3 — Stints and possessions

**Code:** `src/wbbrapm/stints.py`

A **stint** is a maximal run of consecutive events *within a single period* during
which **all ten on-court players stay the same**. A new stint begins whenever any
player is substituted or a new period starts. Stints are the atomic unit of the
regression: each one is a little controlled experiment — "these five vs. those five,
for this many possessions, and here's what happened to the score."

For each stint we record:

- **Both lineups** (the five home, the five away athlete IDs).
- **Elapsed seconds** — boundaries are chained so the stints exactly partition the
  full game clock with no gaps or double-counting. (Brief 4- or 6-player transitional
  states during paired sub events are attributed to the next settled 5-on-5 lineup.)
- **Points by side** — summed from `scoring_play` / `score_value`. These reconcile
  exactly to the final score.
- **Possessions by side** — estimated with the standard box-score proxy:

  ```
  possessions = FGA − OREB + TO + 0.44 × FTA
  ```

  where FGA = field-goal attempts, OREB = offensive rebounds, TO = turnovers, FTA =
  free-throw attempts. The `0.44` (in `config.FTA_POSSESSION_WEIGHT`) is the
  long-standing empirical estimate of how many true possessions a free-throw
  attempt represents (most trips to the line are two shots, but and-ones, technicals,
  and one-and-ones make the average trip cost less than two FTAs). Offensive rebounds
  are subtracted because they extend the *same* possession rather than starting a new
  one. The result is floored at zero.

These are **estimated** possessions, not tracked ones — there is no possession field
in the feed. This is the same proxy used across public basketball analytics.

---

## 5. Stage 4 — The regression (RAPM, ORAPM, DRAPM)

**Code:** `src/wbbrapm/rapm.py`

### The player index

Every D1 player who appears in any usable stint gets her own column. **All non-D1
players are collapsed into a single shared column** (`NON_D1`) — we don't rate them,
but we still need to account for their presence so a D1 player isn't credited or
penalized for facing/partnering a non-D1 opponent. This matters for early-season
games against lower-division opponents.

### The margin model (overall RAPM)

One regression **row per stint**. The columns are the players; the entries are:

- **+1** for each of the five home players,
- **−1** for each of the five away players,
- everything else 0.

The **target** `y` is the home margin per 100 possessions:

```
y = 100 × (home_pts − away_pts) / possessions
```

where `possessions` is the average of the two sides' possession counts. Each row is
**weighted** by that possession count, so a 20-possession stint counts ten times as
much as a 2-possession stint.

Fit a (ridge) regression of `y` on this matrix. The coefficient on a player's column
is her **RAPM**: her marginal effect on margin per 100 possessions, holding the other
nine players on the floor fixed. Because every teammate and opponent is literally a
co-variate in the same row, the coefficient is **automatically adjusted for who she
played with and against** — that's the "adjusted" in *regularized adjusted
plus-minus*. The fitted **intercept** is the home-court advantage.

> **Sign convention.** A +1 column is the home offense *and* defense rolled together;
> the margin model doesn't separate them. Positive RAPM = the team outscores
> opponents by more (per 100 poss.) when she's on the floor.

### The offense/defense split (ORAPM + DRAPM)

To split a player's impact into offense and defense, we build a second system with
**two rows per stint** — one for each side's offensive trip:

- Columns are laid out as `[ offense block | defense block | home-offense indicator ]`,
  so every player has *two* coefficients: one for when she's on offense, one for when
  she's on defense.
- For the row representing side A's offense: A's five players get **+1 in the offense
  block**, B's five players get **+1 in the defense block**.
- The **target** is offensive points per 100 possessions: `y = 100 × pts / poss`.
- The extra `home-offense` indicator column absorbs home-court advantage on offense.

From the fitted coefficients:

- **ORAPM** = the offense-block coefficient — points per 100 the player **adds** on
  offense. Higher is better.
- **DRAPM** = the **negated** defense-block coefficient. The raw defense coefficient
  is points *allowed* per 100, so lower is better; we negate it so that, like every
  other rating, **higher is better**.
- **RAPM (from the split)** = ORAPM + DRAPM. This is the number the leaderboard sorts
  on by default.

> Note there are two RAPM-like numbers: `rapm_margin` from the one-row margin model,
> and `rapm = orapm + drapm` from the two-row split. They're highly correlated; the
> site displays the O/D split version (and its components) because it's more
> informative.

---

## 6. What a ridge regression *is*

This is the heart of the method, so it's worth being explicit.

### Ordinary least squares, and why it fails here

A normal ("ordinary least squares", OLS) regression finds the coefficients that
minimize the **sum of squared errors** between the predicted and actual margins:

```
minimize   Σ  wᵢ · (yᵢ − predictionᵢ)²
         over all coefficients
```

With ~6,000 games, hundreds of thousands of stints, and ~5,000+ player columns, plain
OLS has two fatal problems:

1. **Collinearity / non-identifiability.** Players who almost always share the floor
   (a starter and her usual backcourt partner; two players who only ever sub for each
   other) can't be told apart by the data. OLS responds by handing one a wildly
   positive coefficient and the other a wildly negative one that cancel out — numbers
   that fit the data but are meaningless individually.

2. **Overfitting small samples.** A walk-on who played 12 garbage-time minutes during
   a +30 blowout would get an enormous positive rating, because OLS will do anything
   to shave error off those few rows.

### The ridge fix: penalize large coefficients

Ridge regression adds a **penalty** proportional to the sum of *squared* coefficients:

```
minimize   Σ wᵢ · (yᵢ − predictionᵢ)²   +   λ · Σ (coefficientⱼ)²
           └──────── fit the data ───────┘     └─ keep coefficients small ─┘
```

- The first term is the usual "fit the data" goal.
- The second term, the **L2 penalty**, charges the model for using large
  coefficients. λ (lambda) sets the price.

The consequence is **shrinkage toward zero**:

- A player with **lots of evidence** (many possessions, in many different lineup
  combinations) earns a large coefficient because moving it really does reduce error
  enough to be worth the penalty.
- A player with **thin evidence** can't justify a big coefficient against the penalty,
  so she's pulled toward 0 — i.e., toward "league average." **This is exactly why
  low-minute players can't top the leaderboard.** It's a feature, not a bug.

For collinear pairs, the squared penalty is minimized by **spreading** the effect
evenly rather than letting one explode positive and the other negative. So ridge
gives stable, sensible numbers where OLS gives nonsense.

### Interpreting the units

Because every column is a ±1 player indicator and `y` is per-100-possessions margin,
each coefficient is **points per 100 possessions relative to an average Division I
player**. RAPM of +5 means: replace an average D1 player with this player and, all
else equal, the team's per-100 margin improves by about 5 points. The model is
"centered" — the average rated player is ~0 by construction.

### What ridge is *not*

- It is **not** causal proof; it's an association adjusted for who else was on the
  floor. Coaches don't deploy lineups randomly, so residual confounding exists.
- It does **not** invent playing time; a player only gets separated from her
  teammates to the extent the schedule actually mixed up the lineups.

---

## 7. Choosing the penalty λ by cross-validation

**Code:** `rapm.cv_lambda`, `config.LAMBDA_GRID`

λ controls the whole bias/variance trade-off: too small and you overfit (back toward
OLS nonsense); too large and you over-shrink (everyone looks average). We pick it by
**game-grouped k-fold cross-validation**:

1. Candidate values: `λ ∈ {250, 500, 1000, 2000, 4000, 8000}` (`LAMBDA_GRID`).
2. Split the data into **5 folds**, grouped by `game_id` using `GroupKFold`. **All
   stints from one game stay together** in the same fold — this prevents leakage,
   because stints within a game share lineups and would otherwise let the model "peek."
3. For each fold: train on the other four, predict the held-out fold, and record the
   **possession-weighted mean squared error**.
4. Sum each λ's error across folds; **pick the λ with the lowest total error.**

The margin model and the O/D model get their own λ (different response scales). You
can override CV with `--lam-margin` / `--lam-od` on the command line. The chosen λ
values are written to `data/processed/validation_<season>.json`.

---

## 8. The three model families (ridge / lasso / elastic net)

**Code:** `config.MODEL_SPECS`, `rapm.fit_models_multi`

The leaderboard has a **model toggle**. All three families are fit on the **exact same
design matrices**; they differ only in *how* they penalize coefficients:

| Model | Penalty | Behavior | Settings |
|---|---|---|---|
| **Ridge** | L2: λ·Σβ² | Smooth shrinkage; **every** player gets a (small) nonzero rating. The canonical RAPM. | λ from CV |
| **Lasso** | L1: α·Σ\|β\| | Drives most coefficients to **exactly zero**; only clearly-supported impacts survive. Sparse. | α = 0.02 (margin), 0.01 (O/D) |
| **Elastic Net** | blend of L1 + L2 | A middle ground between lasso's sparsity and ridge's dense shrinkage. | α = 0.05 / 0.02, l1_ratio = 0.5 |

The key difference is **L1 vs. L2**: squaring (L2/ridge) penalizes large coefficients
hard but never quite forces them to zero, so everyone stays in. Absolute value
(L1/lasso) has a constant pull that **clips small coefficients to exactly zero**,
producing a short list of "players the data is confident about" and leaving everyone
else unrated. Elastic net mixes the two (`l1_ratio = 0.5` = half and half). The O/D
system uses a lighter α than the margin system because its response is on a different
scale; both were calibrated on 2025–26.

Each model gets its own ratings file, CSV, JS data, and rating-breakdown. **All three
correlate ≈ 0.90 with team net margin** (see the About page). **Team pages, season
trends, and multi-year history always use ridge** — it's the canonical, everyone-rated
view.

---

## 9. The rating breakdown (raw / teammate / opponent)

**Code:** `rapm.decompose_ratings`

The player modal shows *where* a rating comes from by decomposing ORAPM and DRAPM into
three additive pieces:

```
ORAPM ≈ raw_off + tm_off + opp_off
DRAPM ≈ raw_def + tm_def + opp_def
```

- **raw_\*** — the player's *unadjusted* on-court production vs. league average. This
  is plain on-court/off-court plus-minus: points scored (or allowed) per 100 while she
  was on the floor, minus the league average.
- **tm_\*** — the teammate adjustment: **minus** the possession-weighted quality of
  the teammates she shared the floor with. If she played mostly alongside great
  players, raw numbers overstate her, so this term subtracts that boost.
- **opp_\*** — the opponent adjustment: **plus** the quality of the opposition she
  faced. Facing tougher opponents makes her raw numbers look worse than they are, so
  this term adds it back.

The league average (`100 × total_pts / total_poss` across all stints) is the zero
point. This decomposition is **approximate, not an exact identity** — ridge shrinkage
and home-court advantage are not redistributed into the three buckets — but it's a
faithful guide to whether a rating is driven by raw play, easy company, or schedule.

---

## 10. Season-phase trends

**Code:** `rapm.fit_phase_ratings`, `config.N_PHASES`

The modal's trend chart splits the season into **two date-based halves** (`N_PHASES =
2`): games are sorted by date and divided into two equal-count groups. The O/D model
is **refit on each half separately**, reusing the **same λ** as the full-season model
so the halves are comparable.

Each half therefore has a fraction of the data and shrinks harder (noisier), which is
expected. Phases where a player logged **under 30 minutes** are greyed out in the UI as
thin samples. This shows whether a player trended up or down across the season.

---

## 11. Lineup explorer

**Code:** `src/wbbrapm/site.py` (consumed by `templates/lineups.html.j2`)

Separate from the regression, the explorer aggregates the **raw on-court numbers** for
every same-team combination of 2, 3, 4, and 5 players: their combined points,
possessions, and net rating while *all* of them were on the floor together. These are
**not** RAPM/adjusted — they're descriptive lineup stats. To keep the noise down, each
combination size has a minimum-possession floor:

| Combo size | Minimum possessions |
|---|---|
| 2 players | 100 |
| 3 players | 75 |
| 4 players | 50 |
| 5 players | 50 |

Filters let you slice by team, combo size, specific player, and possession count.

---

## 12. Which games are excluded, and why

**Code:** `config.MIN_SUBS_PER_GAME`, `config.MINUTE_MAE_THRESHOLD`,
`validate.game_minute_mae`, `run.py`

A game must clear **two** gates to enter the model. There are also two structural
reasons a game can be dropped before that. In order:

### Gate 0 — Bad starters (structural)

If a game doesn't have exactly five identifiable starters per team in the box score,
its lineups can't be seeded and it's dropped (`reason="bad starters"`,
`qa_counts.bad_starters`).

### Gate 1 — Too few substitution events

A game with **fewer than 10 substitution events** (`MIN_SUBS_PER_GAME = 10`) cannot be
reconstructed at all — this is how the entirely-sub-less pre-Feb-2025 feeds are caught.
Such games are excluded outright, regardless of anything else (`qa_counts.no_subs`).

### Gate 2 — Minute-reconciliation error (the main quality gate)

For every surviving game we compare the **reconstructed** on-court minutes (Stage 2)
against the **official box-score minutes**, per player, and take the **mean absolute
error (MAE)** in minutes per player. A game is excluded if:

```
mean | reconstructed_minutes − box_minutes |  >  1.5 minutes per player
```

(`MINUTE_MAE_THRESHOLD = 1.5`). The logic: if our reconstructed lineups disagree with
the box score by more than ~1.5 minutes per player on average, the play-by-play for
that game is too garbled to trust, so we leave it out rather than feed bad rows to the
regression.

### What this looked like in 2025–26

From `data/processed/lineup_qa_2026.json`:

- **6,011 games** total in the season feed.
- Mean reconciliation MAE: **0.36 min/player**; median 0.29; 95th percentile 0.71 —
  i.e., the typical game reconstructs almost perfectly.
- **68 games excluded** at the 1.5-minute threshold (plus any caught by Gates 0/1).
- **≈ 5,943 games pass QA** and feed the model.

The exact excluded `game_id`s are listed in that JSON file. If a *whole season* has no
usable game (e.g., a feed with no sub data), the pipeline aborts with an explanation
rather than producing garbage. If more than 20% of games are excluded, it still runs
but prints a partial-coverage warning and records the usable date range.

> **Not an exclusion, but related:** players under **200 minutes** get a "low sample"
> flag (`LOW_MINUTES_FLAG`) but stay in the model — ridge already shrinks them. The
> site's default leaderboard view *hides* players under **100 minutes**
> (`MIN_MINUTES_DEFAULT_FILTER`); you can show them with the minutes filter.

---

## 13. Validation — how we know it works

**Code:** `src/wbbrapm/validate.py`

Three independent checks, written to `data/processed/validation_<season>.json` and
summarized on the site's About page:

1. **Minute reconciliation** (also the exclusion gate, §12): MAE **0.36 min/player**
   across the season. This confirms the lineup reconstruction matches reality.

2. **Points and clock partition exactly.** Stint points sum to the real final scores,
   and stint seconds partition the game clock with no gaps or overlaps — a structural
   integrity check.

3. **Team aggregation sanity check.** Roll each team's player RAPMs up by minutes
   (minute-weighted average) and correlate against the team's actual results:
   - **corr(team RAPM, net margin) = 0.90**
   - **corr(team RAPM, win %) = 0.84**

   All three model families (ridge/lasso/enet) land near 0.90 on net margin. A method
   that produced noise would not align with real team outcomes this tightly.

4. **Shrinkage behaves.** Low-minute players (<100 min) top out around |RAPM| ≈ 5,
   while high-minute stars reach ≈ 20 — direct evidence the ridge penalty is keeping
   thin samples off the extremes.

`make test` runs a unit suite on fixture games covering lineup/stint reconstruction,
star recovery, and shrinkage behavior.

---

## 14. Caveats and known limitations

- **Estimated possessions.** The `FGA − OREB + TO + 0.44·FTA` proxy is standard but
  not exact; there's no tracked-possession field in the feed.
- **ESPN feed quality varies.** Games failing reconciliation are excluded (§12), but
  the surviving games still carry some text-parsing noise that the safety nets and
  ridge shrinkage absorb rather than eliminate.
- **Collinear teammates.** Players who almost always share the floor are genuinely
  hard to separate. Ridge keeps their numbers stable, but read individual O/D splits
  for such players with extra caution.
- **Not causal.** Lineups aren't randomized; RAPM is an adjusted association, not a
  controlled experiment. Usage, role, and coaching decisions are not modeled.
- **Single-season noise.** One season is a limited sample; mid-season trends (§10) are
  noisier still. Treat small RAPM differences between players as ties.
- **Non-D1 opponents are pooled.** All sub-Division-I players share one column, so
  performance against very weak opponents is only coarsely adjusted.

---

## 15. Glossary

- **RAPM** — Regularized Adjusted Plus-Minus. A player's effect on score margin per
  100 possessions, adjusted for teammates and opponents, with a shrinkage penalty.
- **ORAPM / DRAPM** — the offensive and defensive halves of RAPM; RAPM = ORAPM + DRAPM.
  Both are scaled so higher is better.
- **Stint** — a stretch of play where all ten on-court players stay constant; the unit
  of the regression.
- **Possession** — one team's trip down the floor; here estimated as
  `FGA − OREB + TO + 0.44·FTA`.
- **Ridge regression / L2 penalty** — least-squares with a penalty on the sum of
  *squared* coefficients; shrinks estimates smoothly toward zero, never exactly to it.
- **Lasso / L1 penalty** — penalty on the sum of *absolute* coefficients; forces small
  estimates to exactly zero (sparse).
- **Elastic net** — a weighted blend of L1 and L2 penalties.
- **λ (lambda) / α (alpha)** — the strength of the penalty. Larger = more shrinkage.
- **Cross-validation (GroupKFold)** — splitting games into folds (keeping each game
  whole) to choose λ by out-of-sample error.
- **MAE** — mean absolute error; here, average per-player gap between reconstructed and
  box-score minutes, used to exclude bad games (threshold 1.5 min).
- **HCA** — home-court advantage; the regression intercept (margin model) or the
  home-offense indicator coefficient (O/D model).
- **Net margin / net rating** — points scored minus points allowed (per game, or per
  100 possessions).

---

*Every figure here is regenerated by `python run.py --season 2026` (`make season
YEAR=2026`). The authoritative numbers live in `data/processed/lineup_qa_2026.json`
and `data/processed/validation_2026.json`.*
