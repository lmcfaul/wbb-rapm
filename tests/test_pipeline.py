"""Unit tests on a synthetic fixture game + model behavior checks."""
import numpy as np
import pandas as pd
import pytest

from wbbrapm import lineups, rapm, stints

HOME, AWAY = 1, 2
H = [101, 102, 103, 104, 105]
A = [201, 202, 203, 204, 205]
A6 = 206


def _ev(period, rem, type_text, text="", team=None, a1=None, scoring=False, value=0):
    return {
        "game_id": 9999, "period_number": period, "start_quarter_seconds_remaining": rem,
        "type_text": type_text, "text": text, "team_id": team,
        "athlete_id_1": a1, "athlete_id_2": None,
        "scoring_play": scoring, "score_value": value, "shooting_play": type_text == "JumpShot",
        "home_team_id": HOME, "away_team_id": AWAY,
        "home_score": 0, "away_score": 0, "game_play_number": 0,
    }


@pytest.fixture
def fixture_game():
    events = [
        _ev(1, 550, "JumpShot", "h1 makes jumper", HOME, H[0], scoring=True, value=2),
        _ev(1, 300, "Substitution", "A6 subbing in for Away", AWAY, A6),
        _ev(1, 300, "Substitution", "A1 subbing out for Away", AWAY, A[0]),
        _ev(1, 200, "LayUpShot", "a6 makes layup", AWAY, A6, scoring=True, value=2),
        _ev(1, 0, "End Period"),
        _ev(2, 400, "JumpShot", "h2 misses jumper", HOME, H[1]),
        _ev(2, 395, "Defensive Rebound", "a2 rebound", AWAY, A[1]),
        _ev(2, 0, "End Game"),
    ]
    pbp = pd.DataFrame(events)
    pbp["game_play_number"] = range(1, len(pbp) + 1)
    starters = {HOME: set(H), AWAY: set(A)}
    roster = {a: HOME for a in H} | {a: AWAY for a in A} | {A6: AWAY}
    return pbp, starters, roster


def test_lineup_reconstruction(fixture_game):
    pbp, starters, roster = fixture_game
    gl = lineups.reconstruct_game(pbp, starters, roster, {})
    assert gl.ok
    # both teams have exactly 5 on court at every settled event
    for i in (0, 3, 5, 6, 7):
        assert len(gl.home_lineup[i]) == 5
        assert len(gl.away_lineup[i]) == 5
    # the sub swapped A1 for A6
    assert A6 in gl.away_lineup[3] and A[0] not in gl.away_lineup[3]
    # seconds: A1 played 300s of Q1, A6 the remaining 300 + all 600 of Q2
    assert gl.player_seconds[A[0]] == pytest.approx(300)
    assert gl.player_seconds[A6] == pytest.approx(900)
    assert gl.player_seconds[H[0]] == pytest.approx(1200)


def test_forced_on_court_corrects_missed_sub(fixture_game):
    pbp, starters, roster = fixture_game
    # drop the "subbing in" row: A6's layup must force her on court anyway
    pbp = pbp.drop(index=1).reset_index(drop=True)
    gl = lineups.reconstruct_game(pbp, starters, roster, {})
    assert gl.ok
    assert A6 in gl.away_lineup[2]
    assert gl.n_forced_on == 1


def test_stints(fixture_game):
    pbp, starters, roster = fixture_game
    gl = lineups.reconstruct_game(pbp, starters, roster, {})
    sd = stints.build_stints(pbp, gl)
    # Q1 pre-sub, Q1 post-sub, Q2
    assert len(sd) == 3
    assert sd.seconds.tolist() == pytest.approx([300, 300, 600])
    assert sd.seconds.sum() == pytest.approx(gl.elapsed[-1])
    # all stints are 5v5
    assert all(len(l) == 5 for l in sd.home_lineup) and all(len(l) == 5 for l in sd.away_lineup)
    # points land in the right stints and reconcile to the final score
    assert sd.home_pts.tolist() == [2, 0, 0]
    assert sd.away_pts.tolist() == [0, 2, 0]
    # possession proxy: one FGA each for home stint 1 / away stint 2 / home stint 3
    assert sd.home_poss.tolist() == pytest.approx([1, 0, 1])
    assert sd.away_poss.tolist() == pytest.approx([0, 1, 0])


def _synthetic_stints(n_games=60, seed=0):
    """Two fixed lineups per team; team 1's star (id 101) drives +10/100."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        for s in range(8):
            star_on = s % 2 == 0
            home = [101 if star_on else 106, 102, 103, 104, 105]
            away = [201, 202, 203, 204, 205]
            poss = 10.0
            margin = (10.0 if star_on else 0.0) / 100 * poss + rng.normal(0, 1)
            rows.append({
                "game_id": g, "home_lineup": home, "away_lineup": away,
                "home_pts": max(poss + margin, 0), "away_pts": poss,
                "home_poss": poss, "away_poss": poss, "seconds": 240.0,
            })
    return pd.DataFrame(rows)


def test_rapm_recovers_star_and_shrinks():
    sd = _synthetic_stints()
    athlete_team = {a: 1 for a in (101, 102, 103, 104, 105, 106)} | {a: 2 for a in (201, 202, 203, 204, 205)}
    col_of, non_d1, players = rapm.build_player_index(sd, athlete_team, {1, 2})
    coefs = rapm.fit_models(sd, col_of, non_d1, lam_margin=50.0, lam_od=50.0, verbose=False)
    by_id = dict(zip(players, coefs["rapm_margin"]))
    # the star rates above every teammate and every opponent
    assert by_id[101] == max(by_id.values())
    assert by_id[101] > by_id[106] + 2
    # O/D split sums to roughly the same story
    od = dict(zip(players, coefs["orapm"] + coefs["drapm"]))
    assert od[101] == max(od.values())


def test_decomposition_semantics():
    sd = _synthetic_stints()
    athlete_team = {a: 1 for a in (101, 102, 103, 104, 105, 106)} | {a: 2 for a in (201, 202, 203, 204, 205)}
    col_of, non_d1, players = rapm.build_player_index(sd, athlete_team, {1, 2})
    coefs = rapm.fit_models(sd, col_of, non_d1, lam_margin=50.0, lam_od=50.0, verbose=False)
    dec = rapm.decompose_ratings(sd, col_of, non_d1, coefs)
    by = {k: dict(zip(players, dec[k])) for k in rapm.DECOMP_COLS}
    # player 102 is on court for every stint (raw_net ~ +5, half with the star);
    # the teammate adjustment must strip the star's contribution (negative;
    # collinearity in this fixture mutes how much credit the model gives the star)
    assert by["raw_net"][102] == pytest.approx(5.0, abs=1.5)
    assert by["tm_net"][102] < -1.0
    # raw + teammate adj + competition adj approximately recovers RAPM
    rapm_od = dict(zip(players, coefs["orapm"] + coefs["drapm"]))
    for a in (101, 102, 201):
        approx = by["raw_net"][a] + by["tm_net"][a] + by["opp_net"][a]
        assert abs(approx - rapm_od[a]) < 2.5, (a, approx, rapm_od[a])


def test_phase_ratings():
    sd = _synthetic_stints()
    athlete_team = {a: 1 for a in (101, 102, 103, 104, 105, 106)} | {a: 2 for a in (201, 202, 203, 204, 205)}
    col_of, non_d1, players = rapm.build_player_index(sd, athlete_team, {1, 2})
    game_dates = {g: f"2026-01-{g + 1:02d}" for g in range(60)}
    ph = rapm.fit_phase_ratings(sd, col_of, non_d1, lam_od=50.0, game_dates=game_dates)
    assert sorted(ph["phase"].unique()) == [0, 1]  # season splits into halves
    star = ph[ph["athlete_id"] == 101].sort_values("phase")
    # the star's effect is constant across the season: positive in every phase
    assert (star["rapm"] > 1.0).all()
    # phase date labels follow the split
    assert star.iloc[0]["start"] == "2026-01-01" and star.iloc[1]["end"] == "2026-01-60"
    # minutes divide roughly evenly (star plays half of each game's stints)
    assert star["minutes"].std() < star["minutes"].mean() * 0.2
    # explicit phase count still honored
    ph3 = rapm.fit_phase_ratings(sd, col_of, non_d1, lam_od=50.0,
                                 game_dates=game_dates, n_phases=3)
    assert sorted(ph3["phase"].unique()) == [0, 1, 2]


def test_ridge_shrinks_low_sample_players():
    sd = _synthetic_stints()
    # give player 999 one lucky 1-stint cameo (+20 margin in 10 possessions,
    # i.e. a raw on-court margin of +200 per 100)
    cameo = sd.iloc[[0]].copy()
    cameo["home_lineup"] = [[999, 102, 103, 104, 105]]
    cameo["home_pts"] = 30.0
    sd = pd.concat([sd, cameo], ignore_index=True)
    athlete_team = {a: 1 for a in (101, 102, 103, 104, 105, 106, 999)} | {a: 2 for a in (201, 202, 203, 204, 205)}
    col_of, non_d1, players = rapm.build_player_index(sd, athlete_team, {1, 2})

    def coef_999(lam):
        coefs = rapm.fit_models(sd, col_of, non_d1, lam_margin=lam, lam_od=lam, verbose=False)
        return dict(zip(players, coefs["rapm_margin"]))[999]

    light, heavy = coef_999(50.0), coef_999(5000.0)
    # the penalty pulls the thin-sample cameo far below the raw +200, and a
    # heavier penalty shrinks it further toward 0
    assert abs(light) < 100
    assert abs(heavy) < abs(light)
    assert abs(heavy) < 2.0
