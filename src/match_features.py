"""
Team pool and pairwise prediction helpers for the WC26 tournament simulator.

The existing trained model scores a single team's chance of being a winner.
This module turns those team-level scores into pairwise match probabilities.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from features import engineer_features
from models import WC26EnsembleModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIXTURES_PATH = DATA_DIR / "wc26_real_fixtures.csv"

TEAM_ALIASES = {
    "United States": "USA",
}

TEAM_FALLBACKS = {
    "South Africa": {"country_code": "RSA", "confederation": "CAF", "fifa_rank": 61},
    "Czech Republic": {"country_code": "CZE", "confederation": "UEFA", "fifa_rank": 44},
    "Bosnia and Herzegovina": {"country_code": "BIH", "confederation": "UEFA", "fifa_rank": 72},
    "Haiti": {"country_code": "HAI", "confederation": "CONCACAF", "fifa_rank": 83},
    "Scotland": {"country_code": "SCO", "confederation": "UEFA", "fifa_rank": 36},
    "Curaçao": {"country_code": "CUW", "confederation": "CONCACAF", "fifa_rank": 82},
    "Cape Verde": {"country_code": "CPV", "confederation": "CAF", "fifa_rank": 70},
    "Iraq": {"country_code": "IRQ", "confederation": "AFC", "fifa_rank": 58},
    "Norway": {"country_code": "NOR", "confederation": "UEFA", "fifa_rank": 33},
    "Jordan": {"country_code": "JOR", "confederation": "AFC", "fifa_rank": 64},
    "DR Congo": {"country_code": "COD", "confederation": "CAF", "fifa_rank": 60},
    "Uzbekistan": {"country_code": "UZB", "confederation": "AFC", "fifa_rank": 50},
    "Panama": {"country_code": "PAN", "confederation": "CONCACAF", "fifa_rank": 30},
}


@dataclass(frozen=True)
class TeamRating:
    team_name: str
    country_code: str
    confederation: str
    fifa_rank: float
    fifa_points: float
    model_score: float
    strength_index: float


def _read_team_sources() -> pd.DataFrame:
    frames = []

    test_path = DATA_DIR / "test (2).csv"
    if test_path.exists():
        frames.append(pd.read_csv(test_path))

    train_path = DATA_DIR / "train (1).csv"
    if train_path.exists():
        train = pd.read_csv(train_path)
        if "winner" in train.columns:
            train = train.drop(columns=["winner"])
        frames.append(train)

    if not frames:
        raise FileNotFoundError("No team data found in data/train (1).csv or data/test (2).csv")

    return pd.concat(frames, ignore_index=True)


def load_team_pool(limit: int = 48) -> pd.DataFrame:
    """
    Build the tournament team pool from available data.

    The Kaggle-style data has repeated rows per team. We keep each team's best
    available row by FIFA rank, then FIFA points, and use the top 48 entries.
    """
    teams = _read_team_sources()
    teams = teams.sort_values(["fifa_rank", "fifa_points"], ascending=[True, False])
    teams = teams.drop_duplicates("team_name", keep="first").reset_index(drop=True)

    if len(teams) < limit:
        raise ValueError(f"Need at least {limit} unique teams, found {len(teams)}")

    return teams.head(limit).copy()


def load_real_fixtures(fixtures_path: Path = FIXTURES_PATH) -> pd.DataFrame:
    if not fixtures_path.exists():
        raise FileNotFoundError(f"Real fixture file not found: {fixtures_path}")
    fixtures = pd.read_csv(fixtures_path)
    fixtures["match_number"] = fixtures["match_number"].astype(int)
    return fixtures.sort_values("match_number").reset_index(drop=True)


def _source_rows_by_team() -> pd.DataFrame:
    source = _read_team_sources().copy()
    source["canonical_team_name"] = source["team_name"].replace(TEAM_ALIASES)
    source = source.sort_values(["fifa_rank", "fifa_points"], ascending=[True, False])
    return source.drop_duplicates("canonical_team_name", keep="first").reset_index(drop=True)


def _synthesize_team_row(team_name: str, source_rows: pd.DataFrame) -> pd.Series:
    fallback = TEAM_FALLBACKS.get(team_name)
    if fallback is None:
        raise KeyError(f"No team data or fallback metadata available for {team_name}")

    fifa_rank = float(fallback["fifa_rank"])
    same_confed = source_rows[source_rows["confederation"] == fallback["confederation"]]
    candidates = same_confed if not same_confed.empty else source_rows
    nearest_idx = (candidates["fifa_rank"].astype(float) - fifa_rank).abs().idxmin()
    row = source_rows.loc[nearest_idx].copy()

    row["team_name"] = team_name
    row["canonical_team_name"] = team_name
    row["country_code"] = fallback["country_code"]
    row["confederation"] = fallback["confederation"]
    row["fifa_rank"] = fifa_rank
    row["fifa_points"] = max(1180.0, 1905.0 - fifa_rank * 8.5)
    row["host_advantage"] = 0
    return row


def load_fixture_team_pool(fixtures: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build team rows for the real WC26 fixture.

    Teams missing from the Kaggle input are synthesized from the closest
    available team in the same confederation, with explicit rank/code metadata.
    """
    if fixtures is None:
        fixtures = load_real_fixtures()

    group_fixtures = fixtures[fixtures["stage"].str.startswith("Group ")].copy()
    team_names = sorted(set(group_fixtures["home"]).union(group_fixtures["away"]))
    source_rows = _source_rows_by_team()
    source_lookup = {
        row["canonical_team_name"]: row
        for _, row in source_rows.iterrows()
    }

    rows = []
    for team_name in team_names:
        lookup_name = TEAM_ALIASES.get(team_name, team_name)
        if lookup_name in source_lookup:
            row = source_lookup[lookup_name].copy()
            row["team_name"] = team_name
            row["canonical_team_name"] = team_name
        else:
            row = _synthesize_team_row(team_name, source_rows)
        rows.append(row)

    team_pool = pd.DataFrame(rows).reset_index(drop=True)
    if len(team_pool) != 48:
        raise ValueError(f"Expected 48 fixture teams, found {len(team_pool)}")
    return team_pool


def prepare_model_features(team_rows: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    engineered = engineer_features(team_rows)
    feature_names = list(feature_names)
    missing = [name for name in feature_names if name not in engineered.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    return engineered[feature_names].copy()


class PairwiseMatchPredictor:
    """Convert team-level model scores into pairwise match probabilities."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.model = WC26EnsembleModel(random_state=42)
        self.model.load_models(str(models_dir))
        self.feature_names = self.model.xgb_model.get_booster().feature_names
        if not self.feature_names:
            raise ValueError("Loaded model does not expose feature names")

        self.team_rows: Optional[pd.DataFrame] = None
        self.ratings: Dict[str, TeamRating] = {}

    def fit_team_pool(self, team_rows: pd.DataFrame) -> "PairwiseMatchPredictor":
        self.team_rows = team_rows.reset_index(drop=True).copy()
        features = prepare_model_features(self.team_rows, self.feature_names)
        scores = self.model.predict_proba(features)
        engineered = engineer_features(self.team_rows)

        self.ratings = {}
        for idx, row in self.team_rows.iterrows():
            team_name = row["team_name"]
            self.ratings[team_name] = TeamRating(
                team_name=team_name,
                country_code=row["country_code"],
                confederation=row["confederation"],
                fifa_rank=float(row["fifa_rank"]),
                fifa_points=float(row["fifa_points"]),
                model_score=float(scores[idx]),
                strength_index=float(engineered.loc[idx, "strength_index"]),
            )

        return self

    def row_for(self, team_name: str) -> pd.Series:
        if self.team_rows is None:
            raise ValueError("Call fit_team_pool() before requesting team rows")
        matches = self.team_rows[self.team_rows["team_name"] == team_name]
        if matches.empty:
            raise KeyError(f"Unknown team: {team_name}")
        return matches.iloc[0]

    def win_probability(self, team_a: str, team_b: str) -> float:
        """Return probability that team_a beats team_b in a decided match."""
        rating_a = self.ratings[team_a]
        rating_b = self.ratings[team_b]

        score_a = np.clip(rating_a.model_score, 0.02, 0.98)
        score_b = np.clip(rating_b.model_score, 0.02, 0.98)
        logit_delta = np.log(score_a / (1.0 - score_a)) - np.log(score_b / (1.0 - score_b))
        strength_delta = (rating_a.strength_index - rating_b.strength_index) / 850.0
        rank_delta = (rating_b.fifa_rank - rating_a.fifa_rank) / 45.0

        raw = 0.72 * logit_delta + 0.24 * strength_delta + 0.18 * rank_delta
        return float(1.0 / (1.0 + np.exp(-raw)))

    def expected_goals(self, team_a: str, team_b: str) -> float:
        """Estimate team_a goals against team_b before Poisson sampling."""
        row_a = self.row_for(team_a)
        row_b = self.row_for(team_b)
        win_prob = self.win_probability(team_a, team_b)

        attack = float(row_a["goals_scored_avg"])
        opponent_defense = float(row_b["goals_conceded_avg"])
        shot_quality = float(row_a["shots_per_game"]) * float(row_a["shots_on_target_ratio"]) / 6.0
        strength_factor = 0.72 + 0.56 * win_prob
        host_bonus = 0.10 if float(row_a.get("host_advantage", 0.0)) > 0 else 0.0

        lam = np.sqrt(max(attack, 0.05) * max(opponent_defense, 0.05))
        lam = lam * strength_factor + 0.08 * shot_quality + host_bonus
        return float(np.clip(lam, 0.15, 3.8))

    def ratings_table(self) -> pd.DataFrame:
        rows = [rating.__dict__ for rating in self.ratings.values()]
        return pd.DataFrame(rows).sort_values(["model_score", "fifa_points"], ascending=False)
