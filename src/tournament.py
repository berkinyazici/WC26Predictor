"""
World Cup 2026 tournament simulation from group stage to final.

The official 2026 third-place bracket mapping is complex and depends on which
groups produce third-place qualifiers. This simulator uses a transparent seeded
Round of 32 approximation: all qualifiers are ranked by group finish and table
performance, then paired 1v32, 16v17, 8v25, etc.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from match_features import PairwiseMatchPredictor, load_fixture_team_pool, load_real_fixtures


STAGE_ORDER = ["Group", "Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final", "Champion"]
GROUP_NAMES = list("ABCDEFGHIJKL")


@dataclass
class MatchResult:
    match_number: int
    stage: str
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    winner: str
    decided_by: str
    win_probability_a: float
    log_probability: float
    date: str = ""
    local_time: str = ""
    venue: str = ""


@dataclass
class SimulationResult:
    champion: str
    runner_up: str
    group_tables: Dict[str, pd.DataFrame]
    group_matches: List[MatchResult]
    knockout_matches: List[MatchResult]
    stage_reached: Dict[str, str]
    log_probability: float


def build_seeded_groups(team_pool: pd.DataFrame) -> Dict[str, List[str]]:
    """Create 12 balanced groups of 4 teams from FIFA-ranked pots."""
    seeded = team_pool.sort_values(["fifa_rank", "fifa_points"], ascending=[True, False]).reset_index(drop=True)
    groups = {name: [] for name in GROUP_NAMES}

    for pot_idx in range(4):
        pot = seeded.iloc[pot_idx * 12:(pot_idx + 1) * 12]["team_name"].tolist()
        if pot_idx % 2 == 1:
            pot = list(reversed(pot))
        for group_name, team_name in zip(GROUP_NAMES, pot):
            groups[group_name].append(team_name)

    return groups


class TournamentSimulator:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.fixtures = load_real_fixtures()
        self.team_pool = load_fixture_team_pool(self.fixtures)
        self.groups = self._groups_from_fixtures()
        self.predictor = PairwiseMatchPredictor().fit_team_pool(self.team_pool)

    def _poisson_log_probability(self, goals: int, expected_goals: float) -> float:
        expected_goals = max(float(expected_goals), 1e-6)
        return -expected_goals + goals * math.log(expected_goals) - math.lgamma(goals + 1)

    def _groups_from_fixtures(self) -> Dict[str, List[str]]:
        groups = {name: [] for name in GROUP_NAMES}
        group_fixtures = self.fixtures[self.fixtures["stage"].str.startswith("Group ")]
        for _, fixture in group_fixtures.sort_values("match_number").iterrows():
            group_name = fixture["group"]
            for team in [fixture["home"], fixture["away"]]:
                if team not in groups[group_name]:
                    groups[group_name].append(team)
        return groups

    def _empty_table(self, teams: List[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "team": teams,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "points": 0,
        }).set_index("team")

    def _group_match(self, fixture: pd.Series) -> MatchResult:
        team_a = fixture["home"]
        team_b = fixture["away"]
        lam_a = self.predictor.expected_goals(team_a, team_b)
        lam_b = self.predictor.expected_goals(team_b, team_a)
        goals_a = int(self.rng.poisson(lam_a))
        goals_b = int(self.rng.poisson(lam_b))
        log_probability = (
            self._poisson_log_probability(goals_a, lam_a)
            + self._poisson_log_probability(goals_b, lam_b)
        )

        if goals_a > goals_b:
            winner = team_a
        elif goals_b > goals_a:
            winner = team_b
        else:
            winner = "Draw"

        return MatchResult(
            match_number=int(fixture["match_number"]),
            stage="Group",
            team_a=team_a,
            team_b=team_b,
            goals_a=goals_a,
            goals_b=goals_b,
            winner=winner,
            decided_by="90 minutes",
            win_probability_a=self.predictor.win_probability(team_a, team_b),
            log_probability=log_probability,
            date=fixture["date"],
            local_time=fixture["local_time"],
            venue=fixture["venue"],
        )

    def _apply_group_result(self, table: pd.DataFrame, result: MatchResult) -> None:
        a, b = result.team_a, result.team_b
        ga, gb = result.goals_a, result.goals_b

        table.loc[[a, b], "played"] += 1
        table.loc[a, "gf"] += ga
        table.loc[a, "ga"] += gb
        table.loc[b, "gf"] += gb
        table.loc[b, "ga"] += ga

        if ga > gb:
            table.loc[a, ["wins", "points"]] += [1, 3]
            table.loc[b, "losses"] += 1
        elif gb > ga:
            table.loc[b, ["wins", "points"]] += [1, 3]
            table.loc[a, "losses"] += 1
        else:
            table.loc[[a, b], "draws"] += 1
            table.loc[[a, b], "points"] += 1

        table["gd"] = table["gf"] - table["ga"]

    def simulate_group_stage(self) -> Tuple[Dict[str, pd.DataFrame], List[MatchResult], List[str]]:
        group_tables = {}
        group_matches = []
        direct_qualifiers = []
        third_place_rows = []

        for group_name, teams in self.groups.items():
            table = self._empty_table(teams)
            fixtures = self.fixtures[self.fixtures["group"] == group_name].sort_values("match_number")
            for _, fixture in fixtures.iterrows():
                result = self._group_match(fixture)
                self._apply_group_result(table, result)
                group_matches.append(result)

            table = table.reset_index()
            table["group"] = group_name
            table["model_score"] = table["team"].map(lambda team: self.predictor.ratings[team].model_score)
            table = table.sort_values(
                ["points", "gd", "gf", "model_score"],
                ascending=[False, False, False, False],
            ).reset_index(drop=True)
            table["position"] = np.arange(1, len(table) + 1)
            group_tables[group_name] = table

            direct_qualifiers.extend(table.loc[table["position"] <= 2, "team"].tolist())
            third_place_rows.append(table.loc[table["position"] == 3].iloc[0])

        third_places = pd.DataFrame(third_place_rows).sort_values(
            ["points", "gd", "gf", "model_score"],
            ascending=[False, False, False, False],
        )
        best_thirds = third_places.head(8)["team"].tolist()
        qualifiers = direct_qualifiers + best_thirds
        return group_tables, group_matches, qualifiers

    def _knockout_match(self, fixture: pd.Series, team_a: str, team_b: str) -> MatchResult:
        lam_a = self.predictor.expected_goals(team_a, team_b) * 0.92
        lam_b = self.predictor.expected_goals(team_b, team_a) * 0.92
        goals_a = int(self.rng.poisson(lam_a))
        goals_b = int(self.rng.poisson(lam_b))
        p_a = self.predictor.win_probability(team_a, team_b)
        log_probability = (
            self._poisson_log_probability(goals_a, lam_a)
            + self._poisson_log_probability(goals_b, lam_b)
        )

        if goals_a > goals_b:
            winner = team_a
            decided_by = "90 minutes"
        elif goals_b > goals_a:
            winner = team_b
            decided_by = "90 minutes"
        else:
            winner = team_a if self.rng.random() < p_a else team_b
            decided_by = "extra time/penalties"
            shootout_probability = p_a if winner == team_a else 1.0 - p_a
            log_probability += math.log(max(shootout_probability, 1e-6))

        return MatchResult(
            match_number=int(fixture["match_number"]),
            stage=fixture["stage"],
            team_a=team_a,
            team_b=team_b,
            goals_a=goals_a,
            goals_b=goals_b,
            winner=winner,
            decided_by=decided_by,
            win_probability_a=p_a,
            log_probability=log_probability,
            date=fixture["date"],
            local_time=fixture["local_time"],
            venue=fixture["venue"],
        )

    def _third_place_pool(self, group_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        third_rows = []
        for group_name, table in group_tables.items():
            row = table.loc[table["position"] == 3].iloc[0].copy()
            row["third_slot"] = f"3rd Group {group_name}"
            third_rows.append(row)
        third_places = pd.DataFrame(third_rows).sort_values(
            ["points", "gd", "gf", "model_score"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        return third_places.head(8)

    def _resolve_slot(
        self,
        slot: str,
        group_tables: Dict[str, pd.DataFrame],
        match_winners: Dict[int, str],
        match_losers: Dict[int, str],
        best_thirds: pd.DataFrame,
        used_third_groups: set,
    ) -> str:
        if slot.startswith("Winner Match "):
            return match_winners[int(slot.replace("Winner Match ", ""))]
        if slot.startswith("Loser Match "):
            return match_losers[int(slot.replace("Loser Match ", ""))]
        if slot.startswith("Winner Group "):
            group_name = slot.replace("Winner Group ", "")
            return group_tables[group_name].loc[group_tables[group_name]["position"] == 1, "team"].iloc[0]
        if slot.startswith("Runner-up Group "):
            group_name = slot.replace("Runner-up Group ", "")
            return group_tables[group_name].loc[group_tables[group_name]["position"] == 2, "team"].iloc[0]
        if slot.startswith("3rd Group "):
            allowed_groups = slot.replace("3rd Group ", "").split("/")
            candidates = best_thirds[
                best_thirds["group"].isin(allowed_groups)
                & ~best_thirds["group"].isin(used_third_groups)
            ]
            if candidates.empty:
                candidates = best_thirds[~best_thirds["group"].isin(used_third_groups)]
            if candidates.empty:
                raise ValueError(f"Could not resolve third-place slot: {slot}")
            selected = candidates.iloc[0]
            used_third_groups.add(selected["group"])
            return selected["team"]
        return slot

    def simulate_knockout(self, group_tables: Dict[str, pd.DataFrame], qualifiers: List[str]) -> List[MatchResult]:
        knockout_matches = []
        match_winners = {}
        match_losers = {}
        best_thirds = self._third_place_pool(group_tables)
        used_third_groups = set()

        knockout_fixtures = self.fixtures[~self.fixtures["stage"].str.startswith("Group ")].sort_values("match_number")
        for _, fixture in knockout_fixtures.iterrows():
            team_a = self._resolve_slot(
                fixture["home"], group_tables, match_winners, match_losers, best_thirds, used_third_groups
            )
            team_b = self._resolve_slot(
                fixture["away"], group_tables, match_winners, match_losers, best_thirds, used_third_groups
            )
            result = self._knockout_match(fixture, team_a, team_b)
            knockout_matches.append(result)
            match_winners[result.match_number] = result.winner
            match_losers[result.match_number] = result.team_b if result.winner == result.team_a else result.team_a
        return knockout_matches

    def simulate_once(self) -> SimulationResult:
        group_tables, group_matches, qualifiers = self.simulate_group_stage()
        knockout_matches = self.simulate_knockout(group_tables, qualifiers)

        stage_reached = {team: "Group" for team in self.team_pool["team_name"].tolist()}
        for team in qualifiers:
            stage_reached[team] = "Round of 32"

        for match in knockout_matches:
            if match.stage == "Match for third place":
                continue
            stage_reached[match.winner] = {
                "Round of 32": "Round of 16",
                "Round of 16": "Quarterfinal",
                "Quarterfinals": "Semifinal",
                "Semifinals": "Final",
                "Final": "Champion",
            }[match.stage]

        final = [match for match in knockout_matches if match.stage == "Final"][-1]
        runner_up = final.team_b if final.winner == final.team_a else final.team_a
        return SimulationResult(
            champion=final.winner,
            runner_up=runner_up,
            group_tables=group_tables,
            group_matches=group_matches,
            knockout_matches=knockout_matches,
            stage_reached=stage_reached,
            log_probability=sum(match.log_probability for match in group_matches + knockout_matches),
        )

    def run_monte_carlo(self, n_simulations: int = 2000) -> Tuple[pd.DataFrame, SimulationResult]:
        counts = {
            team: {stage: 0 for stage in STAGE_ORDER}
            for team in self.team_pool["team_name"].tolist()
        }

        best_result = None
        best_log_probability = -np.inf
        for sim_idx in range(n_simulations):
            result = self.simulate_once()
            if result.log_probability > best_log_probability:
                best_result = result
                best_log_probability = result.log_probability
            for team, stage in result.stage_reached.items():
                reached_idx = STAGE_ORDER.index(stage)
                for prior_stage in STAGE_ORDER[:reached_idx + 1]:
                    counts[team][prior_stage] += 1

        rows = []
        for team, stage_counts in counts.items():
            rating = self.predictor.ratings[team]
            row = {
                "team": team,
                "country_code": rating.country_code,
                "fifa_rank": rating.fifa_rank,
                "model_score": rating.model_score,
            }
            for stage, count in stage_counts.items():
                row[f"{stage.lower().replace(' ', '_')}_prob"] = count / n_simulations
            rows.append(row)

        probabilities = pd.DataFrame(rows).sort_values("champion_prob", ascending=False).reset_index(drop=True)
        return probabilities, best_result


def run_tournament_simulation(n_simulations: int = 2000, random_state: int = 42):
    simulator = TournamentSimulator(random_state=random_state)
    probabilities, display_result = simulator.run_monte_carlo(n_simulations=n_simulations)
    return simulator, probabilities, display_result
