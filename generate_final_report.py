"""
Generate a full WC26 tournament simulation report with plots.
"""

from pathlib import Path
import os
import sys

import pandas as pd

sys.path.insert(0, "src")

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

from models import WC26EnsembleModel
from tournament import run_tournament_simulation
from visualization import (
    ensure_dir,
    plot_bracket,
    plot_champion_probabilities,
    plot_feature_importance,
    plot_group_tables,
    plot_stage_probabilities,
    plot_tournament_tree,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tournament"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table without tabulate."""
    if df.empty:
        return "_No rows._"

    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: format(value, floatfmt))
        else:
            formatted[col] = formatted[col].astype(str)

    headers = [str(col) for col in formatted.columns]
    rows = formatted.values.tolist()

    def row(values):
        return "| " + " | ".join(str(value) for value in values) + " |"

    lines = [row(headers), row(["---"] * len(headers))]
    lines.extend(row(values) for values in rows)
    return "\n".join(lines)


def _match_rows(matches):
    rows = []
    for match in matches:
        rows.append({
            "match_number": match.match_number,
            "date": match.date,
            "stage": match.stage,
            "team_a": match.team_a,
            "team_b": match.team_b,
            "score": f"{match.goals_a}-{match.goals_b}",
            "winner": match.winner,
            "decided_by": match.decided_by,
            "team_a_win_probability": round(match.win_probability_a, 4),
            "log_probability": round(match.log_probability, 4),
            "venue": match.venue,
        })
    return pd.DataFrame(rows)


def _write_markdown_report(probabilities, display_result, n_simulations, feature_importance):
    report_path = REPORTS_DIR / "final_report.md"
    top = probabilities.head(12)
    knockout_df = _match_rows(display_result.knockout_matches)
    group_match_df = _match_rows(display_result.group_matches)
    group_summary = []
    for group_name, table in display_result.group_tables.items():
        group_summary.extend(table[["group", "position", "team", "points", "gd", "gf"]].to_dict("records"))
    group_summary_df = pd.DataFrame(group_summary)

    lines = []
    lines.append("# WC26 Predictor Final Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"This report simulates the World Cup 2026 tournament from the group stage through the final "
        f"using {n_simulations:,} Monte Carlo runs. The simulation combines the trained WC26 ensemble "
        "model with a score-based Poisson match engine so that group-stage draws, goal difference, "
        "best third-place teams, knockout matches, and penalty decisions can all be represented."
    )
    lines.append("")
    lines.append(f"Most likely champion by simulation frequency: **{top.iloc[0]['team']}** "
                 f"({top.iloc[0]['champion_prob'] * 100:.1f}%).")
    lines.append("")
    lines.append(
        f"Best single simulated tournament path: **{display_result.champion}** wins the final "
        f"against **{display_result.runner_up}**. This path is the highest-likelihood complete "
        f"simulation among the {n_simulations:,} Monte Carlo runs."
    )
    lines.append("")
    lines.append("## Recent Development Summary")
    lines.append("")
    lines.append("The project was upgraded from a seeded placeholder tournament simulation to a real-fixture simulation workflow. The main additions are:")
    lines.append("")
    lines.append("- Official WC26 group and knockout fixture data in `data/wc26_real_fixtures.csv`.")
    lines.append("- Real group-stage dates, venues, and match numbers for all 72 group matches.")
    lines.append("- Official Match 73-104 knockout slots, including `Winner Group`, `Runner-up Group`, `3rd Group`, `Winner Match`, and `Loser Match` dependencies.")
    lines.append("- A fallback team-feature strategy for real WC26 teams that are missing from the original Kaggle team dataset.")
    lines.append("- A 1,000-run Monte Carlo pipeline that produces team progression probabilities.")
    lines.append("- A best-single-simulation selector that chooses the highest-likelihood complete tournament path from the Monte Carlo sample.")
    lines.append("- A two-sided tournament tree image inspired by broadcast-style World Cup bracket graphics.")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- Fixture source: `data/wc26_real_fixtures.csv`, parsed from the public FIFA/Wikipedia fixture listing.")
    lines.append("- Groups: official 2026 World Cup groups and group match dates/venues from the fixture file.")
    lines.append("- Group matches: Poisson score sampling from team attack, opponent defense, and pairwise model strength.")
    lines.append("- Knockout qualification: group winners, runners-up, and the best eight third-place teams.")
    lines.append("- Knockout bracket: official match slots from Match 73 through Match 104.")
    lines.append("- Third-place Round of 32 slots: resolved by selecting the strongest qualifying third-place team from each slot's allowed group set.")
    lines.append("- Knockout draws: resolved by the pairwise win probability as extra time/penalties.")
    lines.append("- Data caveat: teams absent from the Kaggle team-feature data receive confederation/rank-based fallback features.")
    lines.append("")
    lines.append("## Model Notes")
    lines.append("")
    lines.append(
        "The existing ensemble is a calibrated XGBoost and Random Forest model trained on team-level "
        "winner labels. For tournament use, each team receives a model score and pairwise probabilities "
        "are derived from the difference between team model scores, strength index, and FIFA rank."
    )
    lines.append("")
    lines.append("Recommended next model upgrade: retrain on explicit match-level rows with "
                 "`home_win / draw / away_win` labels and direct team-difference features.")
    lines.append("")
    lines.append("## Key Visuals")
    lines.append("")
    lines.append("![Champion probabilities](../outputs/tournament/figures/champion_probabilities.png)")
    lines.append("")
    lines.append("![Stage probabilities](../outputs/tournament/figures/stage_probabilities.png)")
    lines.append("")
    lines.append("![Group tables](../outputs/tournament/figures/group_tables.png)")
    lines.append("")
    lines.append("![Tournament bracket](../outputs/tournament/figures/tournament_bracket.png)")
    lines.append("")
    lines.append("![Best simulation tournament tree](../outputs/tournament/figures/tournament_tree_best.png)")
    lines.append("")
    lines.append("![Feature importance](../outputs/tournament/figures/feature_importance.png)")
    lines.append("")
    lines.append("## Generated Artifacts")
    lines.append("")
    lines.append("| Artifact | Description |")
    lines.append("| --- | --- |")
    lines.append(f"| `outputs/tournament/stage_probabilities.csv` | Team-level probabilities for reaching each tournament stage across {n_simulations:,} simulations. |")
    lines.append("| `outputs/tournament/best_group_matches.csv` | Group-stage matches from the highest-likelihood complete simulation. |")
    lines.append("| `outputs/tournament/best_knockout_matches.csv` | Knockout-stage matches from the highest-likelihood complete simulation. |")
    lines.append("| `outputs/tournament/figures/tournament_tree_best.png` | Filled two-sided tournament tree for the best single simulation. |")
    lines.append("| `outputs/tournament/figures/champion_probabilities.png` | Top champion probabilities across all Monte Carlo simulations. |")
    lines.append("| `outputs/tournament/figures/stage_probabilities.png` | Cumulative stage reach probabilities for leading teams. |")
    lines.append("| `outputs/tournament/figures/group_tables.png` | Group tables from the best single simulation. |")
    lines.append("| `outputs/tournament/figures/feature_importance.png` | Top XGBoost feature importances from the trained model. |")
    lines.append("")
    lines.append("## Champion Probability Table")
    lines.append("")
    lines.append(_markdown_table(top[[
        "team", "country_code", "fifa_rank", "round_of_16_prob", "quarterfinal_prob",
        "semifinal_prob", "final_prob", "champion_prob"
    ]], floatfmt=".3f"))
    lines.append("")
    lines.append("## Simulated Knockout Path")
    lines.append("")
    lines.append(_markdown_table(knockout_df, floatfmt=".3f"))
    lines.append("")
    lines.append("## Simulated Group Tables")
    lines.append("")
    lines.append(_markdown_table(group_summary_df, floatfmt=".3f"))
    lines.append("")
    lines.append("## Simulated Group Matches")
    lines.append("")
    lines.append(_markdown_table(group_match_df, floatfmt=".3f"))
    lines.append("")
    lines.append("## Top Feature Importances")
    lines.append("")
    lines.append(_markdown_table(feature_importance, floatfmt=".3f"))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Because the model is currently team-level rather than match-level, the probabilities should be "
        "read as a structured simulation baseline rather than a final betting-grade forecast. The most "
        "important improvement is to create historical match pair rows and teach the model draws and score "
        "distributions directly."
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main(n_simulations: int = 2000):
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(REPORTS_DIR)

    simulator, probabilities, display_result = run_tournament_simulation(
        n_simulations=n_simulations,
        random_state=42,
    )

    probabilities.to_csv(OUTPUT_DIR / "stage_probabilities.csv", index=False)
    _match_rows(display_result.group_matches).to_csv(OUTPUT_DIR / "best_group_matches.csv", index=False)
    _match_rows(display_result.knockout_matches).to_csv(OUTPUT_DIR / "best_knockout_matches.csv", index=False)
    _match_rows(display_result.group_matches).to_csv(OUTPUT_DIR / "sample_group_matches.csv", index=False)
    _match_rows(display_result.knockout_matches).to_csv(OUTPUT_DIR / "sample_knockout_matches.csv", index=False)

    model = WC26EnsembleModel(random_state=42)
    model.load_models("models")
    feature_importance = model.feature_importance(top_n=15)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    plot_champion_probabilities(probabilities, FIGURES_DIR / "champion_probabilities.png")
    plot_stage_probabilities(probabilities, FIGURES_DIR / "stage_probabilities.png")
    plot_group_tables(display_result.group_tables, FIGURES_DIR / "group_tables.png")
    plot_bracket(display_result.knockout_matches, FIGURES_DIR / "tournament_bracket.png")
    plot_tournament_tree(
        display_result.knockout_matches,
        FIGURES_DIR / "tournament_tree_best.png",
        simulation_log_probability=display_result.log_probability,
    )
    plot_feature_importance(feature_importance, FIGURES_DIR / "feature_importance.png")

    report_path = _write_markdown_report(probabilities, display_result, n_simulations, feature_importance)

    print("Tournament simulation complete")
    print(f"Report: {report_path}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Champion: {probabilities.iloc[0]['team']} ({probabilities.iloc[0]['champion_prob'] * 100:.1f}%)")
    print(f"Best single simulation champion: {display_result.champion}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate WC26 tournament final report")
    parser.add_argument("--simulations", type=int, default=2000, help="Number of Monte Carlo simulations")
    args = parser.parse_args()
    main(n_simulations=args.simulations)
