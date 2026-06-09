"""
Visualization helpers for WC26 tournament simulation outputs.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_champion_probabilities(probabilities: pd.DataFrame, output_path: Path, top_n: int = 16) -> None:
    top = probabilities.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(top)))
    ax.barh(top["team"], top["champion_prob"] * 100, color=colors)
    ax.set_xlabel("Champion probability (%)")
    ax.set_title("Top World Cup 2026 Champion Probabilities")
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(top["champion_prob"] * 100):
        ax.text(value + 0.2, idx, f"{value:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_stage_probabilities(probabilities: pd.DataFrame, output_path: Path, top_n: int = 12) -> None:
    cols = [
        "round_of_32_prob",
        "round_of_16_prob",
        "quarterfinal_prob",
        "semifinal_prob",
        "final_prob",
        "champion_prob",
    ]
    labels = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    top = probabilities.head(top_n).set_index("team")

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(top))
    left = np.zeros(len(top))
    palette = ["#d7e9f7", "#aed4ef", "#7fb9de", "#5598c9", "#2f73ad", "#f0b429"]

    prev = np.zeros(len(top))
    for col, label, color in zip(cols, labels, palette):
        values = top[col].values * 100
        incremental = np.maximum(values - prev, 0)
        ax.barh(y, incremental, left=left, label=label, color=color)
        left += incremental
        prev = values

    ax.set_yticks(y)
    ax.set_yticklabels(top.index)
    ax.invert_yaxis()
    ax.set_xlabel("Stage reach probability, cumulative (%)")
    ax.set_title("Tournament Progression Probability")
    ax.legend(ncol=3, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: Path) -> None:
    data = feature_importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(data["feature_name"], data["importance"], color="#2d6cdf")
    ax.set_xlabel("XGBoost split importance")
    ax.set_title("Model Feature Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_tables(group_tables: Dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(14, 15))
    axes = axes.flatten()

    for ax, (group_name, table) in zip(axes, group_tables.items()):
        ax.axis("off")
        view = table[["position", "team", "points", "gd", "gf"]].copy()
        view.columns = ["#", "Team", "Pts", "GD", "GF"]
        tbl = ax.table(cellText=view.values, colLabels=view.columns, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.35)
        ax.set_title(f"Group {group_name}", fontsize=12, pad=8)

    fig.suptitle("Simulated Group Stage Tables", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bracket(knockout_matches: List, output_path: Path) -> None:
    stage_order = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
    matches_by_stage = {stage: [m for m in knockout_matches if m.stage == stage] for stage in stage_order}

    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_facecolor("#061f2b")
    fig.patch.set_facecolor("#061f2b")
    ax.axis("off")

    x_positions = np.linspace(0.05, 0.88, len(stage_order))
    for x, stage in zip(x_positions, stage_order):
        ax.text(x, 0.96, stage.upper(), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        matches = matches_by_stage[stage]
        y_positions = np.linspace(0.92, 0.08, len(matches)) if len(matches) > 1 else [0.50]
        for y, match in zip(y_positions, matches):
            short_date = match.date.replace("2026-", "")
            if match.decided_by != "90 minutes":
                suffix = " p"
            else:
                suffix = ""
            label = (
                f"M{match.match_number} {short_date}\n"
                f"{match.team_a} {match.goals_a}-{match.goals_b} {match.team_b}{suffix}\n"
                f"W: {match.winner}"
            )
            color = "#0f3a4a"
            edge = "#cf1f5a" if match.winner else "#557"
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=color, edgecolor=edge, linewidth=1.0),
            )

    final = knockout_matches[-1]
    ax.text(0.965, 0.55, "WINNER", ha="center", color="#f0b429", fontsize=12, fontweight="bold")
    ax.text(
        0.965,
        0.48,
        final.winner.upper(),
        ha="center",
        va="center",
        color="white",
        fontsize=18,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#b51f2b", edgecolor="#f0b429", linewidth=1.5),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tournament_tree(knockout_matches: List, output_path: Path, simulation_log_probability=None) -> None:
    """
    Draw a two-sided World Cup bracket similar to broadcast tournament trees.

    The left half contains the first eight Round of 32 matches and their path to
    the first semifinalist. The right half mirrors the remaining eight matches.
    """
    stage_order = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
    by_stage = {stage: [m for m in knockout_matches if m.stage == stage] for stage in stage_order}
    final = by_stage["Final"][0]
    champion = final.winner

    fig, ax = plt.subplots(figsize=(18, 10))
    bg = "#062630"
    panel = "#103944"
    line = "#e31b63"
    winner_fill = "#c81f32"
    gold = "#f5bf24"
    text = "#eef7f8"
    muted = "#a9c5cc"

    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    headings = [
        (0.08, "ROUND OF 32"), (0.21, "R16"), (0.32, "QF"), (0.42, "SF"),
        (0.50, "FINAL"), (0.58, "SF"), (0.68, "QF"), (0.79, "R16"), (0.92, "ROUND OF 32"),
    ]
    for x, label in headings:
        ax.text(x, 0.965, label, color=text, fontsize=12, ha="center", va="center", fontweight="bold", alpha=0.95)

    def short_team(team):
        return team.upper() if len(team) <= 14 else team[:13].upper() + "."

    def match_label(match):
        suffix = " (P)" if match.decided_by != "90 minutes" else ""
        return (
            f"M{match.match_number}  {match.date[5:]}\n"
            f"{short_team(match.team_a):<14} {match.goals_a}\n"
            f"{short_team(match.team_b):<14} {match.goals_b}{suffix}"
        )

    def winner_label(match):
        suffix = "P" if match.decided_by != "90 minutes" else ""
        return f"{short_team(match.winner)}\n{match.goals_a}-{match.goals_b}{suffix}"

    def draw_card(x, y, label, width, height, fontsize=7.5, face=panel, edge=None, align="center"):
        ax.text(
            x,
            y,
            label,
            ha=align,
            va="center",
            color=text,
            fontsize=fontsize,
            linespacing=1.15,
            family="DejaVu Sans Mono" if "\n" in label else "DejaVu Sans",
            bbox=dict(
                boxstyle="round,pad=0.26,rounding_size=0.02",
                facecolor=face,
                edgecolor=edge or face,
                linewidth=1.2,
            ),
        )

    def draw_connector(x1, y1, x2, y2, side):
        mid = (x1 + x2) / 2.0
        ax.plot([x1, mid, mid, x2], [y1, y1, y2, y2], color=line, lw=1.25, alpha=0.95)

    left_r32 = by_stage["Round of 32"][:8]
    right_r32 = by_stage["Round of 32"][8:]
    left_r16 = by_stage["Round of 16"][:4]
    right_r16 = by_stage["Round of 16"][4:]
    left_qf = by_stage["Quarterfinals"][:2]
    right_qf = by_stage["Quarterfinals"][2:]
    left_sf = by_stage["Semifinals"][:1]
    right_sf = by_stage["Semifinals"][1:]

    layout = {
        "left": {
            "r32_x": 0.08, "r16_x": 0.22, "qf_x": 0.33, "sf_x": 0.43,
            "r32_y": np.linspace(0.86, 0.18, 8),
        },
        "right": {
            "r32_x": 0.92, "r16_x": 0.78, "qf_x": 0.67, "sf_x": 0.57,
            "r32_y": np.linspace(0.86, 0.18, 8),
        },
    }

    def draw_side(side_name, r32, r16, qf, sf):
        loc = layout[side_name]
        r32_y = list(loc["r32_y"])
        r16_y = [(r32_y[i * 2] + r32_y[i * 2 + 1]) / 2 for i in range(4)]
        qf_y = [(r16_y[i * 2] + r16_y[i * 2 + 1]) / 2 for i in range(2)]
        sf_y = [(qf_y[0] + qf_y[1]) / 2]

        for match, y in zip(r32, r32_y):
            draw_card(loc["r32_x"], y, match_label(match), 0.15, 0.055, fontsize=6.6)
        for match, y in zip(r16, r16_y):
            draw_card(loc["r16_x"], y, winner_label(match), 0.07, 0.045, fontsize=7.0, edge=line)
        for match, y in zip(qf, qf_y):
            draw_card(loc["qf_x"], y, winner_label(match), 0.07, 0.045, fontsize=7.0, edge=line)
        for match, y in zip(sf, sf_y):
            draw_card(loc["sf_x"], y, winner_label(match), 0.07, 0.045, fontsize=7.0, edge=line)

        for idx, y in enumerate(r32_y):
            draw_connector(loc["r32_x"] + (0.065 if side_name == "left" else -0.065), y,
                           loc["r16_x"] - (0.035 if side_name == "left" else -0.035), r16_y[idx // 2], side_name)
        for idx, y in enumerate(r16_y):
            draw_connector(loc["r16_x"] + (0.035 if side_name == "left" else -0.035), y,
                           loc["qf_x"] - (0.035 if side_name == "left" else -0.035), qf_y[idx // 2], side_name)
        for idx, y in enumerate(qf_y):
            draw_connector(loc["qf_x"] + (0.035 if side_name == "left" else -0.035), y,
                           loc["sf_x"] - (0.035 if side_name == "left" else -0.035), sf_y[0], side_name)
        return sf_y[0]

    left_sf_y = draw_side("left", left_r32, left_r16, left_qf, left_sf)
    right_sf_y = draw_side("right", right_r32, right_r16, right_qf, right_sf)

    final_y = 0.50
    draw_connector(0.465, left_sf_y, 0.49, final_y, "left")
    draw_connector(0.535, right_sf_y, 0.51, final_y, "right")

    draw_card(0.50, 0.56, f"{short_team(final.team_a)} {final.goals_a}", 0.08, 0.04, fontsize=8.2, edge=line)
    draw_card(0.50, 0.47, f"{short_team(final.team_b)} {final.goals_b}", 0.08, 0.04, fontsize=8.2, edge=line)
    ax.plot([0.50, 0.50], [0.535, 0.495], color=line, lw=1.25)

    ax.text(0.50, 0.37, "WINNERS", color=muted, fontsize=10, ha="center", va="center", fontweight="bold")
    ax.text(
        0.50,
        0.31,
        short_team(champion),
        ha="center",
        va="center",
        color=text,
        fontsize=26,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.45,rounding_size=0.08",
            facecolor=winner_fill,
            edgecolor=gold,
            linewidth=1.8,
        ),
    )
    likelihood_text = ""
    if simulation_log_probability is not None:
        likelihood_text = f"\nlog-likelihood: {simulation_log_probability:.1f}"
    ax.text(
        0.50,
        0.235,
        f"Best single simulation from Monte Carlo sample{likelihood_text}",
        ha="center",
        va="center",
        color=muted,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
