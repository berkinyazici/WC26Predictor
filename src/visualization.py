"""
Visualization helpers for WC26 tournament simulation outputs.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
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


FLAG_SPECS = {
    "Algeria": ("algeria", ["#006233", "#ffffff", "#d21034"], None),
    "Argentina": ("horizontal", ["#74acdf", "#ffffff", "#74acdf"], "#f6b40e"),
    "Australia": ("solid", ["#012169"], "#ffffff"),
    "Belgium": ("vertical", ["#000000", "#ffd90c", "#ef3340"], None),
    "Brazil": ("brazil", ["#009b3a", "#ffdf00", "#002776"], None),
    "Canada": ("vertical", ["#d52b1e", "#ffffff", "#d52b1e"], None),
    "Colombia": ("horizontal", ["#fcd116", "#003893", "#ce1126"], None),
    "Croatia": ("horizontal", ["#ff0000", "#ffffff", "#171796"], "#d00000"),
    "Curaçao": ("solid", ["#002b7f"], "#f9e814"),
    "DR Congo": ("diagonal", ["#007fff", "#f7d618", "#ce1021"], None),
    "Ecuador": ("horizontal", ["#ffdd00", "#034ea2", "#ed1c24"], None),
    "England": ("cross", ["#ffffff", "#ce1124"], None),
    "France": ("vertical", ["#002395", "#ffffff", "#ed2939"], None),
    "Germany": ("horizontal", ["#000000", "#dd0000", "#ffce00"], None),
    "Ghana": ("horizontal", ["#ce1126", "#fcd116", "#006b3f"], "#000000"),
    "Iran": ("horizontal", ["#239f40", "#ffffff", "#da0000"], None),
    "Japan": ("circle", ["#ffffff"], "#bc002d"),
    "Mexico": ("vertical", ["#006847", "#ffffff", "#ce1126"], None),
    "Morocco": ("solid", ["#c1272d"], "#006233"),
    "Netherlands": ("horizontal", ["#ae1c28", "#ffffff", "#21468b"], None),
    "New Zealand": ("solid", ["#00247d"], "#cc142b"),
    "Norway": ("nordic", ["#ba0c2f", "#ffffff", "#00205b"], None),
    "Paraguay": ("paraguay", ["#d52b1e", "#ffffff", "#0038a8"], "#1f7a3a"),
    "Portugal": ("vertical", ["#006600", "#ff0000"], "#ffcc00"),
    "Qatar": ("vertical", ["#ffffff", "#8a1538"], None),
    "Senegal": ("vertical", ["#00853f", "#fdef42", "#e31b23"], "#00853f"),
    "South Africa": ("south_africa", ["#007a4d", "#ffb612", "#000000", "#de3831", "#002395", "#ffffff"], None),
    "South Korea": ("circle", ["#ffffff"], "#cd2e3a"),
    "Spain": ("horizontal", ["#aa151b", "#f1bf00", "#aa151b"], None),
    "Sweden": ("nordic", ["#006aa7", "#fecc00", "#fecc00"], None),
    "Switzerland": ("swiss", ["#d52b1e", "#ffffff"], None),
    "Turkey": ("solid", ["#e30a17"], "#ffffff"),
    "United States": ("stripes", ["#b22234", "#ffffff"], "#3c3b6e"),
    "Uruguay": ("stripes", ["#ffffff", "#0038a8"], "#fcd116"),
}


def _draw_flag(ax, team: str, x: float, y: float, width: float, height: float, zorder: int = 5) -> None:
    spec = FLAG_SPECS.get(team)
    border = "#d8eef0"
    if spec is None:
        ax.add_patch(Rectangle((x, y), width, height, facecolor="#263f48", edgecolor=border, lw=0.6, zorder=zorder))
        ax.text(x + width / 2, y + height / 2, team[:3].upper(), color="white", fontsize=5,
                ha="center", va="center", zorder=zorder + 1)
        return

    kind, colors, accent = spec
    ax.add_patch(Rectangle((x, y), width, height, facecolor=colors[0], edgecolor=border, lw=0.6, zorder=zorder))

    if kind == "horizontal":
        stripe_h = height / len(colors)
        for idx, color in enumerate(colors):
            ax.add_patch(Rectangle((x, y + height - (idx + 1) * stripe_h), width, stripe_h,
                                   facecolor=color, edgecolor="none", zorder=zorder + 1))
        if accent:
            ax.add_patch(Circle((x + width * 0.50, y + height * 0.50), min(width, height) * 0.13,
                                facecolor=accent, edgecolor="none", zorder=zorder + 2))
    elif kind == "vertical":
        stripe_w = width / len(colors)
        for idx, color in enumerate(colors):
            ax.add_patch(Rectangle((x + idx * stripe_w, y), stripe_w, height,
                                   facecolor=color, edgecolor="none", zorder=zorder + 1))
        if accent:
            ax.add_patch(Circle((x + width * 0.50, y + height * 0.50), min(width, height) * 0.13,
                                facecolor=accent, edgecolor="none", zorder=zorder + 2))
    elif kind == "solid":
        ax.add_patch(Rectangle((x, y), width, height, facecolor=colors[0], edgecolor="none", zorder=zorder + 1))
        if accent:
            ax.add_patch(Circle((x + width * 0.55, y + height * 0.52), min(width, height) * 0.13,
                                facecolor=accent, edgecolor="none", zorder=zorder + 2))
    elif kind == "circle":
        ax.add_patch(Rectangle((x, y), width, height, facecolor=colors[0], edgecolor="none", zorder=zorder + 1))
        ax.add_patch(Circle((x + width * 0.50, y + height * 0.50), min(width, height) * 0.25,
                            facecolor=accent, edgecolor="none", zorder=zorder + 2))
    elif kind == "cross":
        ax.add_patch(Rectangle((x, y), width, height, facecolor=colors[0], edgecolor="none", zorder=zorder + 1))
        ax.add_patch(Rectangle((x + width * 0.43, y), width * 0.14, height, facecolor=colors[1],
                               edgecolor="none", zorder=zorder + 2))
        ax.add_patch(Rectangle((x, y + height * 0.41), width, height * 0.18, facecolor=colors[1],
                               edgecolor="none", zorder=zorder + 2))
    elif kind == "nordic":
        ax.add_patch(Rectangle((x, y), width, height, facecolor=colors[0], edgecolor="none", zorder=zorder + 1))
        ax.add_patch(Rectangle((x + width * 0.30, y), width * 0.16, height, facecolor=colors[1],
                               edgecolor="none", zorder=zorder + 2))
        ax.add_patch(Rectangle((x, y + height * 0.40), width, height * 0.20, facecolor=colors[1],
                               edgecolor="none", zorder=zorder + 2))
        if len(colors) > 2 and colors[2] != colors[1]:
            ax.add_patch(Rectangle((x + width * 0.34, y), width * 0.08, height, facecolor=colors[2],
                                   edgecolor="none", zorder=zorder + 3))
            ax.add_patch(Rectangle((x, y + height * 0.45), width, height * 0.10, facecolor=colors[2],
                                   edgecolor="none", zorder=zorder + 3))
    elif kind == "swiss":
        ax.add_patch(Rectangle((x, y), width, height, facecolor=colors[0], edgecolor="none", zorder=zorder + 1))
        ax.add_patch(Rectangle((x + width * 0.42, y + height * 0.22), width * 0.16, height * 0.56,
                               facecolor=colors[1], edgecolor="none", zorder=zorder + 2))
        ax.add_patch(Rectangle((x + width * 0.25, y + height * 0.39), width * 0.50, height * 0.18,
                               facecolor=colors[1], edgecolor="none", zorder=zorder + 2))
    elif kind == "brazil":
        ax.add_patch(Rectangle((x, y), width, height, facecolor=colors[0], edgecolor="none", zorder=zorder + 1))
        ax.add_patch(Polygon([
            (x + width * 0.50, y + height * 0.90),
            (x + width * 0.90, y + height * 0.50),
            (x + width * 0.50, y + height * 0.10),
            (x + width * 0.10, y + height * 0.50),
        ], facecolor=colors[1], edgecolor="none", zorder=zorder + 2))
        ax.add_patch(Circle((x + width * 0.50, y + height * 0.50), min(width, height) * 0.23,
                            facecolor=colors[2], edgecolor="none", zorder=zorder + 3))
    elif kind == "diagonal":
        ax.add_patch(Rectangle((x, y), width, height, facecolor=colors[0], edgecolor="none", zorder=zorder + 1))
        ax.add_patch(Polygon([(x, y + height * 0.10), (x + width * 0.12, y),
                              (x + width, y + height * 0.90), (x + width * 0.88, y + height)],
                             facecolor=colors[1], edgecolor="none", zorder=zorder + 2))
        ax.add_patch(Polygon([(x, y + height * 0.18), (x + width * 0.06, y),
                              (x + width, y + height * 0.82), (x + width * 0.94, y + height)],
                             facecolor=colors[2], edgecolor="none", zorder=zorder + 3))
    elif kind == "south_africa":
        ax.add_patch(Rectangle((x, y + height / 2), width, height / 2, facecolor=colors[3], edgecolor="none", zorder=zorder + 1))
        ax.add_patch(Rectangle((x, y), width, height / 2, facecolor=colors[4], edgecolor="none", zorder=zorder + 1))
        ax.add_patch(Polygon([(x, y), (x + width * 0.48, y + height / 2), (x, y + height)],
                             facecolor=colors[0], edgecolor="none", zorder=zorder + 2))
        ax.add_patch(Polygon([(x, y + height * 0.08), (x + width * 0.34, y + height / 2),
                              (x, y + height * 0.92)], facecolor=colors[2], edgecolor="none", zorder=zorder + 3))
    elif kind == "stripes":
        stripe_h = height / 7
        for idx in range(7):
            ax.add_patch(Rectangle((x, y + idx * stripe_h), width, stripe_h,
                                   facecolor=colors[idx % 2], edgecolor="none", zorder=zorder + 1))
        if accent:
            ax.add_patch(Rectangle((x, y + height * 0.45), width * 0.45, height * 0.55,
                                   facecolor=accent, edgecolor="none", zorder=zorder + 2))
    elif kind == "algeria":
        ax.add_patch(Rectangle((x, y), width / 2, height, facecolor=colors[0], edgecolor="none", zorder=zorder + 1))
        ax.add_patch(Rectangle((x + width / 2, y), width / 2, height, facecolor=colors[1], edgecolor="none", zorder=zorder + 1))
        crescent_x = x + width * 0.55
        crescent_y = y + height * 0.50
        radius = min(width, height) * 0.31
        ax.add_patch(Circle((crescent_x, crescent_y), radius, facecolor=colors[2], edgecolor="none", zorder=zorder + 2))
        ax.add_patch(Circle((crescent_x + radius * 0.42, crescent_y), radius * 0.82,
                            facecolor=colors[1], edgecolor="none", zorder=zorder + 3))
        star_cx = x + width * 0.68
        star_cy = y + height * 0.50
        outer = min(width, height) * 0.16
        inner = outer * 0.42
        pts = []
        for idx in range(10):
            angle = np.pi / 2 + idx * np.pi / 5
            r = outer if idx % 2 == 0 else inner
            pts.append((star_cx + r * np.cos(angle), star_cy + r * np.sin(angle)))
        ax.add_patch(Polygon(pts, facecolor=colors[2], edgecolor="none", zorder=zorder + 4))
    elif kind == "paraguay":
        stripe_h = height / 3
        for idx, color in enumerate(colors):
            ax.add_patch(Rectangle((x, y + height - (idx + 1) * stripe_h), width, stripe_h,
                                   facecolor=color, edgecolor="none", zorder=zorder + 1))
        seal_r = min(width, height) * 0.18
        seal_x = x + width * 0.50
        seal_y = y + height * 0.50
        ax.add_patch(Circle((seal_x, seal_y), seal_r, facecolor="#ffffff", edgecolor="#222222", lw=0.25, zorder=zorder + 2))
        ax.add_patch(Circle((seal_x, seal_y), seal_r * 0.72, facecolor="#ffffff", edgecolor=accent, lw=0.35, zorder=zorder + 3))
        ax.add_patch(Circle((seal_x, seal_y), seal_r * 0.23, facecolor="#f6d32d", edgecolor="none", zorder=zorder + 4))
        ax.add_patch(Circle((seal_x, seal_y), seal_r * 0.10, facecolor=accent, edgecolor="none", zorder=zorder + 5))

    ax.add_patch(Rectangle((x, y), width, height, facecolor="none", edgecolor=border, lw=0.6, zorder=zorder + 4))


def plot_tournament_tree(knockout_matches: List, output_path: Path, simulation_log_probability=None) -> None:
    """
    Draw a two-sided World Cup bracket similar to broadcast tournament trees.

    The two halves follow the official match dependency tree, not chronological
    match-number order. This keeps every displayed advancement aligned with the
    real knockout slots.
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

    def short_team(team, limit=13):
        return team.upper() if len(team) <= limit else team[:limit - 1].upper() + "."

    def score_for(match, team):
        score = match.goals_a if team == match.team_a else match.goals_b
        suffix = "P" if match.decided_by != "90 minutes" and team == match.winner else ""
        return f"{score}{suffix}"

    def draw_text_box(x, y, label, fontsize=7.5, face=panel, edge=None, align="center"):
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

    def draw_match_card(x, y, match, side_name):
        card_w, card_h = 0.155, 0.060
        x0 = x - card_w / 2
        y0 = y - card_h / 2
        ax.add_patch(Rectangle((x0, y0), card_w, card_h, facecolor=panel, edgecolor=panel, lw=1.0, zorder=1))
        ax.text(x0 + card_w * 0.50, y0 + card_h * 0.82, f"M{match.match_number}  {match.date[5:]}",
                color=muted, fontsize=5.8, ha="center", va="center", fontweight="bold", zorder=3)

        rows = [(match.team_a, match.goals_a), (match.team_b, match.goals_b)]
        for idx, (team, goals) in enumerate(rows):
            yy = y0 + card_h * (0.58 if idx == 0 else 0.25)
            flag_x = x0 + (0.020 if side_name == "left" else card_w - 0.046)
            _draw_flag(ax, team, flag_x, yy - 0.010, 0.028, 0.020, zorder=4)
            if side_name == "left":
                ax.text(flag_x + 0.034, yy, short_team(team, 12), color=text, fontsize=6.2,
                        ha="left", va="center", fontweight="bold", zorder=4)
                ax.text(x0 + card_w - 0.012, yy, score_for(match, team), color=text, fontsize=6.4,
                        ha="right", va="center", fontweight="bold", zorder=4)
            else:
                ax.text(x0 + 0.012, yy, score_for(match, team), color=text, fontsize=6.4,
                        ha="left", va="center", fontweight="bold", zorder=4)
                ax.text(flag_x - 0.008, yy, short_team(team, 12), color=text, fontsize=6.2,
                        ha="right", va="center", fontweight="bold", zorder=4)

    def draw_winner_flag(x, y, match, scale=1.0):
        flag_w, flag_h = 0.040 * scale, 0.028 * scale
        _draw_flag(ax, match.winner, x - flag_w / 2, y - flag_h / 2, flag_w, flag_h, zorder=5)
        ax.text(x, y - flag_h * 0.90, f"{score_for(match, match.winner)}", color=text, fontsize=5.8 * scale,
                ha="center", va="center", fontweight="bold", zorder=6)

    def draw_connector(x1, y1, x2, y2, side):
        mid = (x1 + x2) / 2.0
        ax.plot([x1, mid, mid, x2], [y1, y1, y2, y2], color=line, lw=1.25, alpha=0.95)

    by_number = {match.match_number: match for match in knockout_matches}

    left_r32 = [by_number[number] for number in [74, 77, 73, 75, 83, 84, 81, 82]]
    left_r16 = [by_number[number] for number in [89, 90, 93, 94]]
    left_qf = [by_number[number] for number in [97, 98]]
    left_sf = [by_number[101]]

    right_r32 = [by_number[number] for number in [76, 78, 79, 80, 86, 88, 85, 87]]
    right_r16 = [by_number[number] for number in [91, 92, 95, 96]]
    right_qf = [by_number[number] for number in [99, 100]]
    right_sf = [by_number[102]]

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
            draw_match_card(loc["r32_x"], y, match, side_name)
        for match, y in zip(r16, r16_y):
            draw_winner_flag(loc["r16_x"], y, match)
        for match, y in zip(qf, qf_y):
            draw_winner_flag(loc["qf_x"], y, match)
        for match, y in zip(sf, sf_y):
            draw_winner_flag(loc["sf_x"], y, match)

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

    _draw_flag(ax, final.team_a, 0.455, 0.545, 0.045, 0.030, zorder=6)
    ax.text(0.505, 0.560, f"{short_team(final.team_a, 11)} {final.goals_a}", color=text, fontsize=8.0,
            ha="left", va="center", fontweight="bold")
    _draw_flag(ax, final.team_b, 0.455, 0.455, 0.045, 0.030, zorder=6)
    ax.text(0.505, 0.470, f"{short_team(final.team_b, 11)} {final.goals_b}", color=text, fontsize=8.0,
            ha="left", va="center", fontweight="bold")
    ax.plot([0.50, 0.50], [0.535, 0.495], color=line, lw=1.25)

    ax.text(0.50, 0.37, "WINNERS", color=muted, fontsize=10, ha="center", va="center", fontweight="bold")
    ax.add_patch(Rectangle((0.43, 0.275), 0.14, 0.070, facecolor=winner_fill, edgecolor=gold, lw=1.8, zorder=3))
    _draw_flag(ax, champion, 0.445, 0.291, 0.048, 0.034, zorder=5)
    ax.text(0.505, 0.310, short_team(champion, 10), ha="left", va="center", color=text,
            fontsize=18, fontweight="bold", zorder=6)
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
