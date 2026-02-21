import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting import (
    apply_theme, create_figure, build_color_maps,
    get_compound_color, add_watermark, save_figure,
)


def compute_team_stats(laps):
    stats = (
        laps.groupby("Team")["LapTimeSeconds"]
        .agg(["min", "median", "mean", "std", "count"])
        .reset_index()
    )
    stats["headline_gap"] = stats["median"] - stats["min"]
    return stats.sort_values("median")


def plot_team_boxes(laps):
    apply_theme()
    team_colors, _ = build_color_maps(laps)

    teams_ordered = (
        laps.groupby("Team")["LapTimeSeconds"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, ax = create_figure(width=14, height=8)

    for i, team in enumerate(teams_ordered):
        team_data = laps[laps["Team"] == team]["LapTimeSeconds"].dropna()
        if team_data.empty:
            continue

        color = team_colors.get(team, "#888888")
        bp = ax.boxplot(
            team_data, positions=[i], widths=0.5,
            showfliers=False, patch_artist=True,
            medianprops=dict(color="#333333", linewidth=2),
            whiskerprops=dict(color=color, linewidth=1.5),
            capprops=dict(color=color, linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][0].set_edgecolor(color)

    ax.set_xticks(range(len(teams_ordered)))
    ax.set_xticklabels(teams_ordered, rotation=45, ha="right")
    ax.set_ylabel("Lap Time (seconds)")
    ax.set_title("Lap Time Distributions by Team")

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_compound_distributions(laps):
    apply_theme()
    team_colors, _ = build_color_maps(laps)
    compounds = [c for c in ["SOFT", "MEDIUM", "HARD"] if c in laps["Compound"].unique()]

    if not compounds:
        compounds = laps["Compound"].dropna().unique().tolist()

    n_compounds = len(compounds)
    if n_compounds == 0:
        return None

    fig, axes = create_figure(width=14, height=5 * n_compounds, nrows=n_compounds)
    if n_compounds == 1:
        axes = [axes]

    for idx, compound in enumerate(compounds):
        ax = axes[idx]
        compound_laps = laps[laps["Compound"] == compound]

        teams_ordered = (
            compound_laps.groupby("Team")["LapTimeSeconds"]
            .median()
            .sort_values()
            .index.tolist()
        )

        for i, team in enumerate(teams_ordered):
            team_data = compound_laps[compound_laps["Team"] == team]["LapTimeSeconds"].dropna()
            if team_data.empty:
                continue

            color = team_colors.get(team, "#888888")
            bp = ax.boxplot(
                team_data, positions=[i], widths=0.5,
                showfliers=False, patch_artist=True,
                medianprops=dict(color="#333333", linewidth=2),
                whiskerprops=dict(color=color, linewidth=1.5),
                capprops=dict(color=color, linewidth=1.5),
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][0].set_edgecolor(color)

        ax.set_xticks(range(len(teams_ordered)))
        ax.set_xticklabels(teams_ordered, rotation=45, ha="right")
        ax.set_ylabel("Lap Time (seconds)")

        compound_color = get_compound_color(compound)
        ax.set_title(f"{compound} Compound", color=compound_color, fontweight="bold")

    fig.suptitle("Lap Time Distributions by Compound", fontsize=18, fontweight="bold", y=1.01)
    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_headline_vs_median(laps):
    apply_theme()
    team_colors, _ = build_color_maps(laps)
    stats = compute_team_stats(laps)

    fig, ax = create_figure(width=12, height=7)

    for _, row in stats.iterrows():
        color = team_colors.get(row["Team"], "#888888")
        ax.scatter(
            row["headline_gap"], row["median"],
            s=row["count"] * 2, color=color,
            edgecolors="#333333", linewidth=0.5, zorder=5,
        )
        ax.annotate(
            row["Team"], (row["headline_gap"], row["median"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=9,
        )

    ax.set_xlabel("Gap: Median to Fastest Lap (seconds)")
    ax.set_ylabel("Median Lap Time (seconds)")
    ax.set_title("Program Focus: Headline Time vs Typical Running Pace")
    ax.invert_yaxis()

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_headline_vs_median_weekover(laps_w1, laps_w2):
    apply_theme()
    team_colors, _ = build_color_maps(pd.concat([laps_w1, laps_w2]))
    stats_w1 = compute_team_stats(laps_w1)
    stats_w2 = compute_team_stats(laps_w2)

    fig, ax = create_figure(width=12, height=7)

    for _, row in stats_w1.iterrows():
        color = team_colors.get(row["Team"], "#888888")
        ax.scatter(
            row["headline_gap"], row["median"],
            s=row["count"] * 1.5, color=color, alpha=0.25,
            edgecolors=color, linewidth=0.5, zorder=3,
        )

    for _, row in stats_w2.iterrows():
        color = team_colors.get(row["Team"], "#888888")
        ax.scatter(
            row["headline_gap"], row["median"],
            s=row["count"] * 2, color=color,
            edgecolors="#333333", linewidth=0.5, zorder=5,
        )
        ax.annotate(
            row["Team"], (row["headline_gap"], row["median"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=9,
        )

        w1_match = stats_w1[stats_w1["Team"] == row["Team"]]
        if not w1_match.empty:
            w1 = w1_match.iloc[0]
            ax.annotate(
                "",
                xy=(row["headline_gap"], row["median"]),
                xytext=(w1["headline_gap"], w1["median"]),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.4, lw=1.5),
            )

    ax.set_xlabel("Gap: Median to Fastest Lap (seconds)")
    ax.set_ylabel("Median Lap Time (seconds)")
    ax.set_title("Program Focus: Week 1 (faded) vs Week 2 (solid)")
    ax.invert_yaxis()

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_median_shift(laps_w1, laps_w2):
    apply_theme()
    from config import TEAM_COLORS, FALLBACK_COLOR

    median_w1 = laps_w1.groupby("Team")["LapTimeSeconds"].median().rename("W1")
    median_w2 = laps_w2.groupby("Team")["LapTimeSeconds"].median().rename("W2")
    merged = pd.concat([median_w1, median_w2], axis=1).dropna()
    merged["Delta"] = merged["W2"] - merged["W1"]
    merged = merged.sort_values("Delta")

    fig, ax = create_figure(width=12, height=7)

    colors = [TEAM_COLORS.get(t, FALLBACK_COLOR) for t in merged.index]
    bars = ax.barh(merged.index, merged["Delta"], color=colors, edgecolor="white")

    ax.axvline(x=0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Median Lap Time Change (seconds, negative = faster)")
    ax.set_title("Median Running Pace: Week 2 vs Week 1")
    ax.invert_yaxis()

    for bar, val in zip(bars, merged["Delta"]):
        offset = 0.05 if val >= 0 else -0.05
        ha = "left" if val >= 0 else "right"
        ax.text(
            bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}s", va="center", ha=ha, fontsize=10, fontweight="bold",
        )

    add_watermark(fig)
    fig.tight_layout()
    return fig


def generate_all(laps):
    figures = {}
    figures["team_boxes"] = plot_team_boxes(laps)
    figures["compound_distributions"] = plot_compound_distributions(laps)
    figures["headline_vs_median"] = plot_headline_vs_median(laps)
    return figures


def generate_week_comparison(laps_w1, laps_w2):
    figures = {}
    figures["headline_weekover"] = plot_headline_vs_median_weekover(laps_w1, laps_w2)
    figures["median_shift"] = plot_median_shift(laps_w1, laps_w2)
    return figures
