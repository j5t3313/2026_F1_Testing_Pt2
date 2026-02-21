import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plotting import (
    apply_theme, create_figure, build_color_maps,
    add_watermark, save_figure,
)


def compute_laps_per_team_day(laps):
    return (
        laps.groupby(["Team", "Day"])
        .size()
        .reset_index(name="Laps")
        .pivot(index="Team", columns="Day", values="Laps")
        .fillna(0)
        .astype(int)
    )


def compute_total_laps(laps):
    totals = laps.groupby("Team").size().reset_index(name="TotalLaps")
    return totals.sort_values("TotalLaps", ascending=False)


def compute_stint_summary(laps):
    stints = (
        laps.groupby(["Team", "Driver", "Day", "Stint"])
        .agg(StintLaps=("LapNumber", "count"))
        .reset_index()
    )

    summary = (
        stints.groupby("Team")
        .agg(
            TotalStints=("Stint", "count"),
            MaxStintLength=("StintLaps", "max"),
            MeanStintLength=("StintLaps", "mean"),
        )
        .reset_index()
        .sort_values("TotalStints", ascending=False)
    )
    return summary


def compute_laps_per_driver(laps):
    return (
        laps.groupby(["Team", "Driver"])
        .size()
        .reset_index(name="Laps")
        .sort_values(["Team", "Laps"], ascending=[True, False])
    )


def plot_laps_heatmap(laps):
    apply_theme()
    grid = compute_laps_per_team_day(laps)
    grid["Total"] = grid.sum(axis=1)
    grid = grid.sort_values("Total", ascending=True)
    display = grid.drop(columns="Total")

    fig, ax = create_figure(width=12, height=8)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#F5F5F5", "#2166AC"])
    im = ax.imshow(display.values, cmap=cmap, aspect="auto")

    ax.set_yticks(range(len(display.index)))
    ax.set_yticklabels(display.index)
    ax.set_xticks(range(len(display.columns)))
    ax.set_xticklabels([f"Day {c}" for c in display.columns])

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val = display.iloc[i, j]
            text_color = "white" if val > display.values.max() * 0.6 else "#333333"
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=13, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Laps Completed")
    ax.set_title("Program Maturity: Laps Completed Per Day")
    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_total_laps_bar(laps):
    apply_theme()
    team_colors, _ = build_color_maps(laps)
    totals = compute_total_laps(laps)

    fig, ax = create_figure(width=12, height=7)
    colors = [team_colors.get(t, "#888888") for t in totals["Team"]]
    bars = ax.barh(totals["Team"], totals["TotalLaps"], color=colors, edgecolor="white")

    for bar, val in zip(bars, totals["TotalLaps"]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Total Laps")
    ax.set_title("Total Laps Completed - Week 2")
    ax.invert_yaxis()
    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_stint_lengths(laps):
    apply_theme()
    team_colors, _ = build_color_maps(laps)
    summary = compute_stint_summary(laps)

    fig, axes = create_figure(width=14, height=6, ncols=2)

    summary_sorted = summary.sort_values("MaxStintLength", ascending=True)
    colors = [team_colors.get(t, "#888888") for t in summary_sorted["Team"]]
    axes[0].barh(summary_sorted["Team"], summary_sorted["MaxStintLength"], color=colors)
    axes[0].set_xlabel("Laps")
    axes[0].set_title("Longest Single Stint")

    summary_sorted2 = summary.sort_values("TotalStints", ascending=True)
    colors2 = [team_colors.get(t, "#888888") for t in summary_sorted2["Team"]]
    axes[1].barh(summary_sorted2["Team"], summary_sorted2["TotalStints"], color=colors2)
    axes[1].set_xlabel("Stints")
    axes[1].set_title("Total Stint Count")

    add_watermark(fig)
    fig.tight_layout()
    return fig


def generate_all(laps):
    figures = {}
    figures["laps_heatmap"] = plot_laps_heatmap(laps)
    figures["total_laps"] = plot_total_laps_bar(laps)
    figures["stint_lengths"] = plot_stint_lengths(laps)
    return figures


def plot_combined_heatmap(laps_w1, laps_w2):
    apply_theme()

    laps_w1 = laps_w1.copy()
    laps_w2 = laps_w2.copy()
    laps_w1["Session"] = "W1D" + laps_w1["Day"].astype(str)
    laps_w2["Session"] = "W2D" + laps_w2["Day"].astype(str)
    combined = pd.concat([laps_w1, laps_w2], ignore_index=True)

    grid = (
        combined.groupby(["Team", "Session"])
        .size()
        .reset_index(name="Laps")
        .pivot(index="Team", columns="Session", values="Laps")
        .fillna(0)
        .astype(int)
    )

    col_order = ["W1D1", "W1D2", "W1D3", "W2D1", "W2D2", "W2D3"]
    col_order = [c for c in col_order if c in grid.columns]
    grid = grid[col_order]

    grid["Total"] = grid.sum(axis=1)
    grid = grid.sort_values("Total", ascending=True)
    display = grid.drop(columns="Total")

    fig, ax = create_figure(width=14, height=9)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#F5F5F5", "#2166AC"])
    im = ax.imshow(display.values, cmap=cmap, aspect="auto")

    ax.set_yticks(range(len(display.index)))
    ax.set_yticklabels(display.index)
    ax.set_xticks(range(len(display.columns)))
    ax.set_xticklabels(display.columns, fontsize=11)

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val = display.iloc[i, j]
            text_color = "white" if val > display.values.max() * 0.6 else "#333333"
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    if len(col_order) == 6:
        ax.axvline(x=2.5, color="#333333", linewidth=2, linestyle="-")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Laps Completed")
    ax.set_title("Program Maturity: Laps Completed Across Both Test Weeks")
    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_lap_delta(laps_w1, laps_w2):
    apply_theme()

    totals_w1 = laps_w1.groupby("Team").size().rename("W1")
    totals_w2 = laps_w2.groupby("Team").size().rename("W2")
    merged = pd.concat([totals_w1, totals_w2], axis=1).fillna(0).astype(int)
    merged["Delta"] = merged["W2"] - merged["W1"]
    merged = merged.sort_values("Delta")

    fig, ax = create_figure(width=12, height=7)

    from config import TEAM_COLORS, FALLBACK_COLOR
    colors = []
    for team in merged.index:
        colors.append(TEAM_COLORS.get(team, FALLBACK_COLOR))

    bars = ax.barh(merged.index, merged["Delta"], color=colors, edgecolor="white")
    ax.axvline(x=0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Lap Count Change (Week 2 - Week 1)")
    ax.set_title("Week-over-Week Lap Count Delta")
    ax.invert_yaxis()

    for bar, val in zip(bars, merged["Delta"]):
        offset = 2 if val >= 0 else -2
        ha = "left" if val >= 0 else "right"
        ax.text(
            bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+d}", va="center", ha=ha, fontsize=10, fontweight="bold",
        )

    add_watermark(fig)
    fig.tight_layout()
    return fig


def generate_week_comparison(laps_w1, laps_w2):
    figures = {}
    figures["combined_heatmap"] = plot_combined_heatmap(laps_w1, laps_w2)
    figures["lap_delta"] = plot_lap_delta(laps_w1, laps_w2)
    return figures
