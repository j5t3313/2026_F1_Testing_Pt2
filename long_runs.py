import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import LONG_RUN_MIN_LAPS
from plotting import (
    apply_theme, create_figure, build_color_maps,
    get_compound_color, add_watermark, save_figure,
)


def identify_long_runs(laps, min_laps=None):
    threshold = min_laps or LONG_RUN_MIN_LAPS

    valid = laps.dropna(subset=["LapTimeSeconds"]).copy()
    valid = valid[valid["LapTimeSeconds"] > 0]

    stints = (
        valid.groupby(["Team", "Driver", "Day", "Stint"])
        .agg(
            StintLaps=("LapTimeSeconds", "count"),
            Compound=("Compound", "first"),
            MeanTime=("LapTimeSeconds", "mean"),
            StdTime=("LapTimeSeconds", "std"),
            MinTime=("LapTimeSeconds", "min"),
            MaxTime=("LapTimeSeconds", "max"),
        )
        .reset_index()
    )

    long_runs = stints[stints["StintLaps"] >= threshold].copy()
    long_runs["CoV"] = long_runs["StdTime"] / long_runs["MeanTime"]
    long_runs["Range"] = long_runs["MaxTime"] - long_runs["MinTime"]

    return long_runs


def get_long_run_laps(laps, long_runs):
    result = []
    for _, run in long_runs.iterrows():
        mask = (
            (laps["Team"] == run["Team"])
            & (laps["Driver"] == run["Driver"])
            & (laps["Day"] == run["Day"])
            & (laps["Stint"] == run["Stint"])
            & (laps["LapTimeSeconds"].notna())
            & (laps["LapTimeSeconds"] > 0)
        )
        stint_laps = laps[mask].copy()
        stint_laps = stint_laps.sort_values("LapNumber")
        stint_laps["StintLapNumber"] = range(1, len(stint_laps) + 1)
        stint_laps["DeltaFromMean"] = stint_laps["LapTimeSeconds"] - stint_laps["LapTimeSeconds"].mean()
        result.append(stint_laps)

    if not result:
        return pd.DataFrame()
    return pd.concat(result, ignore_index=True)


def compute_consistency_by_team(long_runs):
    return (
        long_runs.groupby("Team")
        .agg(
            MeanCoV=("CoV", "mean"),
            MedianCoV=("CoV", "median"),
            NumLongRuns=("CoV", "count"),
            MeanRange=("Range", "mean"),
        )
        .reset_index()
        .sort_values("MedianCoV")
    )


def plot_long_run_traces(laps, long_runs):
    apply_theme()
    team_colors, driver_colors = build_color_maps(laps)
    run_laps = get_long_run_laps(laps, long_runs)

    if run_laps.empty:
        return None

    fig, ax = create_figure(width=14, height=8)

    run_keys = run_laps.groupby(["Team", "Driver", "Day", "Stint"]).ngroups
    for (team, driver, day, stint), group in run_laps.groupby(["Team", "Driver", "Day", "Stint"]):
        color = driver_colors.get(driver, team_colors.get(team, "#888888"))
        ax.plot(
            group["StintLapNumber"], group["DeltaFromMean"],
            color=color, alpha=0.5, linewidth=1.2,
        )

    ax.axhline(y=0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lap Within Stint")
    ax.set_ylabel("Delta from Stint Mean (seconds)")
    ax.set_title(f"Long Run Lap Time Traces (stints ≥ {LONG_RUN_MIN_LAPS} laps)")

    teams_in_data = run_laps["Team"].unique()
    handles = [
        plt.Line2D([0], [0], color=team_colors.get(t, "#888888"), linewidth=2, label=t)
        for t in sorted(teams_in_data)
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=9)

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_consistency_rankings(long_runs):
    apply_theme()
    consistency = compute_consistency_by_team(long_runs)

    if consistency.empty:
        return None

    from plotting import build_color_maps
    import matplotlib.patches as mpatches

    fig, ax = create_figure(width=12, height=7)

    colors = []
    for team in consistency["Team"]:
        from config import TEAM_COLORS, FALLBACK_COLOR
        colors.append(TEAM_COLORS.get(team, FALLBACK_COLOR))

    bars = ax.barh(
        consistency["Team"], consistency["MedianCoV"] * 100,
        color=colors, edgecolor="white",
    )

    for bar, count in zip(bars, consistency["NumLongRuns"]):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"n={count}", va="center", fontsize=10, color="#666666",
        )

    ax.set_xlabel("Median Coefficient of Variation (%)")
    ax.set_title("Long Run Consistency by Team (lower = more consistent)")
    ax.invert_yaxis()

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_long_runs_by_compound(laps, long_runs):
    apply_theme()
    team_colors, driver_colors = build_color_maps(laps)
    run_laps = get_long_run_laps(laps, long_runs)

    if run_laps.empty:
        return None

    compounds = sorted(run_laps["Compound"].dropna().unique())
    n = len(compounds)
    if n == 0:
        return None

    fig, axes = create_figure(width=14, height=5 * n, nrows=n)
    if n == 1:
        axes = [axes]

    for idx, compound in enumerate(compounds):
        ax = axes[idx]
        compound_data = run_laps[run_laps["Compound"] == compound]

        for (team, driver, day, stint), group in compound_data.groupby(
            ["Team", "Driver", "Day", "Stint"]
        ):
            color = driver_colors.get(driver, team_colors.get(team, "#888888"))
            ax.plot(
                group["StintLapNumber"], group["DeltaFromMean"],
                color=color, alpha=0.5, linewidth=1.2,
            )

        ax.axhline(y=0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Lap Within Stint")
        ax.set_ylabel("Delta from Stint Mean (s)")
        compound_color = get_compound_color(compound)
        ax.set_title(f"{compound}", color=compound_color, fontweight="bold")

    fig.suptitle(
        f"Long Run Traces by Compound (stints ≥ {LONG_RUN_MIN_LAPS} laps)",
        fontsize=16, fontweight="bold", y=1.01,
    )
    add_watermark(fig)
    fig.tight_layout()
    return fig


def generate_all(laps):
    long_runs = identify_long_runs(laps)
    figures = {}
    figures["long_run_traces"] = plot_long_run_traces(laps, long_runs)
    figures["consistency_rankings"] = plot_consistency_rankings(long_runs)
    figures["long_runs_by_compound"] = plot_long_runs_by_compound(laps, long_runs)
    return figures


def plot_consistency_comparison(laps_w1, laps_w2):
    apply_theme()
    from config import TEAM_COLORS, FALLBACK_COLOR

    lr_w1 = identify_long_runs(laps_w1)
    lr_w2 = identify_long_runs(laps_w2)

    con_w1 = compute_consistency_by_team(lr_w1).set_index("Team")["MedianCoV"].rename("W1")
    con_w2 = compute_consistency_by_team(lr_w2).set_index("Team")["MedianCoV"].rename("W2")
    merged = pd.concat([con_w1, con_w2], axis=1).dropna()
    merged = merged.sort_values("W2")

    fig, ax = create_figure(width=14, height=8)

    y = np.arange(len(merged))
    bar_height = 0.35

    for i, team in enumerate(merged.index):
        color = TEAM_COLORS.get(team, FALLBACK_COLOR)
        ax.barh(i - bar_height / 2, merged.loc[team, "W1"] * 100,
                height=bar_height, color=color, alpha=0.35, edgecolor=color)
        ax.barh(i + bar_height / 2, merged.loc[team, "W2"] * 100,
                height=bar_height, color=color, alpha=0.9, edgecolor=color)

    ax.set_yticks(y)
    ax.set_yticklabels(merged.index)
    ax.set_xlabel("Median Coefficient of Variation (%)")
    ax.set_title("Long Run Consistency: Week 1 (faded) vs Week 2 (solid)")
    ax.invert_yaxis()

    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(facecolor="#888888", alpha=0.35, label="Week 1"),
        mpatches.Patch(facecolor="#888888", alpha=0.9, label="Week 2"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_pace_delta(laps_w1, laps_w2):
    apply_theme()
    from config import TEAM_COLORS, FALLBACK_COLOR
    from calibration import compute_long_run_pace

    pace_w1 = compute_long_run_pace(laps_w1).set_index("Team")["MeanLongRunPace"].rename("W1")
    pace_w2 = compute_long_run_pace(laps_w2).set_index("Team")["MeanLongRunPace"].rename("W2")
    merged = pd.concat([pace_w1, pace_w2], axis=1).dropna()
    merged["Delta"] = merged["W2"] - merged["W1"]
    merged = merged.sort_values("Delta")

    fig, ax = create_figure(width=12, height=7)

    colors = [TEAM_COLORS.get(t, FALLBACK_COLOR) for t in merged.index]
    bars = ax.barh(merged.index, merged["Delta"], color=colors, edgecolor="white")

    ax.axvline(x=0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Long Run Pace Change (seconds, negative = faster)")
    ax.set_title("Long Run Pace: Week 2 vs Week 1")
    ax.invert_yaxis()

    for bar, val in zip(bars, merged["Delta"]):
        offset = 0.03 if val >= 0 else -0.03
        ha = "left" if val >= 0 else "right"
        ax.text(
            bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}s", va="center", ha=ha, fontsize=10, fontweight="bold",
        )

    add_watermark(fig)
    fig.tight_layout()
    return fig


def generate_week_comparison(laps_w1, laps_w2):
    figures = {}
    figures["consistency_comparison"] = plot_consistency_comparison(laps_w1, laps_w2)
    figures["pace_delta"] = plot_pace_delta(laps_w1, laps_w2)
    return figures
