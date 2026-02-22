import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from config import CACHE_DIR, OUTPUT_DIR, FIGURE_DPI

COLOR_2026 = "#E8002D"
COLOR_BLUE = "#2166AC"


def setup():
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def apply_theme():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.5,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "sans-serif",
        "figure.dpi": FIGURE_DPI,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
    })


def add_watermark(fig):
    fig.text(0.99, 0.01, "@formulasteele", fontsize=9, color="#AAAAAA",
             ha="right", va="bottom", transform=fig.transFigure)


def load_2026_sessions():
    sessions = []
    for day in [1, 2, 3]:
        s = fastf1.get_testing_session(2026, 2, day)
        s.load(telemetry=True, weather=False)
        sessions.append(s)
    return sessions


def load_2025_sessions():
    sessions = []
    for day in [1, 2, 3]:
        s = fastf1.get_testing_session(2025, 1, day)
        s.load(telemetry=True, weather=False)
        sessions.append(s)
    return sessions


def load_2022_sessions():
    sessions = []
    for day in [1, 2, 3]:
        s = fastf1.get_testing_session(2022, 2, day)
        s.load(telemetry=True, weather=False)
        sessions.append(s)
    return sessions


def get_fastest_lap(sessions):
    for session in sessions:
        laps = session.laps
        soft = laps[laps["Compound"] == "SOFT"]
        if not soft.empty:
            fastest = soft.pick_fastest()
            if fastest is not None and not fastest.empty:
                return fastest
    for session in sessions:
        fastest = session.laps.pick_fastest()
        if fastest is not None and not fastest.empty:
            return fastest
    return None


def extract_and_interpolate(lap, n_points=1000):
    tel = lap.get_telemetry()
    if tel is None or tel.empty:
        return None

    dist = tel["Distance"].values
    common = np.linspace(dist.min(), dist.max(), n_points)
    result = pd.DataFrame({
        "Distance": common,
        "Speed": np.interp(common, dist, tel["Speed"].values),
    })

    for col in ["Throttle", "Brake", "nGear"]:
        if col in tel.columns:
            vals = tel[col].values.astype(float)
            result[col] = np.interp(common, dist, vals)

    return result


def plot_stacked_speed_comparison(tel_2026, tel_2025, tel_2022):
    apply_theme()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig.set_facecolor("white")

    ax1.plot(tel_2025["Distance"], tel_2025["Speed"],
             color=COLOR_BLUE, linewidth=1.5, alpha=0.8, label="2025 (Best)")
    ax1.plot(tel_2026["Distance"], tel_2026["Speed"],
             color=COLOR_2026, linewidth=1.8, alpha=0.9, label="2026 (Best)")
    ax1.fill_between(tel_2026["Distance"], tel_2025["Speed"], tel_2026["Speed"],
                     where=tel_2026["Speed"] > tel_2025["Speed"],
                     alpha=0.15, color=COLOR_2026, label="2026 faster")
    ax1.fill_between(tel_2026["Distance"], tel_2025["Speed"], tel_2026["Speed"],
                     where=tel_2026["Speed"] < tel_2025["Speed"],
                     alpha=0.15, color=COLOR_BLUE, label="2025 faster")
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title("Speed Trace: 2026 vs 2025 Pre-Season Testing")
    ax1.legend(loc="lower right")

    ax2.plot(tel_2022["Distance"], tel_2022["Speed"],
             color=COLOR_BLUE, linewidth=1.5, alpha=0.8, label="2022 (Best)")
    ax2.plot(tel_2026["Distance"], tel_2026["Speed"],
             color=COLOR_2026, linewidth=1.8, alpha=0.9, label="2026 (Best)")
    ax2.fill_between(tel_2026["Distance"], tel_2022["Speed"], tel_2026["Speed"],
                     where=tel_2026["Speed"] > tel_2022["Speed"],
                     alpha=0.15, color=COLOR_2026, label="2026 faster")
    ax2.fill_between(tel_2026["Distance"], tel_2022["Speed"], tel_2026["Speed"],
                     where=tel_2026["Speed"] < tel_2022["Speed"],
                     alpha=0.15, color=COLOR_BLUE, label="2022 faster")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Speed (km/h)")
    ax2.set_title("Speed Trace: 2026 vs 2022 Pre-Season Testing")
    ax2.legend(loc="lower right")

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_stacked_speed_delta(tel_2026, tel_2025, tel_2022):
    apply_theme()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.set_facecolor("white")

    delta_25 = tel_2026["Speed"].values - tel_2025["Speed"].values
    dist = tel_2026["Distance"].values

    ax1.fill_between(dist, 0, delta_25, where=delta_25 >= 0,
                     alpha=0.6, color=COLOR_2026, label="2026 faster")
    ax1.fill_between(dist, 0, delta_25, where=delta_25 < 0,
                     alpha=0.6, color=COLOR_BLUE, label="2025 faster")
    ax1.axhline(y=0, color="#333333", linewidth=0.8)
    ax1.set_ylabel("Speed Delta (km/h)")
    ax1.set_title("Speed Advantage: 2026 vs 2025")
    ax1.legend(loc="upper right")

    delta_22 = tel_2026["Speed"].values - tel_2022["Speed"].values

    ax2.fill_between(dist, 0, delta_22, where=delta_22 >= 0,
                     alpha=0.6, color=COLOR_2026, label="2026 faster")
    ax2.fill_between(dist, 0, delta_22, where=delta_22 < 0,
                     alpha=0.6, color=COLOR_BLUE, label="2022 faster")
    ax2.axhline(y=0, color="#333333", linewidth=0.8)
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Speed Delta (km/h)")
    ax2.set_title("Speed Advantage: 2026 vs 2022")
    ax2.legend(loc="upper right")

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_full_telemetry(tel_2026, tel_2022):
    apply_theme()

    has_throttle = "Throttle" in tel_2026.columns and "Throttle" in tel_2022.columns
    has_brake = "Brake" in tel_2026.columns and "Brake" in tel_2022.columns
    has_gear = "nGear" in tel_2026.columns and "nGear" in tel_2022.columns

    n_rows = 1 + int(has_throttle) + int(has_brake) + int(has_gear)
    height_ratios = [3] + [1] * (n_rows - 1)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(16, 4 + 2.5 * n_rows),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    fig.set_facecolor("white")

    row = 0

    axes[row].plot(tel_2022["Distance"], tel_2022["Speed"],
                   color=COLOR_BLUE, linewidth=1.5, alpha=0.8, label="2022 (Best)")
    axes[row].plot(tel_2026["Distance"], tel_2026["Speed"],
                   color=COLOR_2026, linewidth=1.8, alpha=0.9, label="2026 (Best)")
    axes[row].set_ylabel("Speed (km/h)")
    axes[row].set_title("Speed Trace: 2026 vs 2022 Bahrain Testing")
    axes[row].legend(loc="lower right")
    row += 1

    if has_throttle:
        axes[row].plot(tel_2022["Distance"], tel_2022["Throttle"],
                       color=COLOR_BLUE, linewidth=1.2, alpha=0.8)
        axes[row].plot(tel_2026["Distance"], tel_2026["Throttle"],
                       color=COLOR_2026, linewidth=1.5, alpha=0.9)
        axes[row].set_ylabel("Throttle %")
        axes[row].set_ylim(-5, 105)
        row += 1

    if has_brake:
        axes[row].plot(tel_2022["Distance"], tel_2022["Brake"],
                       color=COLOR_BLUE, linewidth=1.2, alpha=0.8)
        axes[row].plot(tel_2026["Distance"], tel_2026["Brake"],
                       color=COLOR_2026, linewidth=1.5, alpha=0.9)
        axes[row].set_ylabel("Brake")
        axes[row].set_yticks([0, 1])
        axes[row].set_yticklabels(["Off", "On"])
        axes[row].set_ylim(-0.1, 1.1)
        row += 1

    if has_gear:
        axes[row].plot(tel_2022["Distance"], tel_2022["nGear"],
                       color=COLOR_BLUE, linewidth=1.2, alpha=0.8)
        axes[row].plot(tel_2026["Distance"], tel_2026["nGear"],
                       color=COLOR_2026, linewidth=1.5, alpha=0.9)
        axes[row].set_ylabel("Gear")
        row += 1

    axes[-1].set_xlabel("Distance (m)")

    add_watermark(fig)
    fig.tight_layout()
    return fig


def save(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def run():
    setup()

    print("Loading 2026 W2...")
    sessions_2026 = load_2026_sessions()

    print("Loading 2025...")
    sessions_2025 = load_2025_sessions()

    print("Loading 2022...")
    sessions_2022 = load_2022_sessions()

    lap_2026 = get_fastest_lap(sessions_2026)
    lap_2025 = get_fastest_lap(sessions_2025)
    lap_2022 = get_fastest_lap(sessions_2022)

    tel_2026 = extract_and_interpolate(lap_2026)
    tel_2025 = extract_and_interpolate(lap_2025)
    tel_2022 = extract_and_interpolate(lap_2022)

    print("Generating charts:")

    fig = plot_stacked_speed_comparison(tel_2026, tel_2025, tel_2022)
    save(fig, "speed_comparison_stacked.png")

    fig = plot_stacked_speed_delta(tel_2026, tel_2025, tel_2022)
    save(fig, "speed_delta_stacked.png")

    fig = plot_full_telemetry(tel_2026, tel_2022)
    save(fig, "full_telemetry_2022_vs_2026.png")

    print("Done.")


if __name__ == "__main__":
    run()