from pathlib import Path
from config import OUTPUT_DIR
from data_loader import setup, load_2026_w1, load_2026_w2, load_2025, get_clean_laps
from plotting import apply_theme, save_figure
import reliability
import distributions
import long_runs
import speed_traces
import calibration


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)
    setup()
    apply_theme()

    print("Loading 2026 Week 1 data...")
    sessions_w1, laps_w1 = load_2026_w1()
    clean_w1 = get_clean_laps(laps_w1)
    print(f"  {len(laps_w1)} total laps, {len(clean_w1)} after filtering")

    print("Loading 2026 Week 2 data...")
    sessions_w2, laps_w2 = load_2026_w2()
    clean_w2 = get_clean_laps(laps_w2)
    print(f"  {len(laps_w2)} total laps, {len(clean_w2)} after filtering")

    print("Loading 2025 baseline data...")
    sessions_2025, laps_2025 = load_2025()
    clean_2025 = get_clean_laps(laps_2025)
    print(f"  {len(laps_2025)} total laps, {len(clean_2025)} after filtering")

    print("\n--- Module 1: Reliability & Program Maturity (Week 2) ---")
    rel_figs = reliability.generate_all(laps_w2)
    for name, fig in rel_figs.items():
        if fig is not None:
            path = save_figure(fig, f"w2_reliability_{name}.png")
            print(f"  Saved: {path}")

    print("\n--- Module 1b: Reliability Week-over-Week ---")
    rel_comp_figs = reliability.generate_week_comparison(laps_w1, laps_w2)
    for name, fig in rel_comp_figs.items():
        if fig is not None:
            path = save_figure(fig, f"compare_reliability_{name}.png")
            print(f"  Saved: {path}")

    print("\n--- Module 2: Lap Time Distributions (Week 2) ---")
    dist_figs = distributions.generate_all(clean_w2)
    for name, fig in dist_figs.items():
        if fig is not None:
            path = save_figure(fig, f"w2_distributions_{name}.png")
            print(f"  Saved: {path}")

    print("\n--- Module 2b: Distributions Week-over-Week ---")
    dist_comp_figs = distributions.generate_week_comparison(clean_w1, clean_w2)
    for name, fig in dist_comp_figs.items():
        if fig is not None:
            path = save_figure(fig, f"compare_distributions_{name}.png")
            print(f"  Saved: {path}")

    print("\n--- Module 3: Long Run Consistency (Week 2) ---")
    lr_figs = long_runs.generate_all(clean_w2)
    for name, fig in lr_figs.items():
        if fig is not None:
            path = save_figure(fig, f"w2_long_runs_{name}.png")
            print(f"  Saved: {path}")

    print("\n--- Module 3b: Long Runs Week-over-Week ---")
    lr_comp_figs = long_runs.generate_week_comparison(clean_w1, clean_w2)
    for name, fig in lr_comp_figs.items():
        if fig is not None:
            path = save_figure(fig, f"compare_long_runs_{name}.png")
            print(f"  Saved: {path}")

    print("\n--- Module 4: Speed Traces (2026 W2 vs 2025) ---")
    st_figs = speed_traces.generate_speed_traces(sessions_w2, sessions_2025)
    for name, fig in st_figs.items():
        if fig is not None:
            path = save_figure(fig, f"w2_speed_traces_{name}.png")
            print(f"  Saved: {path}")

    print("\n--- Module 5: Calibration (Week 2 standalone) ---")
    cal_result = calibration.generate_all(clean_2025, clean_w2)
    if cal_result is not None and not isinstance(cal_result, dict):
        cal_figs, pace_2025, pace_w2, comparison = cal_result
        for name, fig in cal_figs.items():
            if fig is not None:
                path = save_figure(fig, f"w2_calibration_{name}.png")
                print(f"  Saved: {path}")

        print("\n  2026 Week 2 Long Run Pace:")
        print(pace_w2[["Team", "MeanLongRunPace", "DeltaToLeader", "TestingRank", "NumLongRuns"]].to_string(index=False))

    print("\n--- Module 5b: Calibration Week-over-Week ---")
    cal_week_result = calibration.generate_week_comparison(clean_2025, clean_w1, clean_w2)
    if cal_week_result:
        cal_week_figs, pace_w1, pace_w2_cal, week_comp = cal_week_result
        for name, fig in cal_week_figs.items():
            if fig is not None:
                path = save_figure(fig, f"compare_calibration_{name}.png")
                print(f"  Saved: {path}")

        print("\n  Week 1 Long Run Pace:")
        print(pace_w1[["Team", "MeanLongRunPace", "DeltaToLeader", "TestingRank", "NumLongRuns"]].to_string(index=False))

        print("\n  Week 2 Long Run Pace:")
        print(pace_w2_cal[["Team", "MeanLongRunPace", "DeltaToLeader", "TestingRank", "NumLongRuns"]].to_string(index=False))

        if not week_comp.empty:
            print("\n  Week Comparison Table:")
            display_cols = [
                "Team_2026", "WCC_Finish_2025",
                "Testing_Delta_W1", "Testing_Rank_W1",
                "Testing_Delta_W2", "Testing_Rank_W2",
            ]
            available = [c for c in display_cols if c in week_comp.columns]
            print(week_comp[available].to_string(index=False))

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
