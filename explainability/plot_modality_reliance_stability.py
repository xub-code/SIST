"""
Plot modality reliance stability based on LOU explainability.

Input:
    explainability_modality_reliance/sample_seed_lou_metrics.csv

Output:
    plot/
        modality_reliance_run_stability.png/pdf/jpg
        modality_reliance_sample_distribution.png/pdf/jpg

Changes:
    1. Run stability figure:
       - line plot instead of bar chart
       - no error bars
       - Speech: circle marker
       - Text: triangle marker
       - both lines use black color

    2. Sample distribution figure:
       - refined boxplot
       - black jitter points for individual samples

    3. Figure size:
       - both panels use a fixed 1.6 x 1.6 inch canvas
       - exported files retain the exact canvas size

    4. Axis labels:
       - the x-axis title is omitted from the run-stability panel
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR = Path("explainability_modality_reliance")
OUTPUT_DIR = Path("plot")

SEEDS = [2024, 42, 0, 1, 123]

DPI = 600
FIGURE_SIZE_INCHES = (1.6, 1.6)

# Nature-style color palette
SPEECH_COLOR = "#0072B2"
TEXT_COLOR = "#D55E00"
POINT_COLOR = "#404040"
SPEECH_FILL = "#DCEAF7"
TEXT_FILL = "#FCE5D6"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Nimbus Roman",
            "DejaVu Serif",
        ],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
    }
)


def load_results():
    file_path = INPUT_DIR / "sample_seed_lou_metrics.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "sample_id",
        "seed",
        "speech_reliance_pct",
        "text_reliance_pct",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


def build_run_statistics(df):
    rows = []

    for index, seed in enumerate(SEEDS):

        run_df = df[df["seed"] == seed]

        if len(run_df) == 0:
            raise RuntimeError(
                f"No data found for seed={seed}"
            )

        rows.append(
            {
                "run": f"R{index + 1}",
                "speech_mean": run_df["speech_reliance_pct"].mean(),
                "text_mean": run_df["text_reliance_pct"].mean(),
            }
        )

    return pd.DataFrame(rows)


def build_sample_statistics(df):
    return (
        df.groupby("sample_id", as_index=False)
        [
            [
                "speech_reliance_pct",
                "text_reliance_pct",
            ]
        ]
        .mean()
    )


def style_axis(ax):

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.tick_params(
        direction="in",
        width=0.8,
        length=2.5,
    )

    # Keep the y-axis tick labels close to the axis without crowding it.
    ax.tick_params(
        axis="y",
        pad=1.5,
    )


def save_figure(fig, name):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for suffix in ["png", "pdf", "jpg"]:

        # Reassert the requested physical canvas size before every export.
        # Do not use bbox_inches="tight", because it changes the final size.
        fig.set_size_inches(
            *FIGURE_SIZE_INCHES,
            forward=True,
        )

        fig.savefig(
            OUTPUT_DIR / f"{name}.{suffix}",
            dpi=DPI,
            bbox_inches=None,
            facecolor="white",
            edgecolor="none",
        )


def plot_run_stability(run_df):

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE_INCHES
    )

    x = np.arange(len(run_df))

    ax.plot(
        x,
        run_df["speech_mean"],
        color=SPEECH_COLOR,
        marker="o",
        markersize=4,
        linewidth=1.0,
        label="Speech",
    )

    ax.plot(
        x,
        run_df["text_mean"],
        color=TEXT_COLOR,
        marker="^",
        markersize=4,
        linewidth=1.0,
        linestyle="--",
        label="Text",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        run_df["run"]
    )

    ax.set_ylabel(
        "Reliance (%)",
        labelpad=1.0,
    )

    ax.set_ylim(
        0,
        100
    )

    ax.legend(
        frameon=False,
        handlelength=1.5,
    )

    style_axis(ax)

    fig.subplots_adjust(
        left=0.17,
        right=0.98,
        # Keep the R1--R5 labels close to the lower canvas boundary.
        bottom=0.09,
        top=0.96,
    )

    save_figure(
        fig,
        "modality_reliance_run_stability"
    )

    plt.close(fig)


def plot_sample_distribution(sample_df):

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE_INCHES
    )

    values = [
        sample_df["speech_reliance_pct"].values,
        sample_df["text_reliance_pct"].values,
    ]

    box = ax.boxplot(
        values,
        labels=["Speech", "Text"],
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "linewidth": 1.0
        },
        whiskerprops={
            "linewidth": 0.8
        },
        capprops={
            "linewidth": 0.8
        },
        boxprops={
            "linewidth": 0.8
        },
    )

    box["boxes"][0].set_facecolor(SPEECH_FILL)
    box["boxes"][1].set_facecolor(TEXT_FILL)

    for patch in box["boxes"]:
        patch.set_alpha(0.9)

    rng = np.random.default_rng(42)

    for index, value in enumerate(values, start=1):

        jitter = rng.uniform(
            -0.08,
            0.08,
            len(value)
        )

        ax.scatter(
            np.ones(len(value)) * index + jitter,
            value,
            s=6,
            color=POINT_COLOR,
            alpha=0.45,
            linewidths=0,
        )

    ax.set_ylabel(
        "Reliance (%)",
        labelpad=1.0,
    )

    ax.set_ylim(
        0,
        100
    )

    style_axis(ax)

    fig.subplots_adjust(
        left=0.17,
        right=0.98,
        # Reduce unused space below the Speech and Text labels.
        bottom=0.10,
        top=0.96,
    )

    save_figure(
        fig,
        "modality_reliance_sample_distribution"
    )

    plt.close(fig)


def main():

    df = load_results()

    run_df = build_run_statistics(df)

    sample_df = build_sample_statistics(df)

    print(run_df)

    print(sample_df.describe())

    plot_run_stability(run_df)

    plot_sample_distribution(sample_df)

    print("[OK] Figures generated.")


if __name__ == "__main__":
    main()
