import re
from pathlib import Path
from typing import TypedDict

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger


class LogEntry(TypedDict):
    epoch: int
    num_samples_seen: int
    train_loss: float
    test_loss: float
    test_accuracy: float


def parse_log_line(line: str) -> LogEntry | None:
    """Parse a single log line to extract training metrics."""
    pattern = r"Epoch (\d+).+Num Samples Seen: ([\d,]+).+Train Loss: ([\d.]+).+Test Loss: ([\d.]+).+Test Accuracy: ([\d.]+)%"
    match = re.search(pattern, line)
    if not match:
        return None

    return {
        "epoch": int(match.group(1)),
        "num_samples_seen": int(match.group(2).replace(",", "")),
        "train_loss": float(match.group(3)),
        "test_loss": float(match.group(4)),
        "test_accuracy": float(match.group(5)),
    }


def parse_log_file(log_path: Path) -> pd.DataFrame:
    """Parse the entire log file and return a DataFrame with training metrics."""
    entries: list[LogEntry] = []
    with open(log_path) as f:
        for line in f:
            entry = parse_log_line(line)
            if entry:
                entries.append(entry)
    return pd.DataFrame(entries)


def create_plots(df: pd.DataFrame, outdir: Path) -> None:
    """Create and save training metric plots."""
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))

    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Train and Test Loss vs Epoch
    sns.lineplot(data=df, x="epoch", y="train_loss", ax=axes[0, 0], label="Train Loss")
    sns.lineplot(data=df, x="epoch", y="test_loss", ax=axes[0, 0], label="Test Loss")
    axes[0, 0].set_title("Loss vs Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    # Plot 2: Train and Test Loss vs Samples Seen
    sns.lineplot(
        data=df, x="num_samples_seen", y="train_loss", ax=axes[0, 1], label="Train Loss"
    )
    sns.lineplot(
        data=df, x="num_samples_seen", y="test_loss", ax=axes[0, 1], label="Test Loss"
    )
    axes[0, 1].set_title("Loss vs Samples Seen")
    axes[0, 1].set_xlabel("Samples Seen")
    axes[0, 1].set_ylabel("Loss")

    # Plot 3: Test Accuracy vs Epoch
    sns.lineplot(data=df, x="epoch", y="test_accuracy", ax=axes[1, 0])
    axes[1, 0].set_title("Test Accuracy vs Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")

    # Plot 4: Test Accuracy vs Samples Seen
    sns.lineplot(data=df, x="num_samples_seen", y="test_accuracy", ax=axes[1, 1])
    axes[1, 1].set_title("Test Accuracy vs Samples Seen")
    axes[1, 1].set_xlabel("Samples Seen")
    axes[1, 1].set_ylabel("Accuracy (%)")

    # Adjust layout and save
    plt.tight_layout()
    plot_path = outdir / "training_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Plots saved to {plot_path}")


@click.command()
@click.argument(
    "outdir",
    type=Path,
)
def main(outdir: Path) -> None:
    """Create training metric plots from log file."""
    outdir = outdir.resolve()
    log_path = outdir / "train.log"

    if not log_path.exists():
        raise click.BadParameter(f"Log file not found at {log_path}")

    logger.info(f"Reading log file from {log_path}")
    df = parse_log_file(log_path)
    logger.info(f"Found {len(df)} training entries")

    create_plots(df, outdir)


if __name__ == "__main__":
    main()
