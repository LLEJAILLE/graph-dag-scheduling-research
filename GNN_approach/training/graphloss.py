import os
import re

import matplotlib.pyplot as plt

LOG_FILES = [
    "GCN_BC_v1.pth_epochs_loss_log.txt",
    "GCN_BC_v2_d2.pth_epochs_loss_log.txt",
    "GCN_BC_v3_d3.pth_epochs_loss_log.txt",
]

EPOCH_LOSS_PATTERN = re.compile(r"Epoch\s+(\d+),\s+Loss:\s+([0-9.eE+-]+)")
TIME_PATTERN = re.compile(r"Training completed in\s+([0-9]*\.?[0-9]+)\s+minutes\.")


def read_lines_with_fallback_encoding(file_path):
    for encoding in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.readlines()
        except UnicodeError:
            continue
    raise UnicodeError(f"Could not decode {file_path} with known encodings.")


def moving_average(values, window=5):
    if not values:
        return []
    averaged = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


def parse_log(log_file):
    lines = read_lines_with_fallback_encoding(log_file)
    epochs = []
    losses = []
    training_time = None

    for line in lines:
        epoch_loss_match = EPOCH_LOSS_PATTERN.search(line)
        if epoch_loss_match:
            epochs.append(int(epoch_loss_match.group(1)))
            losses.append(float(epoch_loss_match.group(2)))
            continue

        time_match = TIME_PATTERN.search(line)
        if time_match:
            training_time = float(time_match.group(1))

    return epochs, losses, training_time


def build_three_plots(log_file):
    epochs, losses, training_time = parse_log(log_file)
    if not epochs:
        print(f"Skipped {log_file}: no epoch/loss lines found.")
        return

    smooth_losses = moving_average(losses, window=5)
    tail_start = max(0, int(len(epochs) * 0.8))

    base_name = os.path.splitext(log_file)[0]
    time_text = f"{training_time:.2f} min" if training_time is not None else "unknown"

    fig, axes = plt.subplots(3, 1, figsize=(5, 18))
    fig.suptitle(f"{base_name} | Training time: {time_text}")

    axes[0].plot(epochs, losses, color="tab:blue", linewidth=1.8)
    axes[0].set_title("Raw Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, smooth_losses, color="tab:orange", linewidth=1.8)
    axes[1].set_title("Smoothed Loss (window=5)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs[tail_start:], losses[tail_start:], color="tab:green", linewidth=1.8)
    axes[2].set_title("Final 20% Epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = f"{base_name}_3plots.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    for log_file in LOG_FILES:
        build_three_plots(log_file)
    plt.show()


if __name__ == "__main__":
    main()
