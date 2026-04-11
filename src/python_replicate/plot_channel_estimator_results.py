from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

RESULT_PATH = Path("python_replicate/output/channel_estimator_validation.pt")
FIG_PATH = Path("python_replicate/output/channel_estimator_channels.png")
HIST_PATH = Path("python_replicate/output/channel_estimator_dl_mse.png")
SYMBOL_DIR = Path("python_replicate/output")


def plot_channel_heatmaps(estimates: dict, metrics: dict) -> None:
    kinds = list(estimates.keys())
    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4), constrained_layout=True)
    if len(kinds) == 1:
        axes = [axes]
    for ax, kind in zip(axes, kinds):
        est = estimates[kind]
        magnitude = torch.abs(est).numpy()
        im = ax.imshow(magnitude, aspect="auto", cmap="viridis")
        ax.set_title(f"{kind} |H|")
        ax.set_xlabel("OFDM symbol")
        ax.set_ylabel("Subcarrier")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ber = metrics[kind]["ber"]
        if ber == ber:  # not NaN
            ax.text(
                0.02,
                0.95,
                f"BER={ber:.3e}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
    fig.suptitle("Estimated Channels (Magnitude)")
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)


def plot_dl_mse(metrics: dict) -> None:
    if "DL" not in metrics:
        return
    mse_samples = metrics["DL"]["mse_samples"]
    if mse_samples.numel() == 0:
        return
    data = mse_samples.numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=50, color="tab:blue", alpha=0.8)
    plt.title("DL Payload Equalization Error Distribution")
    plt.xlabel("Per-symbol MSE")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(HIST_PATH, dpi=200)
    plt.close()


def plot_symbol_traces(metrics: dict) -> None:
    SYMBOL_DIR.mkdir(parents=True, exist_ok=True)
    for kind, meta in metrics.items():
        traces = meta.get("traces")
        if traces is None or traces["tx"].numel() == 0:
            continue
        tx = traces["tx"].numpy()
        rx = traces["rx"].numpy()
        eq = traces["eq"].numpy()
        x = range(len(tx))
        fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
        entries = [("Tx", tx), ("Rx (FFT)", rx), ("Eq (ZF)", eq)]
        for row, (label, data) in enumerate(entries):
            axes[row][0].plot(x, data.real, color="tab:blue")
            axes[row][0].set_ylabel(f"{label} Re")
            axes[row][1].plot(x, data.imag, color="tab:orange")
            axes[row][1].set_ylabel(f"{label} Im")
        axes[-1][0].set_xlabel("Data symbol index")
        axes[-1][1].set_xlabel("Data symbol index")
        fig.suptitle(f"{kind} Symbol Traces (first {len(tx)} samples)")
        out_path = SYMBOL_DIR / f"channel_estimator_{kind.lower()}_symbols.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {kind} symbol trace plot to {out_path}")


def main() -> None:
    blob = torch.load(RESULT_PATH)
    estimates = blob["estimates"]
    metrics = blob["metrics"]
    # convert tensors to numpy-friendly format
    estimates = {k: v.clone().detach() for k, v in estimates.items()}
    metrics = {}
    for k, v in blob["metrics"].items():
        metrics[k] = {
            "ber": float(v.get("ber", float("nan"))),
            "mse_samples": v.get("mse_samples", torch.tensor([])).clone().detach(),
            "traces": {
                name: tensor.clone().detach()
                for name, tensor in v.get("traces", {}).items()
            },
        }
    plot_channel_heatmaps(estimates, metrics)
    plot_dl_mse(metrics)
    plot_symbol_traces(metrics)
    print(f"Saved channel heatmaps to {FIG_PATH}")
    if metrics["DL"]["mse_samples"].numel() > 0:
        print(f"Saved DL MSE histogram to {HIST_PATH}")


if __name__ == "__main__":
    main()
