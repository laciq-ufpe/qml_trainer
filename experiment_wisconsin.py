"""
Experiment: train Wisconsin Breast Cancer on all architecture × encoding combinations.

Saves to results/:
  - training_curves.png   — loss per epoch for every combination
  - metrics.png           — train/test accuracy bar chart
  - metrics.csv           — raw numbers
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from qmlvqc import (
    VQC,
    AngleEncoding,
    BasicEntangler,
    BCE,
    PhaseEncoding,
    StronglyEntangled,
)
from qmlvqc.data import load_wisconsin


N_QUBITS = 5
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 32
RESULTS_DIR = "results"

ARCHITECTURES = {
    "StronglyEntangled(3)": StronglyEntangled(n_layers=3),
    "BasicEntangler(4)": BasicEntangler(n_layers=4),
}

ENCODINGS = {
    "AngleEncoding": AngleEncoding,
    "PhaseEncoding": PhaseEncoding,
}


def run_experiment(arch_name, arch, enc_name, enc_cls, X_train, y_train, X_test, y_test):
    label = f"{arch_name} + {enc_name}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    model = VQC(
        n_qubits=N_QUBITS,
        architecture=arch,
        loss=BCE(),
        encoding=enc_cls(),
    )
    model.fit(X_train, y_train, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"  Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")

    return {
        "label": label,
        "history": model.history_,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }


def plot_training_curves(results: list[dict], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results:
        ax.plot(range(1, len(r["history"]) + 1), r["history"], marker="o", markersize=3, label=r["label"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE)")
    ax.set_title("Training Loss — Wisconsin Breast Cancer")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_metrics(results: list[dict], output_path: str) -> None:
    labels = [r["label"] for r in results]
    train_accs = [r["train_acc"] for r in results]
    test_accs = [r["test_acc"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_train = ax.bar(x - width / 2, train_accs, width, label="Train", color="steelblue")
    bars_test = ax.bar(x + width / 2, test_accs, width, label="Test", color="coral")

    for bar in (*bars_train, *bars_test):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Train / Test Accuracy — Wisconsin Breast Cancer")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_csv(results: list[dict], output_path: str) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["combination", "train_acc", "test_acc", "final_loss"])
        for r in results:
            writer.writerow([r["label"], r["train_acc"], r["test_acc"], r["history"][-1]])
    print(f"Saved: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_wisconsin()

    results = []
    for arch_name, arch in ARCHITECTURES.items():
        for enc_name, enc_cls in ENCODINGS.items():
            result = run_experiment(
                arch_name, arch, enc_name, enc_cls,
                X_train, y_train, X_test, y_test,
            )
            results.append(result)

    plot_training_curves(results, os.path.join(RESULTS_DIR, "training_curves.png"))
    plot_metrics(results, os.path.join(RESULTS_DIR, "metrics.png"))
    save_csv(results, os.path.join(RESULTS_DIR, "metrics.csv"))

    print("\nDone. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
