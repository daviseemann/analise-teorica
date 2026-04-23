import matplotlib.pyplot as plt
import numpy as np
from utils.select_lib import rejector
from utils.metrics import error


def plot_2D(data: np.array, labels: np.array, figsize=(12, 10)) -> None:

    n_classes = np.unique(labels).size

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0, :])  # ocupa as 2 colunas (gráfico 1)
    ax2 = fig.add_subplot(gs[1, 0])  # gráfico 2
    ax3 = fig.add_subplot(gs[1, 1])  # gráfico 3

    ax1.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", edgecolor="k")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("Dados Gaussianos")
    ax1.grid()

    for i in range(n_classes):
        class_data = data[labels == i]
        ax2.hist(class_data[:, 0], alpha=0.5, label=f"Classe {i}")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("Frequência")
    ax2.set_title("Distribuição de x1 por Classe")
    ax2.legend()

    for i in range(n_classes):
        class_data = data[labels == i]
        ax3.hist(class_data[:, 1], alpha=0.5, label=f"Classe {i}")
    ax3.set_xlabel("x2")
    ax3.set_ylabel("Frequência")
    ax3.set_title("Distribuição de x2 por Classe")
    ax3.legend()

    plt.tight_layout()
    plt.show()


def plot_features(data, labels):

    n_classes = np.unique(labels).size
    n_features = data.shape[1]

    fig, axes = plt.subplots(
        n_classes, n_features, figsize=(3 * n_features, 2 * n_classes)
    )
    for i in range(n_classes):
        for j in range(n_features):
            axes[i, j].hist(data[labels == i, j], bins=20, alpha=0.7)
            axes[i, j].set_title(f"Classe {i} - Feature {j}")
    plt.tight_layout()
    plt.show()


def plot_thresholds(
    data: np.array,
    labels: np.array,
    preds: np.array,
    confidence: np.array,
    thresholds: list,
) -> None:

    n_plots = len(thresholds) - 1

    fig, ax = plt.subplots(n_plots, figsize=(10, 10))

    for i, threshold in enumerate(thresholds[1:]):
        mask = rejector(confidence, threshold)
        masked_data = data[mask]
        accuracia = 1 - error(labels[mask], preds[mask])
        error_mask = labels[mask] != preds[mask]

        ax[i].scatter(
            masked_data[:, 0],
            masked_data[:, 1],
            c=error_mask,
            cmap="viridis",
            edgecolor="k",
            alpha=0.5,
        )
        ax[i].set_title(
            f"Máscara com {threshold=} - {masked_data.shape[0]} amostras, accuracia={accuracia:.2f}"
        )
        ax[i].grid()

    plt.tight_layout()
    plt.show()
