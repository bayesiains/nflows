import numpy as np
from matplotlib import pyplot as plt


def plot_hist_marginals(
    data,
    weights=None,
    lims=None,
    ground_truth=None,
    upper=True,
    rasterized=True,
    show_xticks=False,
):
    """
    Plots marginal histograms and pairwise scatter plots of a dataset.
    """

    data = np.asarray(data)
    n_bins = int(np.sqrt(data.shape[0]))
    n_bins = 25

    if data.ndim == 1:

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.hist(data, weights=weights, bins=n_bins, density=True, rasterized=rasterized)
        ax.set_ylim([0.0, ax.get_ylim()[1]])
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        if lims is not None:
            ax.set_xlim(lims)
        if ground_truth is not None:
            ax.vlines(ground_truth, 0, ax.get_ylim()[1], color="r")

    else:

        n_dim = data.shape[1]
        fig = plt.figure(figsize=(8, 8))

        if weights is None:
            col = "k"
            vmin, vmax = None, None
        else:
            col = weights
            vmin, vmax = 0.0, np.max(weights)

        if lims is not None:
            lims = np.asarray(lims)
            lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims

        for i in range(n_dim):
            for j in range(i, n_dim) if upper else range(i + 1):

                ax = fig.add_subplot(n_dim, n_dim, i * n_dim + j + 1)

                if i == j:
                    ax.hist(
                        data[:, i],
                        weights=weights,
                        bins=n_bins,
                        density=True,
                        rasterized=rasterized,
                        color="black",
                        alpha=0.75,
                    )
                    ax.set_ylim([0.0, ax.get_ylim()[1]])
                    ax.tick_params(
                        axis="y", which="both", left=False, right=False, labelleft=False
                    )
                    if i < n_dim - 1 and not upper:
                        ax.tick_params(axis="x", which="both", labelbottom=False)
                    if lims is not None:
                        ax.set_xlim(lims[i])
                    if ground_truth is not None:
                        ax.vlines(ground_truth[i], 0, ax.get_ylim()[1], color="C1")
                    ax.set_xlabel(r"$ \theta_{:} $".format(i + 1), fontsize=16)
                    if not show_xticks:
                        ax.set_xticks([])

                else:
                    ax.scatter(
                        data[:, j],
                        data[:, i],
                        c=col,
                        s=1,
                        marker="o",
                        vmin=vmin,
                        vmax=vmax,
                        cmap="binary",
                        edgecolors="none",
                        rasterized=rasterized,
                        alpha=0.1,
                    )
                    if i < n_dim - 1:
                        ax.tick_params(axis="x", which="both", labelbottom=False)
                    if j > 0:
                        ax.tick_params(axis="y", which="both", labelleft=False)
                    if j == n_dim - 1:
                        ax.tick_params(axis="y", which="both", labelright=True)
                    if lims is not None:
                        ax.set_xlim(lims[j])
                        ax.set_ylim(lims[i])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if ground_truth is not None:
                        ax.scatter(
                            ground_truth[j],
                            ground_truth[i],
                            c="C1",
                            s=16,
                            marker="o",
                            edgecolors="none",
                        )
    plt.tight_layout()
    return fig


def plot_hist_marginals_pair(
    data1,
    data2,
    weights=None,
    lims=None,
    ground_truth=None,
    upper=True,
    rasterized=True,
    show_xticks=False,
):
    n_dim = data1.shape[1]
    fig = plt.figure(figsize=(((n_dim + 2) / n_dim) * 8, 8))
    n_bins = 25

    if weights is None:
        col = "k"
        vmin, vmax = None, None
    else:
        col = weights
        vmin, vmax = 0.0, np.max(weights)

    if lims is not None:
        lims = np.asarray(lims)
        lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims

    for i in range(n_dim):
        for j in range(i + 1):

            ax = fig.add_subplot(n_dim, n_dim + 2, i * (n_dim + 2) + j + 1)

            if i == j:
                ax.hist(
                    data1[:, i],
                    weights=weights,
                    bins=n_bins,
                    density=True,
                    rasterized=rasterized,
                    color="black",
                    alpha=0.75,
                )
                ax.set_ylim([0.0, ax.get_ylim()[1]])
                ax.tick_params(
                    axis="y", which="both", left=False, right=False, labelleft=False
                )
                if i < n_dim - 1 and not upper:
                    ax.tick_params(axis="x", which="both", labelbottom=False)
                if lims is not None:
                    ax.set_xlim(lims[i])
                if ground_truth is not None:
                    ax.vlines(ground_truth[i], 0, ax.get_ylim()[1], color="C1")
                ax.set_xlabel(r"$ \theta_{:} $".format(i + 1), fontsize=16)
                if not show_xticks:
                    ax.set_xticks([])

            else:
                ax.scatter(
                    data1[:, j],
                    data1[:, i],
                    c=col,
                    s=1,
                    marker="o",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="binary",
                    edgecolors="none",
                    rasterized=rasterized,
                    alpha=0.1,
                )
                if i < n_dim - 1:
                    ax.tick_params(axis="x", which="both", labelbottom=False)
                if j > 0:
                    ax.tick_params(axis="y", which="both", labelleft=False)
                if j == n_dim - 1:
                    ax.tick_params(axis="y", which="both", labelright=True)
                if lims is not None:
                    ax.set_xlim(lims[j])
                    ax.set_ylim(lims[i])
                ax.set_xticks([])
                ax.set_yticks([])
                if ground_truth is not None:
                    ax.scatter(
                        ground_truth[j],
                        ground_truth[i],
                        c="C1",
                        s=16,
                        marker="o",
                        edgecolors="none",
                    )

    for i in range(n_dim):
        for j in range(i, n_dim):

            ax = fig.add_subplot(n_dim, n_dim + 2, i * (n_dim + 2) + j + 2 + 1)

            if i == j:
                ax.hist(
                    data2[:, i],
                    weights=weights,
                    bins=n_bins,
                    density=True,
                    rasterized=rasterized,
                    color="black",
                    alpha=0.75,
                )
                ax.set_ylim([0.0, ax.get_ylim()[1]])
                ax.tick_params(
                    axis="y", which="both", left=False, right=False, labelleft=False,
                )
                if i < n_dim - 1 and not upper:
                    ax.tick_params(axis="x", which="both", labelbottom=False)
                if lims is not None:
                    ax.set_xlim(lims[i])
                if ground_truth is not None:
                    ax.vlines(ground_truth[i], 0, ax.get_ylim()[1], color="C1")
                ax.set_xlabel(r"$ \theta_{:} $".format(i + 1), fontsize=16)
                if not show_xticks:
                    ax.set_xticks([])

            else:
                ax.scatter(
                    data2[:, j],
                    data2[:, i],
                    c=col,
                    s=1,
                    marker="o",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="binary",
                    edgecolors="none",
                    rasterized=rasterized,
                    alpha=0.1,
                )
                if i < n_dim - 1:
                    ax.tick_params(axis="x", which="both", labelbottom=False)
                if j > 0:
                    ax.tick_params(axis="y", which="both", labelleft=False)
                if j == n_dim - 1:
                    ax.tick_params(axis="y", which="both", labelright=True)
                if lims is not None:
                    ax.set_xlim(lims[j])
                    ax.set_ylim(lims[i])
                ax.set_xticks([])
                ax.set_yticks([])
                if ground_truth is not None:
                    ax.scatter(
                        ground_truth[j],
                        ground_truth[i],
                        c="C1",
                        s=16,
                        marker="o",
                        edgecolors="none",
                    )

    plt.tight_layout()
    return fig


def main():
    x = np.random.randn(1000, 5)
    y = 3 * np.random.randn(1000, 5)
    plot_hist_marginals_pair(x, y, lims=[-10, 10], ground_truth=np.zeros(5))
    plt.show()


if __name__ == "__main__":
    main()
