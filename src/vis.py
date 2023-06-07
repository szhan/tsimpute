import math
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sample_path(path, site_pos, tracks=None, window=None):
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.plot(
        site_pos,
        path,
    )
    # Add tracks
    if tracks is not None:
        for i in np.arange(len(tracks)):
            ax.plot(
                tracks[i][0],
                np.repeat(-(i + 1) * 1_000, len(tracks[i][0])),
                marker="|",
                color=tracks[i][1],
                linestyle=""
            )
    if window is not None:
        assert len(window) == 2
        ax.set_xlim(window[0], window[1])
    ax.set_ylabel("Index of sample")
    ax.set_xlabel("Genomic position");


def compare_sample_paths(path_1, path_2):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 5,))

    bool_identical_samples = np.equal(path_1, path_2)
    bool_different_samples = np.invert(bool_identical_samples)

    sample_indices = np.arange(len(path_1))

    ax.plot(
        sample_indices[bool_identical_samples],
        path_1[bool_identical_samples],
        color="black", marker="o", linestyle=""
    )
    ax.plot(
        sample_indices[bool_different_samples],
        path_1[bool_different_samples],
        color="blue", marker="o", linestyle=""
    )
    ax.plot(
        sample_indices[bool_different_samples],
        path_2[bool_different_samples],
        color="orange", marker="o", linestyle=""
    )

    ax.set_ylabel("Index of sample")
    ax.set_xlabel("Index of site");


def plot_iqs_versus_maf(
    df,
    method,
    min_iqs=0.90,
    max_maf=0.50,
    subsample_frac=0.01,
    out_png_file=None,
    dpi=100,
):
    assert 0.0 <= subsample_frac <= 1.0

    subsample_size = math.ceil(df.shape[0] * subsample_frac)
    subsample = np.random.choice(np.arange(df.shape[0]), subsample_size)

    values = np.vstack([df["ref_ma_freq"][subsample], df["iqs"][subsample]])
    kernel = stats.gaussian_kde(values)
    x = kernel(np.vstack([df["ref_ma_freq"], df["iqs"]]))

    num_sites_min_iqs = np.sum(df["iqs"] >= min_iqs)
    prop_sites_min_iqs = num_sites_min_iqs / float(df.shape[0])

    fig, ax = plt.subplots(figsize=(7, 7,))

    ax.set_title(
        f"{method}"
        "\n"
        f"% sites with min IQS: {round(prop_sites_min_iqs * 100.0, 2)}",
        size=20
    )
    ax.set_xlim([0, max_maf])
    ax.set_ylabel("IQS", size=20)
    ax.set_xlabel("MAF", size=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    g = sns.scatterplot(
        y="iqs",
        x="ref_ma_freq",
        data=df,
        c=x,
        cmap="viridis",
        #x_jitter=True,
        ax=ax,
        alpha=0.2
    )

    if out_png_file is not None:
        g.get_figure().savefig(out_png_file, dpi=dpi)
