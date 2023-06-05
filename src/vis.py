import numpy as np
import matplotlib.pyplot as plt


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
