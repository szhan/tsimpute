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
