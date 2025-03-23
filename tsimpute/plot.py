import math
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
