import click
import matplotlib.pyplot as plt


@click.command()
@click.option(
    "--in_trees_file",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input tree sequence file.",
)
@click.option(
    "--out_png_file",
    "-o",
    type=click.Path(exists=False),
    required=True,
    help="Output PNG file.",
)
def plot_tajima_d_versus_site(ts, out_png_file):
    """
    Plot site-wise Tajima's D over sites in a chromosome in a tree sequence.

    :param tskit.TreeSequence ts: A tree sequence.
    """
    D = ts.Tajimas_D(mode="site", windows="sites")

    plt.plot(D)
    plt.xlabel("Site ID")
    plt.ylabel("Tajima's D")
    plt.savefig(out_png_file)


if __name__ == "__main__":
    plot_tajima_d_versus_site()
