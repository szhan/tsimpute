import click
import matplotlib.pyplot as plt
import numpy as np
import tskit


@click.command()
@click.option(
    "--in_trees_file",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input tree sequence file.",
)
@click.option(
    "--out_csv_file",
    "-o1",
    type=click.Path(exists=False),
    required=True,
    help="Output CSV file.",
)
@click.option(
    "--out_png_file",
    "-o2",
    type=click.Path(exists=False),
    required=True,
    help="Output PNG file.",
)
def plot_tajima_d_versus_site(in_trees_file, out_csv_file, out_png_file):
    """
    Plot site-wise Tajima's D over sites in a chromosome in a tree sequence.

    :param str in_trees_file: Input file containing a tree sequence.
    :param str out_png_file: Output file with plot in PNG.
    """
    ts = tskit.load(in_trees_file)

    D = ts.Tajimas_D(mode="site", windows="sites")

    plt.plot(D)
    plt.xlabel("Site ID")
    plt.ylabel("Tajima's D")
    plt.savefig(out_png_file)

    np.savetxt(fname=out_csv_file, X=D, header="site_wise_tajimas_d")


if __name__ == "__main__":
    plot_tajima_d_versus_site()
