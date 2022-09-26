"""
Get basic statistics about the properties of a tree sequence.
"""
import click
from collections import OrderedDict
import csv
import sys
import tskit

sys.path.append("./python")
import util


@click.command()
@click.option(
    "--in_trees_file", "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input trees file"
)
@click.option(
    "--out_csv_file", "-o",
    type=click.Path(exists=False),
    required=True,
    help="Output CSV file containing the statistics"
)
def main(in_trees_file, out_csv_file):
    """
    :param str in_trees_file: Path to file containing a tree sequence.
    :param str out_csv_file: Path to file with statistics.
    """
    ts = tskit.load(in_trees_file)

    stats = OrderedDict()
    stats["sequence_length"] = ts.sequence_length
    stats["num_individuals"] = ts.num_individuals
    stats["num_samples"] = ts.num_samples
    stats["num_trees"] = ts.num_trees
    stats["num_sites"] = ts.num_sites
    stats["num_nodes"] = ts.num_nodes
    stats["num_edges"] = ts.num_edges
    stats["num_mutations"] = ts.num_mutations
    stats["num_singletons"] = util.count_singletons(ts)
    stats["num_inference_sites"]= util.count_inference_sites(ts)

    with open(out_csv_file, "w") as f:
        w = csv.writer(f)
        for k, v in stats.items():
            w.writerow([k, v])


if __name__ == "__main__":
    main()