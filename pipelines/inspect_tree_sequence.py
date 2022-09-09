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
    type=click.Path(exists=True, file_okay=True),
    required=True,
    help="Input trees file"
)
@click.option(
    "--out_csv_file", "-o",
    type=click.Path(file_okay=True),
    required=True,
    help="Output CSV file containing tree sequence statistics"
)
def get_tree_sequence_statistics(in_trees_file, out_csv_file):
    """
    :param str in_trees_file:
    :param str out_csv_file:
    :return: None
    :rtype: None
    """
    ts = tskit.load(in_trees_file)

    stats = OrderedDict()
    stats["sequence_length"] = ts.sequence_length
    stats["num_ndividuals"] = ts.num_individuals
    stats["num_samples"] = ts.num_samples
    stats["num_trees"] = ts.num_trees
    stats["num_sites"] = ts.num_sites
    stats["num_nodes"] = ts.num_nodes
    stats["num_edges"] = ts.num_edges
    stats["num_mutations"] = ts.num_mutations
    stats["num_singletons"] = util.count_singletons(ts)

    with open(out_csv_file, "w") as f:
        w = csv.writer(f)
        for k, v in stats.items():
            w.writerow([k, v])

    return None


if __name__ == "__main__":
    get_tree_sequence_statistics()
