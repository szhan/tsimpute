"""
Run the standard tsinfer pipeline, while allowing for specification of mismatch ratios
in both the ancestors and samples matching steps.

See https://tsinfer.readthedocs.io/en/latest/index.html
"""
import click
from datetime import datetime
import logging
import msprime
import tskit
import tsinfer


@click.command()
@click.option(
    "--in_samples_file",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input samples file.",
)
@click.option(
    "--out_dir",
    "-o",
    type=click.Path(exists=True),
    required=True,
    help="Output directory.",
)
@click.option("--out_prefix", "-p", type=str, required=True, help="Output file prefix.")
@click.option(
    "--recombination_rate",
    "-r",
    type=float,
    default=None,
    help="Uniform recombination rate.",
)
@click.option(
    "--genetic_map",
    "-g",
    type=click.Path(exists=True),
    default=None,
    help="HapMap3-formatted genetic map file.",
)
@click.option(
    "--mmr_ancestors",
    "-a",
    type=float,
    default=None,
    help="Mismatch ratio for matching ancestors.",
)
@click.option(
    "--mmr_samples",
    "-s",
    type=float,
    default=None,
    help="Mismatch ratio for matching samples.",
)
@click.option(
    "--truncate_ancestors",
    is_flag=True,
    default=False,
    help="Truncate ancestors before matching ancestors?",
)
@click.option("--num_threads", "-t", type=int, default=1, help="Number of CPUs.")
def run_tsinfer(
    in_samples_file,
    out_dir,
    out_prefix,
    recombination_rate,
    genetic_map,
    mmr_ancestors,
    mmr_samples,
    truncate_ancestors,
    num_threads,
):
    """
    :param str in_samples_file: Samples file for tsinfer input.
    :param str out_dir: Output directory.
    :param str out_prefix: Prefix of output filenames.
    :param float recombination_rate: Uniform genome-wide recombination rate (default = None).
    :param str genetic_map: Genetic map file for msprime input (default = None).
    :param float mmr_ancestors: Mismatch ratio used when matching ancestors (default = None).
    :param float mmr_samples: Mismatch ratio used when matching samples (default = None).
    :param bool truncate_ancestors: Truncate ancestors before matching ancestors? (default = False).
    :param int num_threads: CPUs (default = 1).
    """
    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"start: {start_datetime}")

    logging.info(f"dep: tskit {tskit.__version__}")
    logging.info(f"dep: tsinfer {tsinfer.__version__}")
    logging.info(f"par: recombination_rate = {recombination_rate}")
    logging.info(f"par: genetic_map = {genetic_map}")
    logging.info(f"par: mmr_ancestors = {mmr_ancestors}")
    logging.info(f"par: mmr_samples = {mmr_samples}")
    logging.info(f"par: truncate_ancestors = {truncate_ancestors}")

    logging.info(f"Loading samples file: {in_samples_file}")
    sample_data = tsinfer.load(in_samples_file)

    if genetic_map is not None:
        logging.info(f"Loading genetic map: {genetic_map}")
        logging.info("Using this instead of a genome-wide recombination rate.")
        recombination_rate = msprime.RateMap.read_hapmap(genetic_map)

    out_ancestors_file = out_dir + "/" + out_prefix + ".ancestors"
    out_ancestors_ts_file = out_dir + "/" + out_prefix + ".ancestors.trees"
    out_inferred_ts_file = out_dir + "/" + out_prefix + ".inferred.trees"

    logging.info("Generating ancestors.")
    ancestor_data = tsinfer.generate_ancestors(
        sample_data=sample_data, path=out_ancestors_file, num_threads=num_threads
    )

    if truncate_ancestors:
        logging.info("Truncating ancestors.")
        # TODO: Allow user to specify lower and upper time bounds.
        ancestor_data = ancestor_data.truncate_ancestors(
            lower_time_bound=0.4, upper_time_bound=0.6
        )

    logging.info("Matching ancestors.")
    ancestors_ts = tsinfer.match_ancestors(
        sample_data=sample_data,
        ancestor_data=ancestor_data,
        recombination_rate=recombination_rate,
        mismatch_ratio=mmr_ancestors,
        num_threads=num_threads,
    )
    logging.info(f"Writing ancestors tree sequence to file: {out_ancestors_ts_file}")
    ancestors_ts.dump(out_ancestors_ts_file)

    logging.info("Matching samples.")
    inferred_ts = tsinfer.match_samples(
        sample_data=sample_data,
        ancestors_ts=ancestors_ts,
        recombination_rate=recombination_rate,
        mismatch_ratio=mmr_samples,
        num_threads=num_threads,
    )
    logging.info(f"Writing inferred tree sequence to file: {out_inferred_ts_file}")
    inferred_ts.dump(out_inferred_ts_file)

    end_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"end: {end_datetime}")


if __name__ == "__main__":
    run_tsinfer()
