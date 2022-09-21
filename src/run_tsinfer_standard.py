"""
Run the standard tsinfer pipeline, while allowing for specification of mismatch ratios
in both the ancestors and samples matching steps.

See https://tsinfer.readthedocs.io/en/latest/index.html
"""
import click
import tskit
import tsinfer


@click.command()
@click.option(
    "--in_samples_file",
    "-i",
    type=click.Path(exists=True, file_okay=True),
    required=True,
    help="Input samples file",
)
@click.option(
    "--out_dir",
    "-o",
    type=click.Path(exists=True, dir_okay=True),
    required=True,
    help="Output directory",
)
@click.option("--out_prefix", "-p", type=str, required=True, help="Output file prefix")
@click.option(
    "--mmr_ancestors",
    "-a",
    type=float,
    default=1,
    help="Mismatch ratio used when matching ancestor haplotypes",
)
@click.option(
    "--mmr_samples",
    "-s",
    type=float,
    default=1,
    help="Mismatch ratio used when matching sample haplotypes",
)
@click.option(
    "--num_threads", "-t", type=int, default=1, help="Number of threads to use"
)
def run_standard_tsinfer_pipeline(
    in_samples_file,
    out_dir,
    out_prefix,
    num_threads,
):
    """
    :param str in_samples_file: Samples file used for tsinfer input.
    :param str out_dir: Output directory.
    :param str out_prefix: Prefix of output filenames.
    :param float mmr_ancestors: Mismatch ratio used when matching ancestors.
    :param float mmr_samples: Mismatch ratio used when matching samples.
    :param int num_threads: Number of CPUs.
    """
    out_ancestors_file = out_dir + "/" + out_prefix + ".ancestors"
    out_ancestors_ts_file = out_dir + "/" + out_prefix + ".ancestors.trees"
    out_inferred_ts_file = out_dir + "/" + out_prefix + ".inferred.trees"

    print("INFO: START")
    print(f"DEPS: tskit {tskit.__version__}")
    print(f"DEPS: tsinfer {tsinfer.__version__}")

    print("INFO: Loading samples file")
    sample_data = tsinfer.load(in_samples_file)

    print("INFO: Generating ancestors")
    ancestor_data = tsinfer.generate_ancestors(
        sample_data=sample_data, path=out_ancestors_file, num_threads=num_threads
    )

    print("INFO: Matching ancestors")
    ancestors_ts = tsinfer.match_ancestors(
        sample_data=sample_data, ancestor_data=ancestor_data, num_threads=num_threads
    )
    ancestors_ts.dump(out_ancestors_ts_file)

    print("INFO: Matching samples")
    inferred_ts = tsinfer.match_samples(
        sample_data=sample_data, ancestors_ts=ancestors_ts, num_threads=num_threads
    )
    inferred_ts.dump(out_inferred_ts_file)

    print("INFO: END")


if __name__ == "__main__":
    run_standard_tsinfer_pipeline()
