from importlib.metadata import requires
import click
import sys
from pathlib import Path

import tsinfer

sys.path.append("./python")
import read_vcf


@click.command()
@click.option(
    "--in_vcf_file", "-i",
    type=click.Path(exists=True, file_okay=True),
    required=True,
    help="Input (gzipped) VCF file"
)
@click.option(
    "--out_dir", "-o",
    type=click.Path(exists=True, dir_okay=True),
    required=True,
    help="Output directory"
)
@click.option(
    "--out_prefix", "-p",
    type=str,
    required=True,
    help="Output file prefix"
)
@click.option(
    "--num_threads", "-t",
    type=int,
    default=1,
    help="Number of threads to use"
)
def run_standard_tsinfer_pipeline(
    vcf_file,
    out_dir,
    out_prefix,
    num_threads
):
    """
    See https://tsinfer.readthedocs.io/en/latest/index.html

    :param click.Path vcf_file:
    :param click.Path out_dir:
    :param str out_prefix:
    :param int num_threads:
    :return: None
    :rtype: None
    """
    out_path = Path(out_dir)
    samples_file =  out_path / out_prefix + ".samples"
    ancestors_file = out_path / out_prefix + ".ancestors"
    ancestors_ts_file = out_path / out_prefix + ".ancestors.trees"
    inferred_ts_file = out_path / out_prefix + ".inferred.trees"

    print(" ".join(["INFO" + ":" + "START"]))

    print(" ".join(["INFO" + ":" + "Parsing VCF file"]))
    sample_data = read_vcf.create_sample_data_from_vcf_file(
        vcf_file=vcf_file,
        ploidy_level=2,
        samples_file=samples_file
    )

    print(" ".join(["INFO" + ":" + "Generating ancestors"]))
    ancestor_data = tsinfer.generate_ancestors(
        sample_data=sample_data,
        path=ancestors_file,
        num_threads=num_threads
    )

    print(" ".join(["INFO" + ":" + "Matching ancestors"]))
    ancestors_ts = tsinfer.match_ancestors(
        sample_data=sample_data,
        ancestor_data=ancestor_data,
        num_threads=num_threads
    )
    ancestors_ts.dump(ancestors_ts_file)

    print(" ".join(["INFO" + ":" + "Matching samples"]))
    inferred_ts = tsinfer.match_samples(
        sample_data=sample_data,
        ancestors_ts=ancestors_ts,
        num_threads=num_threads
    )
    inferred_ts.dump(inferred_ts_file)

    print(" ".join(["INFO" + ":" + "END"]))

    return None


if __name__ == "__main__":
    run_standard_tsinfer_pipeline()
