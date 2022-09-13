import click
import sys
import tskit
import tsinfer
import cyvcf2

sys.path.append("./python")
import read_vcf


@click.command()
@click.option(
    "--in_file",
    "-i",
    type=click.Path(exists=True, file_okay=True),
    required=True,
    help="Input (gzipped) VCF file",
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
    "--ancestral_alleles_file",
    "-a",
    type=click.Path(exists=True, file_okay=True),
    default=None,
    help="Input VCF file with ancestral alleles.",
)
@click.option(
    "--num_threads", "-t", type=int, default=1, help="Number of threads to use"
)
def run_standard_tsinfer_pipeline(
    in_file, out_dir, out_prefix, ancestral_alleles_file, num_threads
):
    """
    TODO

    See https://tsinfer.readthedocs.io/en/latest/index.html

    :param str in_file:
    :param str out_dir:
    :param str out_prefix:
    :param str ancestral_alleles_file:
    :param int num_threads:
    :return: None
    :rtype: None
    """
    out_samples_file = out_dir + "/" + out_prefix + ".samples"
    out_ancestors_file = out_dir + "/" + out_prefix + ".ancestors"
    out_ancestors_ts_file = out_dir + "/" + out_prefix + ".ancestors.trees"
    out_inferred_ts_file = out_dir + "/" + out_prefix + ".inferred.trees"

    print("INFO: START")
    print(f"DEPS: tskit {tskit.__version__}")
    print(f"DEPS: tsinfer {tsinfer.__version__}")
    print(f"DEPS: cyvcf2 {cyvcf2.__version__}")

    map_aa = None
    if ancestral_alleles_file is not None:
        print("INFO: Parsing VCF file with ancestral alleles")
        vcf_aa = read_vcf.get_vcf(ancestral_alleles_file, num_threads=num_threads)
        map_aa, _ = read_vcf.extract_ancestral_alleles_from_vcf(vcf_aa, seq_name_prefix="chr")

    print("INFO: Parsing input VCF file")
    vcf = read_vcf.get_vcf(in_file, num_threads=num_threads)
    sample_data = read_vcf.create_sample_data_from_vcf(
        vcf,
        samples_file=out_samples_file,
        ploidy_level=2,
        ancestral_alleles=map_aa
    )

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

    return None


if __name__ == "__main__":
    run_standard_tsinfer_pipeline()
