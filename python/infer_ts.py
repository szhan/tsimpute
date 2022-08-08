import tskit
import tsinfer

import sys

sys.path.append("python/")
import read_vcf


def infer_ts(
    ref_vcf_file,
    miss_vcf_file,
    imputed_vcf_file,
    contig_id
):
    """
    TODO
    """
    sd_ref = read_vcf.create_sample_data_from_vcf_file(ref_vcf_file)
    sd_miss = read_vcf.create_sample_data_from_vcf_file(miss_vcf_file)
    ad_ref = tsinfer.generate_ancestors(sample_data=sd_ref)

    # This step is to infer a tree sequence from the sample data.
    ts_anc_ref = tsinfer.match_ancestors(
        sample_data=sd_ref,
        ancestor_data=ad_ref
    )

    ts_matched = tsinfer.match_samples(
        sample_data=sd_miss,
        ancestors_ts=ts_anc_ref
    )

    with open(imputed_vcf_file, "w") as vcf:
        ts_matched.write_vcf(vcf, contig_id = contig_id)
