import click
from datetime import datetime
import sys
from pathlib import Path

import numpy as np

import tskit
import tsinfer

sys.path.append("./python")
import masks
import measures
import util


### Helper functions
@click.command()
@click.option(
    "--in_reference_trees_file", "-r",
    type=click.Path(exists=True),
    required=True,
    help="Input tree sequence file with reference samples."
)
@click.option(
    "--in_target_samples_file", "-t",
    type=click.Path(exists=True),
    required=True,
    help="Input samples file with target samples."
)
@click.option(
    "--in_chip_file", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Input list of positions to retain before imputation."
)
@click.option(
    "--out_dir", "-o",
    type=click.Path(exists=True),
    required=True,
    help="Output directory."
)
@click.option(
    "--out_prefix", "-p",
    type=str,
    required=True,
    help="Prefix of the output file."
)
def run_pipeline(
    in_reference_trees_file,
    in_target_samples_file,
    in_chip_file,
    out_dir,
    out_prefix,
):
    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    ts_ref = tskit.load(in_reference_trees_file)
    sd_target = tsinfer.load(in_target_samples_file)

    assert util.is_biallelic(ts_ref), f"Tree sequence has non-biallelic sites."
    assert util.is_biallelic(sd_target), f"Sample data has non-biallelic sites."

    chip_site_pos = masks.parse_site_position_file(in_chip_file)
    mask_site_pos = []

    ts_anc = tsinfer.eval_util.make_ancestors_ts(ts=ts_ref, remove_leaves=True)

    sd_compat = util.make_compatible_sample_data(sd_target, ts_anc)

    for v in sd_compat.variants():
        if v.site.position not in chip_site_pos:
            mask_site_pos.append(v.site.position)

    assert len(set(chip_site_pos) & set(mask_site_pos)) == 0, \
        f"Chip and mask site positions are not mutually exclusive."

    sd_masked = masks.mask_sites_in_sample_data(
        sd_compat, sites=mask_site_pos, site_type="position"
    )

    ts_imputed = tsinfer.match_samples(sample_data=sd_masked, ancestors_ts=ts_anc)

    ### Evaluate imputation performance
    results = None
    for v_ref, v_compat, v_masked, v_imputed in zip(
        ts_ref.variants(),  # Reference genomes from which to get the minor allele and MAF
        sd_compat.variants(),  # Query genomes before site masking
        sd_masked.variants(),  # Query genomes after site masking
        ts_imputed.variants(),  # Query genomes with imputed sites
    ):
        if v_imputed.site.position in mask_site_pos:
            # CHECK that ancestral states are identical.
            assert (
                v_ref.alleles[0] == sd_compat.sites_alleles[v_compat.site.id][0]
            )
            assert (
                v_ref.alleles[0] == sd_masked.sites_alleles[v_masked.site.id][0]
            )
            assert v_ref.alleles[0] == v_imputed.alleles[0]

            assert set(v_masked.genotypes) == set([-1])
            assert not np.any(v_imputed.genotypes == -1)

            # Note: A minor allele in `ts_ref` may be a major allele in `sd_query`.
            freqs_ref = v_ref.frequencies()
            af_0 = freqs_ref[v_ref.alleles[0]]
            af_1 = freqs_ref[v_ref.alleles[1]]

            # Get Minor Allele Index and Frequency from `ts_ref`.
            # Definition of a minor allele: MAF < 0.50.
            if af_1 < af_0:
                ref_ma_index = 1
                ref_ma_freq = af_1
            else:
                ref_ma_index = 0
                ref_ma_freq = af_0

            # Assess imputation performance
            total_concordance = measures.compute_concordance(
                genotypes_true=v_compat.genotypes,
                genotypes_imputed=v_imputed.genotypes,
            )
            iqs = measures.compute_iqs(
                genotypes_true=v_compat.genotypes,
                genotypes_imputed=v_imputed.genotypes,
            )

            # line.shape = (1, 4)
            line = np.array(
                [
                    [v_ref.site.position, ref_ma_index, ref_ma_freq, total_concordance, iqs],
                ]
            )
            if results is None:
                results = line
            else:
                results = np.append(results, line, axis=0)

    end_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    ### Write results
    out_file = out_dir + "/" + out_prefix + "imputation.csv"

    header_text = (
        "\n".join(
            [
                "#" + "start_timestamp" + "=" + f"{start_datetime}",
                "#" + "end_timestamp" + "=" + f"{end_datetime}",
                "#" + "tskit" + "=" + f"{tskit.__version__}",
                "#" + "tsinfer" + "=" + f"{tsinfer.__version__}",
                "#" + "size_ref" + "=" + f"{ts_ref.num_samples}",
                "#" + "size_query" + "=" + f"{sd_compat.num_samples}",
            ]
        )
        + "\n"
    )

    header_text += ",".join(
        ["position", "ref_ma_index", "ref_ma_freq", "total_concordance", "iqs"]
    )

    np.savetxt(
        out_file,
        results,
        fmt="%.10f",
        delimiter=",",
        newline="\n",
        comments="",
        header=header_text,
    )


if __name__ == "__main__":
    run_pipeline()
