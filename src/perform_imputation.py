import click
from datetime import datetime
import sys
import numpy as np
import tskit
import tsinfer

sys.path.append("./python")
import masks
import measures
import util


@click.command()
@click.option(
    "--in_reference_trees_file",
    "-i1",
    type=click.Path(exists=True),
    required=True,
    help="Input tree sequence file with reference samples.",
)
@click.option(
    "--in_target_samples_file",
    "-i2",
    type=click.Path(exists=True),
    required=True,
    help="Input samples file with target samples.",
)
@click.option(
    "--in_chip_file",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Input list of positions to retain before imputation.",
)
@click.option(
    "--out_dir",
    "-o",
    type=click.Path(exists=True),
    required=True,
    help="Output directory.",
)
@click.option(
    "--out_prefix", "-p", type=str, required=True, help="Prefix of the output file."
)
@click.option(
    "--recombination_rate",
    "-r",
    type=float,
    default=None,
    help="Uniform recombination rate",
)
@click.option(
    "--mmr_samples",
    "-s",
    type=float,
    default=None,
    help="Mismatch ratio used when matching sample haplotypes",
)
@click.option("--num_threads", "-t", type=int, default=1, help="Number of CPUs.")
def run_pipeline(
    in_reference_trees_file,
    in_target_samples_file,
    in_chip_file,
    out_dir,
    out_prefix,
    recombination_rate,
    mmr_samples,
    num_threads,
):
    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    ts_ref = tskit.load(in_reference_trees_file)
    sd_target = tsinfer.load(in_target_samples_file)

    chip_site_pos = masks.parse_site_position_file(in_chip_file)
    mask_site_pos = []

    ts_anc = tsinfer.eval_util.make_ancestors_ts(ts=ts_ref, remove_leaves=True)

    sd_compat = util.make_compatible_sample_data(sd_target, ts_anc)

    for v in sd_compat.variants():
        if v.site.position not in chip_site_pos:
            mask_site_pos.append(v.site.position)

    assert (
        len(set(chip_site_pos) & set(mask_site_pos)) == 0
    ), f"Chip and mask site positions are not mutually exclusive."

    sd_masked = masks.mask_sites_in_sample_data(
        sd_compat, sites=mask_site_pos, site_type="position"
    )

    ts_imputed = tsinfer.match_samples(
        sample_data=sd_masked,
        ancestors_ts=ts_anc,
        recombination_rate=recombination_rate,
        mismatch_ratio=mmr_samples,
        num_threads=num_threads,
    )

    assert (
        v_ref.num_sites
        == v_compat.num_sites
        == v_masked.num_sites
        == v_imputed.num_sites
    ), f"Different number of sites in the tree sequences and sample data."

    ### Evaluate imputation performance
    results = None
    num_non_biallelic_masked_sites = 0

    for v_ref, v_compat, v_masked, v_imputed in zip(
        ts_ref.variants(),  # Reference genomes from which to get the minor allele and MAF
        sd_compat.variants(),  # Query genomes BEFORE site masking
        sd_masked.variants(),  # Query genomes AFTER site masking
        ts_imputed.variants(),  # Query genomes with imputed sites
    ):
        if v_imputed.site.position in mask_site_pos:
            if len(set(v_ref.alleles) - {None}) != 2:
                num_non_biallelic_masked_sites += 1
                continue

            ref_ancestral_allele = v_ref.alleles[0]
            ref_derived_allele = v_ref.alleles[1]

            # CHECK that ancestral states are identical.
            assert ref_ancestral_allele == sd_compat.sites_alleles[v_compat.site.id][0]
            assert ref_ancestral_allele == sd_masked.sites_alleles[v_masked.site.id][0]
            assert ref_ancestral_allele == v_imputed.alleles[0]

            assert set(v_masked.genotypes) == set([-1])
            assert not np.any(v_imputed.genotypes == -1)

            # Get Minor Allele index and frequency from `ts_ref`.
            # Definition of a minor allele: MAF < 0.50.
            ref_freqs = v_ref.frequencies()
            ref_af_0 = ref_freqs[ref_ancestral_allele]
            ref_af_1 = ref_freqs[ref_derived_allele]

            if ref_af_1 < ref_af_0:
                ref_ma_index = 1
                ref_ma_freq = ref_af_1
            else:
                ref_ma_index = 0
                ref_ma_freq = ref_af_0

            # Get Minor Allele index and frequency from `ts_imputed`.
            imputed_freqs = v_imputed.frequencies()
            imputed_af_0 = imputed_freqs[ref_ancestral_allele]
            imputed_af_1 = imputed_freqs[ref_derived_allele]

            if imputed_af_1 < imputed_af_0:
                imputed_ma_index = 1
                imputed_ma_freq = imputed_af_1
            else:
                imputed_ma_index = 0
                imputed_ma_freq = imputed_af_0

            # Assess imputation performance
            total_concordance = measures.compute_concordance(
                genotypes_true=v_compat.genotypes,
                genotypes_imputed=v_imputed.genotypes,
            )
            iqs = measures.compute_iqs(
                genotypes_true=v_compat.genotypes,
                genotypes_imputed=v_imputed.genotypes,
            )

            # line.shape = (1, 7)
            line = np.array(
                [
                    [
                        v_ref.site.position,
                        ref_ma_index,
                        ref_ma_freq,
                        imputed_ma_index,
                        imputed_ma_freq,
                        total_concordance,
                        iqs,
                    ],
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
                "#" + "masked_sites" + "=" + f"{len(mask_site_pos)}",
                "#"
                + "non_biallelic_masked_sites"
                + "="
                + f"{num_non_biallelic_masked_sites}",
            ]
        )
        + "\n"
    )

    header_text += ",".join(
        [
            "position",
            "ref_ma_index",
            "ref_ma_freq",
            "imputed_ma_index",
            "imputed_ma_freq",
            "total_concordance",
            "iqs",
        ]
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
