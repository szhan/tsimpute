import click
from datetime import datetime
import sys
import numpy as np
import tqdm
import tskit
import tsinfer

sys.path.append("./src")
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
    print("INFO: START")
    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    print("INFO: Loading input files")
    ts_ref = tskit.load(in_reference_trees_file)
    sd_target = tsinfer.load(in_target_samples_file)

    chip_site_pos = masks.parse_site_position_file(in_chip_file)

    print("INFO: Making ancestors tree sequence")
    if tsinfer.__version__ == "0.2.4.dev27+gd61ae2f":
        ts_anc = tsinfer.eval_util.make_ancestors_ts(ts=ts_ref, remove_leaves=True)
    else:
        # The samples argument is not actually used.
        ts_anc = tsinfer.eval_util.make_ancestors_ts(
            samples=None, ts=ts_ref, remove_leaves=True
        )

    print("INFO: Making samples compatible with the ancestors tree sequence")
    sd_compat = util.make_compatible_sample_data(sd_target, ts_anc)

    sd_compat_sites_isnotin_chip = np.isin(
        sd_compat.sites_position[:], chip_site_pos, assume_unique=True, invert=True
    )
    mask_site_pos = sd_compat.sites_position[:][sd_compat_sites_isnotin_chip]

    assert (
        len(set(chip_site_pos) & set(mask_site_pos)) == 0
    ), f"Chip and mask site positions are not mutually exclusive."

    print("INFO: Masking sites")
    sd_masked = masks.mask_sites_in_sample_data(
        sd_compat, sites=mask_site_pos, site_type="position"
    )

    print("INFO: Imputing target samples")
    ts_imputed = tsinfer.match_samples(
        sample_data=sd_masked,
        ancestors_ts=ts_anc,
        recombination_rate=recombination_rate,
        mismatch_ratio=mmr_samples,
        num_threads=num_threads,
    )

    assert (
        ts_ref.num_sites
        == sd_compat.num_sites
        == sd_masked.num_sites
        == ts_imputed.num_sites
    ), f"Different number of sites in the tree sequences and sample data."

    print("INFO: Evaluating imputation performance")
    results = None
    num_non_biallelic_masked_sites = 0

    for v_ref, v_compat, v_masked, v_imputed in tqdm.tqdm(
        zip(
            ts_ref.variants(),  # Reference genomes from which to get the minor allele and MAF
            sd_compat.variants(),  # Query genomes BEFORE site masking
            sd_masked.variants(),  # Query genomes AFTER site masking
            ts_imputed.variants(),  # Query genomes with imputed sites
        )
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
            ref_freqs = v_ref.frequencies(
                remove_missing=True
            )  # Dict: allele -> frequency
            ref_af_0 = ref_freqs[ref_ancestral_allele]
            ref_af_1 = ref_freqs[ref_derived_allele]

            if ref_af_1 < ref_af_0:
                ref_ma_index = 1
                ref_ma_freq = ref_af_1
            else:
                ref_ma_index = 0
                ref_ma_freq = ref_af_0

            # Get Minor Allele index and frequency from `ts_imputed`.
            imputed_freqs = v_imputed.frequencies(remove_missing=True)
            imputed_af_0 = imputed_freqs[ref_ancestral_allele]
            imputed_af_1 = (
                imputed_freqs[ref_derived_allele] if imputed_af_0 < 1.0 else 0.0
            )

            if imputed_af_1 < imputed_af_0:
                imputed_ma_index = 1
                imputed_ma_freq = imputed_af_1
            else:
                imputed_ma_index = 0
                imputed_ma_freq = imputed_af_0

            assert np.sum(v_imputed.genotypes) == 0
            
            # Assess imputation performance
            total_concordance = measures.compute_concordance(
                genotypes_true=v_compat.genotypes,
                genotypes_imputed=v_imputed.genotypes,
            )
            iqs = measures.compute_iqs(
                genotypes_true=v_compat.genotypes[:20],
                genotypes_imputed=v_imputed.genotypes[:20],
                ploidy=2,
            )
            print(v_compat.genotypes[:20])
            print(v_imputed.genotypes[:20])
            print(iqs)

            # line.shape = (1, 7)
            line = np.array(
                [
                    [
                        int(v_ref.site.position),
                        int(ref_ma_index),
                        ref_ma_freq,
                        int(imputed_ma_index),
                        imputed_ma_freq,
                        total_concordance,
                        iqs,
                    ],
                ]
            )

            results = line if results is None else np.append(results, line, axis=0)

    end_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    print("INFO: Writing results to file")
    out_file = out_dir + "/" + out_prefix + ".imputation.csv"

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
            "ref_minor_allele_index",
            "ref_minor_allele_freq",
            "imputed_minor_allele_index",
            "imputed_minor_allele_freq",
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
