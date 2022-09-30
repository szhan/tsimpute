import click
import sys
import tqdm
import numpy as np
import tskit
import tsinfer

sys.path.append("./src")
import masks
import measures


@click.command()
@click.option(
    "--in_imputed_file",
    "-i1",
    type=click.Path(exists=True),
    required=True,
    help="Input trees or samples file with imputed genotypes.",
)
@click.option(
    "--in_true_samples_file",
    "-i2",
    type=click.Path(exists=True),
    required=True,
    help="Input samples file with true genotypes.",
)
@click.option(
    "--in_reference_trees_file",
    "-r",
    type=click.Path(exists=True),
    required=True,
    help="Input tree sequence file with reference samples.",
)
@click.option(
    "--remove_leaves",
    type=bool,
    required=True,
    help="Remove leaves when making ancestors tree sequence."
)
@click.option(
    "--in_chip_file",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Input list of positions to retain before imputation.",
)
@click.option(
    "--out_csv_file",
    "-o",
    type=click.Path(exists=False),
    required=True,
    help="Output CSV file.",
)
def evaluate_imputation(
    in_imputed_file,
    in_true_samples_file,
    in_reference_trees_file,
    remove_leaves,
    in_chip_file,
    out_csv_file,
):
    # TODO: Take in samples file as well.
    ts_imputed = tskit.load(in_imputed_file)
    sd_true = tsinfer.load(in_true_samples_file)
    ts_ref = tskit.load(in_reference_trees_file)

    if tsinfer.__version__ == "0.2.4.dev27+gd61ae2f":
        ts_anc = tsinfer.eval_util.make_ancestors_ts(
            ts=ts_ref, remove_leaves=remove_leaves
        )
    else:
        # The samples argument is not actually used.
        ts_anc = tsinfer.eval_util.make_ancestors_ts(
            samples=None, ts=ts_ref, remove_leaves=remove_leaves
    )

    ts_imputed_site_pos = ts_imputed.sites_position # Mask sites and chip sites
    sd_true_site_pos = sd_true.sites_position[:]
    #ts_ref_site_pos = ts_ref.sites_position
    ts_anc_site_pos = ts_anc.sites_position # Mask sites and chip sites

    assert set(ts_anc_site_pos).issubset(set(ts_imputed_site_pos))
    assert set(ts_anc_site_pos).issubset(set(sd_true_site_pos))

    # Define mask site positions relative to the ancestors ts of the ref. panel.
    chip_site_pos = masks.parse_site_position_file(in_chip_file)
    mask_site_pos = np.sort(list(set(ts_anc_site_pos) - set(chip_site_pos)))

    vars_ts_imputed = ts_imputed.variants()
    vars_sd_true = sd_true.variants()
    vars_ts_ref = ts_ref.variants()
    vars_ts_anc = ts_anc.variants()

    v_ts_imputed = next(vars_ts_imputed)
    v_sd_true = next(vars_sd_true)
    v_ts_ref = next(vars_ts_ref)
    v_ts_anc = next(vars_ts_anc)

    results = None
    for pos in tqdm.tqdm(mask_site_pos):
        while v_ts_imputed.site.position != pos:
            v_ts_imputed = next(vars_ts_imputed)
        while v_sd_true.site.position != pos:
            v_sd_true = next(vars_sd_true)
        while v_ts_ref.site.position != pos:
            v_ts_ref = next(vars_ts_ref)
        while v_ts_anc.site.position != pos:
            v_ts_anc = next(vars_ts_anc)
        
        # Variant objects have ordered lists of alleles.
        ref_ancestral_allele = v_ts_ref.alleles[0]  # Denoted by 0
        ref_derived_allele = v_ts_ref.alleles[1]    # Denoted by 1

        # CHECK that ancestral states are identical.
        assert ref_ancestral_allele == v_ts_imputed.site.ancestral_state
        assert ref_ancestral_allele == v_sd_true.site.ancestral_state

        # Get Minor Allele index and frequency from `ts_imputed`.
        imputed_freqs = v_ts_imputed.frequencies(remove_missing=True)
        imputed_af_0 = imputed_freqs[ref_ancestral_allele]
        imputed_af_1 = imputed_freqs[ref_derived_allele] if ref_derived_allele in imputed_freqs else 0.0

        if imputed_af_1 < imputed_af_0:
            imputed_ma_index = 1
            imputed_ma_freq = imputed_af_1
        else:
            imputed_ma_index = 0
            imputed_ma_freq = imputed_af_0

        # Get Minor Allele index and frequency from `ts_ref`.
        ref_freqs = v_ts_ref.frequencies(remove_missing=True)
        ref_af_0 = ref_freqs[ref_ancestral_allele]
        ref_af_1 = ref_freqs[ref_derived_allele]

        if ref_af_1 < ref_af_0:
            ref_ma_index = 1
            ref_ma_freq = ref_af_1
        else:
            ref_ma_index = 0
            ref_ma_freq = ref_af_0

        # Calculate imputation performance metrics
        iqs = measures.compute_iqs(
            gt_true=v_sd_true.genotypes,
            gt_imputed=v_ts_imputed.genotypes,
            ploidy=2,
        )

        line = np.array(
            [
                [
                    pos,
                    ref_ma_index,
                    ref_ma_freq,
                    imputed_ma_index,
                    imputed_ma_freq,
                    iqs,
                ],
            ]
        )

        results = line if results is None else np.append(
            results, line, axis=0)

    # Write results to file
    header_text = (
        "\n".join(
            [
                "#" + "tskit" + "=" + f"{tskit.__version__}",
                "#" + "tsinfer" + "=" + f"{tsinfer.__version__}",
                "#" + "in_imputed_file" + "=" + f"{in_imputed_file}",
                "#" + "in_true_samples_file" + "=" + f"{in_true_samples_file}",
                "#" + "in_reference_trees_file" + "=" + f"{in_reference_trees_file}",
                "#" + "in_chip_file" + "=" + f"{in_chip_file}",
                "#" + "num_chip_sites" + "=" + f"{len(chip_site_pos)}",
                "#" + "num_mask_sites" + "=" + f"{len(mask_site_pos)}",
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
            "iqs",
        ]
    )

    np.savetxt(
        out_csv_file,
        results,
        fmt="%.10f",
        delimiter=",",
        newline="\n",
        comments="",
        header=header_text,
    )


if __name__ == "__main__":
    evaluate_imputation()
