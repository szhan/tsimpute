import click
from datetime import datetime
import json
import sys
from git import Repo
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
    "--in_file_type",
    type=click.Choice(["trees", "samples"]),
    required=True,
    help="Does the input file contain 'trees' or 'samples'?",
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
    help="Remove leaves when making ancestors tree sequence.",
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
@click.option(
    "--min_maf",
    type=float,
    default=0,
    help="Minimum threshold on MAF in the reference samples.",
)
@click.option(
    "--min_iqs",
    type=float,
    default=float("-Inf"),
    help="Minimum threshold on IQS, which can be negative.",
)
@click.option(
    "--max_iqs",
    type=float,
    default=1.0,
    help="Maximum threshold on IQS.",
)
@click.option(
    "--flip_alleles",
    is_flag=True,
    default=False,
    help="Flip ancestral alleles and derived alleles in imputed genotypes before computing IQS.",
)
def evaluate_imputation(
    in_imputed_file,
    in_file_type,
    in_true_samples_file,
    in_reference_trees_file,
    remove_leaves,
    in_chip_file,
    out_csv_file,
    min_maf,
    min_iqs,
    max_iqs,
    flip_alleles,
):
    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"INFO: START {start_datetime}")

    print(f"DEPS: tskit {tskit.__version__}")
    print(f"DEPS: tsinfer {tsinfer.__version__}")
    repo = Repo(search_parent_directories=True)
    print(f"DEPS: tsimpute URL {repo.remotes.origin.url}")
    print(f"DEPS: tsimpute SHA {repo.head.object.hexsha}")

    data_imputed = (
        tskit.load(in_imputed_file)
        if in_file_type == "trees"
        else tsinfer.load(in_imputed_file)
    )
    sd_true = tsinfer.load(in_true_samples_file)
    ts_ref = tskit.load(in_reference_trees_file)

    ts_anc = tsinfer.eval_util.make_ancestors_ts(ts=ts_ref, remove_leaves=remove_leaves)

    data_imputed_site_pos = (
        data_imputed.sites_position
        if in_file_type == "trees"
        else data_imputed.sites_position[:]
    )
    sd_true_site_pos = sd_true.sites_position[:]
    ts_ref_site_pos = ts_ref.sites_position
    ts_anc_site_pos = ts_anc.sites_position

    # Define chip and mask sites relative to the ancestors ts from the reference ts.
    chip_site_pos_all = masks.parse_site_position_file(in_chip_file)
    chip_site_pos = np.sort(list(set(ts_anc_site_pos) & set(chip_site_pos_all)))
    mask_site_pos = set(ts_anc_site_pos) - set(chip_site_pos)  # Must NOT be in chip set
    mask_site_pos = mask_site_pos & set(sd_true_site_pos)  # Must be in ground-truth set
    mask_site_pos = np.sort(
        list(mask_site_pos & set(data_imputed_site_pos))
    )  # Must be in imputed set

    assert set(mask_site_pos).issubset(set(data_imputed_site_pos))
    assert set(mask_site_pos).issubset(set(sd_true_site_pos))
    assert set(mask_site_pos).issubset(set(ts_ref_site_pos))
    assert set(mask_site_pos).issubset(set(ts_anc_site_pos))

    vars_data_imputed = data_imputed.variants()
    vars_sd_true = sd_true.variants()
    vars_ts_ref = ts_ref.variants()
    vars_ts_anc = ts_anc.variants()

    v_data_imputed = next(vars_data_imputed)
    v_sd_true = next(vars_sd_true)
    v_ts_ref = next(vars_ts_ref)
    v_ts_anc = next(vars_ts_anc)

    # Get a simplified ts for arity calculations.
    ts_ref_simp = ts_ref.simplify()

    results = None
    for pos in tqdm.tqdm(mask_site_pos):
        while v_data_imputed.site.position != pos:
            v_data_imputed = next(vars_data_imputed)
        while v_sd_true.site.position != pos:
            v_sd_true = next(vars_sd_true)
        while v_ts_ref.site.position != pos:
            v_ts_ref = next(vars_ts_ref)
        while v_ts_anc.site.position != pos:
            v_ts_anc = next(vars_ts_anc)

        # Variant objects have ordered lists of alleles.
        ref_ancestral_allele = v_ts_ref.alleles[0]  # Denoted by 0
        ref_derived_allele = v_ts_ref.alleles[1]  # Denoted by 1

        # Check ancestral alleles are identical.
        assert ref_ancestral_allele == v_data_imputed.site.ancestral_state
        assert ref_ancestral_allele == v_sd_true.site.ancestral_state

        # Get Minor Allele index and frequency from `ts_ref`.
        ref_freqs = v_ts_ref.frequencies(remove_missing=True)
        ref_af_0 = ref_freqs[ref_ancestral_allele]
        ref_af_1 = ref_freqs[ref_derived_allele]
        ref_ma_index = 1 if ref_af_1 < ref_af_0 else 0
        ref_ma_freq = ref_af_1 if ref_af_1 < ref_af_0 else ref_af_0

        # Skip sites with a MAF too low.
        if ref_ma_freq < min_maf:
            continue

        # Get Minor Allele index and frequency from `data_imputed`.
        if in_file_type == "trees":
            imputed_freqs = v_data_imputed.frequencies(remove_missing=True)
            imputed_af_0 = imputed_freqs[ref_ancestral_allele]
            imputed_af_1 = (
                imputed_freqs[ref_derived_allele]
                if ref_derived_allele in imputed_freqs
                else 0.0
            )
            imputed_ma_index = 1 if imputed_af_1 < imputed_af_0 else 0
            imputed_ma_freq = (
                imputed_af_1 if imputed_af_1 < imputed_af_0 else imputed_af_0
            )
        else:
            # Variant objects from SampleData do not yet have frequencies().
            # TODO: Update when they do have such functionality.
            imputed_ma_index = float("nan")
            imputed_ma_freq = float("nan")

        imputed_genotypes = v_data_imputed.genotypes
        if flip_alleles:
            # Flip alleles before computing IQS.
            # This is useful for checking whether the ancestral allele is wrong.
            imputed_genotypes = np.where(imputed_genotypes == 0, 1, 0)

        # Calculate imputation performance metrics.
        iqs = measures.compute_iqs(
            gt_true=v_sd_true.genotypes,
            gt_imputed=imputed_genotypes,
            ploidy=2,
        )

        # Skip sites with an IQS below or above a specified threshold.
        if iqs < min_iqs or iqs > max_iqs:
            continue

        # Obtain site information.
        # Get the mutations at this site.
        num_muts = np.sum(ts_ref.mutations_site == v_ts_ref.site.id)

        # Check whether the ancestral allele used to build the `ts_ref` is REF.
        ts_ref_site_metadata = json.loads(v_ts_ref.site.metadata)
        assert "REF" in ts_ref_site_metadata
        is_aa_ref = 1 if ts_ref_site_metadata["REF"] == ref_ancestral_allele else 0

        # Check whether the ancestral allele used to build the `ts_ref`
        # is best explained by parsimony using the inferred tree and observed genotypes.
        ts_ref_var = tskit.Variant(ts_ref)
        ts_ref_var.decode(site_id=v_ts_ref.site.id)
        ts_ref_tree = ts_ref.at(position=pos)
        parsimonious_aa, _ = ts_ref_tree.map_mutations(
            genotypes=ts_ref_var.genotypes, alleles=ts_ref_var.alleles
        )
        is_aa_parsimonious = 1 if ref_ancestral_allele == parsimonious_aa else 0

        # Calculate the mean arity of the tree covering this site position.
        #tree = ts_ref_simp.at(pos)  # Exclude unary nodes
        #parent_id, count = np.unique(
        #    tree.parent_array[tree.preorder()], return_counts=True
        #)
        #tree_arity = count[parent_id != tskit.NULL].mean()

        line = np.array(
            [
                [
                    pos,
                    ref_ma_index,
                    ref_ma_freq,
                    imputed_ma_index,
                    imputed_ma_freq,
                    iqs,
                    num_muts,
                    is_aa_ref,
                    is_aa_parsimonious,
                    #tree_arity,
                ],
            ]
        )

        results = line if results is None else np.append(results, line, axis=0)

    # Write results to file
    header_text = (
        "\n".join(
            [
                # Run information
                "#" + "tskit" + "=" + f"{tskit.__version__}",
                "#" + "tsinfer" + "=" + f"{tsinfer.__version__}",
                "#" + "in_imputed_file" + "=" + f"{in_imputed_file}",
                "#" + "in_true_samples_file" + "=" + f"{in_true_samples_file}",
                "#" + "in_reference_trees_file" + "=" + f"{in_reference_trees_file}",
                "#" + "remove_leaves" + "=" + f"{remove_leaves}",
                "#" + "in_chip_file" + "=" + f"{in_chip_file}",
                "#" + "out_csv_file" + "=" + f"{out_csv_file}",
                "#" + "min_maf" + "=" + f"{min_maf}",
                "#" + "min_iqs" + "=" + f"{min_iqs}",
                "#" + "max_iqs" + "=" + f"{max_iqs}",
                "#" + "flip_alleles" + "=" + f"{flip_alleles}",
                # Site statistics
                "#" + "num_sites_data_imputed" + "=" + f"{len(data_imputed_site_pos)}",
                "#" + "num_sites_sd_true" + "=" + f"{len(sd_true_site_pos)}",
                "#" + "num_sites_ts_ref" + "=" + f"{len(ts_ref_site_pos)}",
                "#" + "num_sites_ts_anc" + "=" + f"{len(ts_anc_site_pos)}",
                "#" + "num_chip_sites_all" + "=" + f"{len(chip_site_pos_all)}",
                "#" + "num_chip_sites" + "=" + f"{len(chip_site_pos)}",  # in ts_anc
                "#"
                + "num_mask_sites"
                + "="
                + f"{len(mask_site_pos)}",  # in ts_anc and data_imputed
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
            "num_muts",
            "is_aa_ref",
            "is_aa_parsimonious",
            #"tree_arity",
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

    end_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"INFO: END {end_datetime}")


if __name__ == "__main__":
    evaluate_imputation()
