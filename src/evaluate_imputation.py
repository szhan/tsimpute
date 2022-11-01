import click
from datetime import datetime
import json
import sys
from git import Repo
from tqdm import tqdm
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
    in_file_type,
    in_true_samples_file,
    in_reference_trees_file,
    in_chip_file,
    out_csv_file,
):
    start_dt = datetime.now()
    start_dt_str = start_dt.strftime("%d/%m/%Y %H:%M:%S")
    print(f"INFO: START {start_dt_str}")

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

    data_imputed_site_pos = (
        data_imputed.sites_position
        if in_file_type == "trees"
        else data_imputed.sites_position[:]
    )
    sd_true_site_pos = sd_true.sites_position[:]
    ts_ref_site_pos = ts_ref.sites_position

    # Define chip and mask sites relative to the ref. ts.
    chip_site_pos_all = masks.parse_site_position_file(in_chip_file, one_based=True)
    ts_ref_sites_isin_chip = np.isin(
        ts_ref_site_pos, chip_site_pos_all, assume_unique=True,
    )
    chip_site_pos = ts_ref_site_pos[ts_ref_sites_isin_chip]
    mask_site_pos = ts_ref_site_pos[np.invert(ts_ref_sites_isin_chip)]

    #mask_site_pos = mask_site_pos & set(sd_true_site_pos)  # Must be in truth set
    #mask_site_pos = np.sort(
    #    list(mask_site_pos & set(data_imputed_site_pos))
    #)  # Must be in imputed set

    assert set(mask_site_pos).issubset(set(data_imputed_site_pos))
    assert set(mask_site_pos).issubset(set(sd_true_site_pos))
    assert set(mask_site_pos).issubset(set(ts_ref_site_pos))

    vars_data_imputed = data_imputed.variants()
    vars_sd_true = sd_true.variants()
    vars_ts_ref = ts_ref.variants()

    v_data_imputed = next(vars_data_imputed)
    v_sd_true = next(vars_sd_true)
    v_ts_ref = next(vars_ts_ref)

    # Get a simplified ts for arity calculations.
    #ts_ref_simp = ts_ref.simplify()

    results = None
    for pos in tqdm(mask_site_pos):
        while v_data_imputed.site.position != pos:
            v_data_imputed = next(vars_data_imputed)
        while v_sd_true.site.position != pos:
            v_sd_true = next(vars_sd_true)
        while v_ts_ref.site.position != pos:
            v_ts_ref = next(vars_ts_ref)

        # Variant objects have ordered lists of alleles.
        ref_ancestral_allele = v_ts_ref.alleles[0]
        ref_derived_allele = v_ts_ref.alleles[1]

        # Check ancestral alleles are identical.
        assert ref_ancestral_allele == v_data_imputed.site.ancestral_state
        assert ref_ancestral_allele == v_sd_true.site.ancestral_state

        # Get Minor Allele index and frequency from ref. ts.
        ref_freqs = v_ts_ref.frequencies(remove_missing=True)
        ref_af_0 = ref_freqs[ref_ancestral_allele]
        ref_af_1 = ref_freqs[ref_derived_allele]
        ref_ma_index = 1 if ref_af_1 < ref_af_0 else 0
        ref_ma_freq = ref_af_1 if ref_af_1 < ref_af_0 else ref_af_0

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
            # Variant objects from SampleData do not have frequencies().
            # TODO: Update if they get such functionality.
            imputed_ma_index = float("nan")
            imputed_ma_freq = float("nan")

        imputed_genotypes = v_data_imputed.genotypes

        # Calculate imputation performance metrics.
        iqs = measures.compute_iqs(
            gt_true=v_sd_true.genotypes,
            gt_imputed=imputed_genotypes,
            ploidy=2,
        )

        # Obtain site information.
        # Get the mutations at this site.
        num_muts = np.sum(ts_ref.mutations_site == v_ts_ref.site.id)

        # Check whether the ancestral allele used to build the ref. ts is REF.
        ts_ref_site_metadata = json.loads(v_ts_ref.site.metadata)
        assert "REF" in ts_ref_site_metadata
        is_aa_ref = 1 if ts_ref_site_metadata["REF"] == ref_ancestral_allele else 0

        # Check whether the ancestral allele used to build the ref. ts
        # is best explained by parsimony using the imputed tree and genotypes.
        is_aa_parsimonious = -1
        if in_file_type == "trees":
            data_imputed_tree = data_imputed.at(position=pos)
            parsimonious_aa, _ = data_imputed_tree.map_mutations(
                genotypes=v_data_imputed.genotypes, alleles=v_data_imputed.alleles
            )
            is_aa_parsimonious = 1 if ref_ancestral_allele == parsimonious_aa else 0

        # Determine the proportion of ancestral alleles wrongly imputed as derived alleles.
        prop_wrong_alleles_0 = 0
        num_wrongly_imputed_alleles = np.sum(
            v_sd_true.genotypes != imputed_genotypes)
        if num_wrongly_imputed_alleles > 0:
            prop_wrong_alleles_0 = 1 - np.sum(
                v_sd_true.genotypes[np.where(v_sd_true.genotypes != imputed_genotypes)]
            ) / num_wrongly_imputed_alleles

        # Calculate the mean arity of the tree covering this site position.
        # tree = ts_ref_simp.at(pos)  # Exclude unary nodes
        # parent_id, count = np.unique(
        #    tree.parent_array[tree.preorder()], return_counts=True
        # )
        # tree_arity = count[parent_id != tskit.NULL].mean()

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
                    prop_wrong_alleles_0,
                    # tree_arity,
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
                "#" + "in_chip_file" + "=" + f"{in_chip_file}",
                "#" + "out_csv_file" + "=" + f"{out_csv_file}",
                # Site statistics
                "#" + "num_sites_data_imputed" + "=" + f"{len(data_imputed_site_pos)}",
                "#" + "num_sites_sd_true" + "=" + f"{len(sd_true_site_pos)}",
                "#" + "num_sites_ts_ref" + "=" + f"{len(ts_ref_site_pos)}",
                "#" + "num_chip_sites_all" + "=" + f"{len(chip_site_pos_all)}",
                "#" + "num_chip_sites" + "=" + f"{len(chip_site_pos)}",
                "#"
                + "num_mask_sites"
                + "="
                + f"{len(mask_site_pos)}",  # in ts_ref and data_imputed
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
            "prop_wrong_alleles_0",
            # "tree_arity",
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

    end_dt = datetime.now()
    end_dt_str = end_dt.strftime("%d/%m/%Y %H:%M:%S")
    print(f"INFO: END {end_dt_str}")
    print(f"INFO: DURATION {str(end_dt - start_dt)}")


if __name__ == "__main__":
    evaluate_imputation()
