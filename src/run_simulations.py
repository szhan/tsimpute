import click
from datetime import datetime
import json
import sys
import numpy as np
import msprime
import tskit
import tsinfer

sys.path.append("./src")
import masks
import measures
import util
import simulate_ts


@click.command()
@click.option("--index", "-i", type=int, required=True, help="Replicate index.")
@click.option(
    "--sampling_time",
    "-s",
    type=float,
    required=True,
    help="Time to sample query genomes.",
)
@click.option(
    "--prop_mask_sites",
    "-p",
    type=float,
    required=True,
    help="Proportion of sites to mask.",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["test", "simple", "ten_pop"], case_sensitive=False),
    required=True,
    help="Perform a run using pre-set simulation parameters.",
)
@click.option(
    "--pop_ref",
    type=click.Choice(["YRI", "CHB", "CEU"], case_sensitive=False),
    default=None,
    help="Population of reference genomes. Used only if model ten_pop is set.",
)
@click.option(
    "--pop_query",
    type=click.Choice(["YRI", "CHB", "CEU"], case_sensitive=False),
    default=None,
    help="Population of query genomes. Used only if model ten_pop is set.",
)
@click.option("--out_dir", "-o", type=click.Path(exists=True), help="Output directory")
@click.option(
    "--out_prefix", type=str, default="sim", help="Prefix of the output file."
)
@click.option("--num_threads", "-t", type=int, default=1, help="Number of CPUs.")
@click.option(
    "--verbose",
    is_flag=True,
    help="Print out site information after each processing step.",
)
def run_pipeline(
    index,
    sampling_time,
    prop_mask_sites,
    model,
    pop_ref,
    pop_query,
    out_dir,
    out_prefix,
    num_threads,
    verbose,
):
    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"INFO: START {start_datetime}")

    if model == "test":
        ts_full, _, samples_ref, inds_query, _ = simulate_ts.get_ts_toy()
    elif model == "simple":
        ts_full, _, samples_ref, inds_query, _ = simulate_ts.get_ts_single_panmictic(
            time_query=sampling_time
        )
    elif model == "ten_pop":
        assert pop_ref is not None and pop_query is not None
        ts_full, _, samples_ref, inds_query, _ = simulate_ts.get_ts_ten_pop(
            pop_ref=pop_ref, pop_query=pop_query
        )

    if verbose:
        print("INFO: TS full")
        util.count_sites_by_type(ts_full)

    ### Create an ancestor ts from the reference genomes
    # Remove all the branches leading to the query genomes.
    ts_ref = ts_full.simplify(samples_ref, filter_sites=False)

    if verbose:
        print(f"INFO: TS ref genomes {ts_ref.num_samples}")
        print(f"INFO: TS ref sites {ts_ref.num_sites}")
        print(f"INFO: TS ref trees {ts_ref.num_trees}")
        print(f"INFO: TS ref stats")
        util.count_sites_by_type(ts_ref)

    # Multiallelic sites are automatically removed when generating an ancestor ts.
    # Sites which are biallelic in the full sample set but monoallelic in the ref. sample set are removed.
    # So, only biallelic sites are retained in the ancestor ts.
    if tsinfer.__version__ == "0.2.4.dev27+gd61ae2f":
        ts_anc = tsinfer.eval_util.make_ancestors_ts(ts=ts_ref, remove_leaves=True)
    else:
        # The samples argument is not actually used.
        ts_anc = tsinfer.eval_util.make_ancestors_ts(
            samples=None, ts=ts_ref, remove_leaves=True
        )

    if verbose:
        print(f"INFO: TS anc genomes {ts_anc.num_samples}")
        print(f"INFO: TS anc sites {ts_anc.num_sites}")
        print(f"INFO: TS anc trees {ts_anc.num_trees}")
        print(f"INFO: TS anc stats")
        util.count_sites_by_type(ts_anc)

    ### Create a SampleData object holding the query genomes
    sd_full = tsinfer.SampleData.from_tree_sequence(ts_full)
    sd_query = sd_full.subset(inds_query)

    if verbose:
        print(f"INFO: SD query genomes {sd_query.num_samples}")
        print(f"INFO: SD query sites {sd_query.num_sites}")
        print(f"INFO: SD query stats")
        util.count_sites_by_type(sd_query)

    assert util.check_site_positions_ts_issubset_sd(ts_anc, sd_query)

    sd_query_true = util.make_compatible_sample_data(
        sample_data=sd_query, ancestors_ts=ts_anc
    )

    ### Create a SampleData object with masked sites
    # Identify sites which are in both `ts_anc` and `sd_query`.
    # This is a superset of the sites in `sd_query` to be masked and imputed.
    shared_site_ids, shared_site_pos = util.compare_sites_sd_and_ts(
        sd_query_true, ts_anc, is_common=True
    )

    if verbose:
        print(f"INFO: shared sites {len(shared_site_ids)}")

    # Identify sites in `sd_query` but not in `ts_anc`, which are not to be imputed.
    exclude_site_ids, exclude_site_pos = util.compare_sites_sd_and_ts(
        sd_query_true, ts_anc, is_common=False
    )

    if verbose:
        print(f"INFO: exclude sites {len(exclude_site_ids)}")

    assert len(set(shared_site_ids) & set(exclude_site_ids)) == 0
    assert len(set(shared_site_pos) & set(exclude_site_pos)) == 0

    # Select mask sites in `ts_anc` to impute.
    # This is a subset of 'shared_site_ids'
    mask_site_ids = masks.pick_mask_sites_random(
        sites=shared_site_ids,
        prop_mask_sites=prop_mask_sites,
    )
    mask_site_pos = sd_query_true.sites_position[:][mask_site_ids]

    if verbose:
        print(f"INFO: mask sites {len(mask_site_ids)}")

    assert set(mask_site_ids).issubset(set(shared_site_ids))
    assert set(mask_site_pos).issubset(set(shared_site_pos))

    sd_query_mask = masks.mask_sites_in_sample_data(
        sample_data=sd_query_true, sites=mask_site_pos, site_type="position"
    )

    ### Impute the query genomes
    ts_imputed = tsinfer.match_samples(
        sample_data=sd_query_mask, ancestors_ts=ts_anc, num_threads=num_threads
    )

    ### Evaluate imputation performance
    ts_ref_site_pos = ts_ref.sites_position
    sd_query_true_site_pos = sd_query_true.sites_position[:]
    sd_query_mask_site_pos = sd_query_mask.sites_position[:]
    ts_imputed_site_pos = ts_imputed.sites_position

    assert len(ts_ref_site_pos) == len(sd_query_true_site_pos)
    assert len(ts_ref_site_pos) == len(sd_query_mask_site_pos)
    assert len(ts_ref_site_pos) == len(ts_imputed_site_pos)

    assert set(ts_ref_site_pos) == set(sd_query_true_site_pos)
    assert set(ts_ref_site_pos) == set(sd_query_mask_site_pos)
    assert set(ts_ref_site_pos) == set(ts_imputed_site_pos)

    results = None
    for v_ref, v_query_true, v_query_mask, v_query_imputed in zip(
        ts_ref.variants(),  # Reference genomes from which to get the minor allele and MAF
        sd_query_true.variants(),  # Query genomes BEFORE masking sites
        sd_query_mask.variants(),  # Query genomes AFTER masking sites
        ts_imputed.variants(),  # Query genomes with mask sites imputed
    ):
        pos = v_ref.site.position
        if pos in mask_site_pos:
            # CHECK that ancestral states are identical.
            ref_ancestral_allele = v_ref.alleles[0]
            ref_derived_allele = v_ref.alleles[1]

            assert (
                ref_ancestral_allele
                == sd_query_true.sites_alleles[v_query_true.site.id][0]
            )
            assert (
                ref_ancestral_allele
                == sd_query_mask.sites_alleles[v_query_mask.site.id][0]
            )
            assert ref_ancestral_allele == v_query_imputed.alleles[0]

            # TODO:
            #   Why doesn't `v.num_alleles` always reflect the number of genotypes
            #   after simplifying?
            if len(set(v_ref.genotypes)) == 1:
                # Monoallelic sites in `ts_ref` are not imputed
                # TODO: Revisit
                continue

            assert v_ref.num_alleles == 2
            assert set(v_query_mask.genotypes) == set([-1])
            assert not np.any(v_query_imputed.genotypes == -1)

            # Note: A minor allele in `ts_ref` may be a major allele in `sd_query`
            ref_freqs = v_ref.frequencies()
            ref_af_0 = ref_freqs[ref_ancestral_allele]
            ref_af_1 = ref_freqs[ref_derived_allele]

            # Get MAF from `ts_ref`
            ma_index = 1 if ref_af_1 < ref_af_0 else 0
            ma_freq = ref_af_1 if ref_af_1 < ref_af_0 else ref_af_0

            # Assess imputation performance
            iqs = measures.compute_iqs(
                gt_true=v_query_true.genotypes,
                gt_imputed=v_query_imputed.genotypes,
                ploidy=1,
            )

            line = np.array([[pos, ma_index, ma_freq, iqs]])
            results = line if results is None else np.append(results, line, axis=0)

    end_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"INFO: END {end_datetime}")

    ### Write results to file
    out_csv_file = out_dir + "/" + out_prefix + "_" + str(index) + ".csv"

    ### Get parameter values from provenances
    prov = [p for p in ts_full.provenances()]
    prov_anc = json.loads(prov[0].record)
    prov_mut = json.loads(prov[1].record)
    assert prov_anc["parameters"]["command"] == "sim_ancestry"
    assert prov_mut["parameters"]["command"] == "sim_mutations"
    seed_anc = prov_anc["parameters"]["random_seed"]
    seed_mut = prov_mut["parameters"]["random_seed"]

    eff_pop_size = prov_anc["parameters"]["population_size"]
    recombination_rate = prov_anc["parameters"]["recombination_rate"]
    mutation_rate = prov_mut["parameters"]["rate"]
    ploidy_level = prov_anc["parameters"]["ploidy"]
    sequence_length = prov_anc["parameters"]["sequence_length"]

    header_text = (
        "\n".join(
            [
                "#" + "start_timestamp" + "=" + f"{start_datetime}",
                "#" + "end_timestamp" + "=" + f"{end_datetime}",
                "#" + "msprime" + "=" + f"{msprime.__version__}",
                "#" + "tskit" + "=" + f"{tskit.__version__}",
                "#" + "tsinfer" + "=" + f"{tsinfer.__version__}",
                "#" + "seed_sim_ancestry" + "=" + f"{seed_anc}",
                "#" + "seed_sim_mutations" + "=" + f"{seed_mut}",
                "#" + "index" + "=" + f"{index}",
                "#" + "size_ref" + "=" + f"{ts_ref.num_samples}",
                "#" + "size_query" + "=" + f"{sd_query.num_samples}",
                "#" + "sampling_time" + "=" + f"{sampling_time}",
                "#" + "prop_missing_sites" + "=" + f"{prop_mask_sites}",
                "#" + "eff_pop_size" + "=" + f"{eff_pop_size}",
                "#" + "recombination_rate" + "=" + f"{recombination_rate}",
                "#" + "mutation_rate" + "=" + f"{mutation_rate}",
                "#" + "ploidy_level" + "=" + f"{ploidy_level}",
                "#" + "sequence_length" + "=" + f"{sequence_length}",
                "#" + "model" + "=" + f"{model}",
                "#" + "pop_ref" + "=" + f"{pop_ref}",
                "#" + "pop_query" + "=" + f"{pop_query}",
            ]
        )
        + "\n"
    )

    header_text += ",".join(
        ["position", "ref_minor_allele_index", "ref_minor_allele_freq", "iqs"]
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

    print("INFO: Finished writing results to file")


if __name__ == "__main__":
    run_pipeline()
