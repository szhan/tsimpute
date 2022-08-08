#!/usr/bin/env python
# coding: utf-8

import click
from datetime import datetime
import gzip
import json
import sys

import numpy as np

import msprime
import tskit
import tsinfer
from tsinfer import make_ancestors_ts

sys.path.append("python/")
import masks
import measures
import util
import sim_ts


### Helper functions
def make_compatible_sample_data(sample_data, ancestors_ts):
    """
    Make an editable copy of a `sample_data` object, and edit it so that:
    (1) the derived alleles in `sample_data` not in `ancestors_ts` are marked as MISSING;
    (2) the allele list in `new_sample_data` corresponds to the allele list in `ancestors_ts`.

    N.B. Two `SampleData` attributes `sites_alleles` and `sites_genotypes`,
    which are not explained in the tsinfer API doc, are used to facilitate the editing.

    :param SampleData sample_data:
    :param TreeSequence ancestors_ts:
    :return SampleData:
    """
    new_sample_data = sample_data.copy()

    # Iterate through the sites in `ancestors_ts` using one generator,
    # while iterating through the sites in `sample_data` using another generator,
    # letting the latter generator catch up.
    sd_variants = sample_data.variants()
    sd_v = next(sd_variants)
    for ts_site in ancestors_ts.sites():
        while sd_v.site.position != ts_site.position:
            # Sites in `samples_data` but not in `ancestors_ts` are not imputed.
            # Also, leave them as is in the `sample_data`, but keep track of them.
            sd_v = next(sd_variants)

        sd_site_id = sd_v.site.id  # Site id in `sample_data`

        # CHECK that all the sites in `ancestors_ts` are biallelic.
        assert len(ts_site.alleles) == 2

        # Get the derived allele in `ancestors_ts` in nucleotide space
        ts_ancestral_allele = ts_site.ancestral_state
        ts_derived_allele = ts_site.alleles - {ts_ancestral_allele}
        assert len(ts_derived_allele) == 1  # CHECK
        ts_derived_allele = tuple(ts_derived_allele)[0]

        # CHECK that the ancestral allele should be the same
        # in both `ancestors_ts` and `sample_data`.
        assert ts_ancestral_allele == sd_v.alleles[0]

        if ts_derived_allele not in sd_v.alleles:
            # Case 1:
            # If the derived alleles in the `sample_data` are not in `ancestors_ts`,
            # then mark them as missing.
            #
            # The site in `sample_data` may be mono-, bi-, or multiallelic.
            #
            # We cannot determine whether the extra derived alleles in `sample_data`
            # are derived from 0 or 1 in `ancestors_ts` anyway.
            new_sample_data.sites_genotypes[sd_site_id] = np.where(
                sd_v.genotypes != 0,  # Keep if ancestral
                tskit.MISSING_DATA,  # Otherwise, flag as missing
                0,
            )
            print(
                f"Site {sd_site_id} has no matching derived alleles in the query samples."
            )
            # Update allele list
            new_sample_data.sites_alleles[sd_site_id] = [ts_ancestral_allele]
        else:
            # The allele lists in `ancestors_ts` and `sample_data` may be different.
            ts_derived_allele_index = sd_v.alleles.index(ts_derived_allele)

            if ts_derived_allele_index == 1:
                # Case 2:
                # Both the ancestral and derived alleles correspond exactly.
                if len(sd_v.alleles) == 2:
                    continue
                # Case 3:
                # The derived allele in `ancestors_ts` is indexed as 1 in `sample_data`,
                # so mark alleles >= 2 as missing.
                new_sample_data.sites_genotypes[sd_site_id] = np.where(
                    sd_v.genotypes > 1,  # 0 and 1 should be kept "as is"
                    tskit.MISSING_DATA,  # Otherwise, flag as missing
                    sd_v.genotypes,
                )
                print(
                    f"Site {sd_site_id} has extra derived allele(s) in the query samples (set as missing)."
                )
            else:
                # Case 4:
                #   The derived allele in `ancestors_ts` is NOT indexed as 1 in `sample_data`,
                #   so the alleles in `sample_data` needs to be reordered,
                #   such that the 1-indexed allele is also indexed as 1 in `ancestors_ts`.
                new_sample_data.sites_genotypes[sd_site_id] = np.where(
                    sd_v.genotypes == 0,
                    0,  # Leave ancestral allele "as is"
                    np.where(
                        sd_v.genotypes == ts_derived_allele_index,
                        1,  # Change it to 1 so that it corresponds to `ancestors_ts`
                        tskit.MISSING_DATA,  # Otherwise, mark as missing
                    ),
                )
                print(
                    f"Site {sd_site_id} has the target derived allele at a different index."
                )
            # Update allele list
            new_sample_data.sites_alleles[sd_site_id] = [
                ts_ancestral_allele,
                ts_derived_allele,
            ]

    new_sample_data.finalise()

    return new_sample_data


@click.command()
@click.option("--index", "-i", type=int, required=True, help="Replicate index.")
@click.option(
    "--time_query",
    "-t",
    type=float,
    required=True,
    help="Time to sample query genomes.",
)
@click.option(
    "--prop_missing_sites",
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
@click.option("--out_prefix", type=str, default="sim", help="Prefix of the output file.")
@click.option(
    "--verbose",
    is_flag=True,
    help="Print out site information after each processing step.",
)
def run_pipeline(
    index,
    time_query,
    prop_missing_sites,
    out_prefix,
    model,
    pop_ref,
    pop_query,
    verbose,
):
    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    if model == "test":
        ts_full, _, samples_ref, inds_query, _ = sim_ts.get_ts_toy()
    elif model == "simple":
        ts_full, _, samples_ref, inds_query, _ = sim_ts.get_ts_single_panmictic(
            time_query=time_query
        )
    elif model == "ten_pop":
        assert pop_ref is not None and pop_query is not None
        ts_full, _, samples_ref, inds_query, _ = sim_ts.get_ts_ten_pop(
            pop_ref=pop_ref, pop_query=pop_query
        )

    if verbose:
        print("TS full")
        util.count_sites_by_type(ts_full)

    ### Create an ancestor ts from the reference genomes
    # Remove all the branches leading to the query genomes
    ts_ref = ts_full.simplify(samples_ref, filter_sites=False)

    if verbose:
        print(
            f"TS ref has {ts_ref.num_samples} sample genomes ({ts_ref.sequence_length} bp)"
        )
        print(f"TS ref has {ts_ref.num_sites} sites and {ts_ref.num_trees} trees")
        print("TS ref")
        util.count_sites_by_type(ts_ref)

    # Multiallelic sites are automatically removed when generating an ancestor ts.
    # Sites which are biallelic in the full sample set but monoallelic in the ref. sample set are removed.
    # So, only biallelic sites are retained in the ancestor ts.
    ts_anc = make_ancestors_ts(ts=ts_ref, remove_leaves=True)

    if verbose:
        print(
            f"TS anc has {ts_anc.num_samples} sample genomes ({ts_anc.sequence_length} bp)"
        )
        print(f"TS anc has {ts_anc.num_sites} sites and {ts_anc.num_trees} trees")
        print("TS anc")
        util.count_sites_by_type(ts_anc)

    ### Create a SampleData object holding the query genomes
    sd_full = tsinfer.SampleData.from_tree_sequence(ts_full)
    sd_query = sd_full.subset(inds_query)

    if verbose:
        print(
            f"SD query has {sd_query.num_samples} sample genomes ({sd_query.sequence_length} bp)"
        )
        print(f"SD query has {sd_query.num_sites} sites")
        print("SD query")
        util.count_sites_by_type(sd_query)

    assert util.check_site_positions_ts_issubset_sd(ts_anc, sd_query)

    sd_query_true = make_compatible_sample_data(
        sample_data=sd_query, ancestors_ts=ts_anc
    )

    ### Create a SampleData object with masked sites
    # Identify sites in both `sd_query` and `ts_anc`.
    # This is a superset of the sites in `sd_query` to be masked and imputed.
    shared_site_ids, shared_site_positions = util.compare_sites_sd_and_ts(
        sd_query_true, ts_anc, is_common=True
    )

    if verbose:
        print(f"Shared sites: {len(shared_site_ids)}")

    # Identify sites in `sd_query` but not in `ts_anc`, which are not to be imputed.
    exclude_site_ids, exclude_site_positions = util.compare_sites_sd_and_ts(
        sd_query_true, ts_anc, is_common=False
    )

    if verbose:
        print(f"Exclude sites: {len(exclude_site_ids)}")

    assert len(set(shared_site_ids).intersection(set(exclude_site_ids))) == 0
    assert (
        len(set(shared_site_positions).intersection(set(exclude_site_positions))) == 0
    )

    # Select sites in `sd_query` to mask and impute.
    # This is a subset of 'shared_site_ids'
    masked_site_ids = masks.pick_masked_sites_random(
        site_ids=shared_site_ids,
        prop_masked_sites=prop_missing_sites,
    )
    masked_site_positions = [
        s.position for s in sd_query_true.sites(ids=masked_site_ids)
    ]

    if verbose:
        print(f"Masked sites: {len(masked_site_ids)}")

    assert set(masked_site_ids).issubset(set(shared_site_ids))
    assert set(masked_site_positions).issubset(set(shared_site_positions))

    sd_query_masked = masks.mask_sites_in_sample_data(
        sd_query_true, masked_sites=masked_site_ids
    )

    ### Impute the query genomes
    ts_imputed = tsinfer.match_samples(sample_data=sd_query_masked, ancestors_ts=ts_anc)

    ### Evaluate imputation performance
    ts_ref_site_positions = [s.position for s in ts_ref.sites()]
    sd_query_true_site_positions = [s.position for s in sd_query_true.sites()]
    sd_query_masked_site_positions = [s.position for s in sd_query_masked.sites()]
    ts_imputed_site_positions = [s.position for s in ts_imputed.sites()]

    assert len(ts_ref_site_positions) == len(sd_query_true_site_positions)
    assert len(ts_ref_site_positions) == len(sd_query_masked_site_positions)
    assert len(ts_ref_site_positions) == len(ts_imputed_site_positions)

    assert set(ts_ref_site_positions) == set(sd_query_true_site_positions)
    assert set(ts_ref_site_positions) == set(sd_query_masked_site_positions)
    assert set(ts_ref_site_positions) == set(ts_imputed_site_positions)

    results = None
    for v_ref, v_query_true, v_query_masked, v_query_imputed in zip(
        ts_ref.variants(),  # Reference genomes from which to get the minor allele and MAF
        sd_query_true.variants(),  # Query genomes before site masking
        sd_query_masked.variants(),  # Query genomes with masked sites
        ts_imputed.variants(),  # Query genomes with masked sites imputed
    ):
        if v_query_imputed.site.position in masked_site_positions:
            # CHECK that ancestral states are identical.
            assert (
                v_ref.alleles[0] == sd_query_true.sites_alleles[v_query_true.site.id][0]
            )
            assert (
                v_ref.alleles[0]
                == sd_query_masked.sites_alleles[v_query_masked.site.id][0]
            )
            assert v_ref.alleles[0] == v_query_imputed.alleles[0]

            # TODO:
            #   Why doesn't `v.num_alleles` always reflect the number of genotypes
            #   after simplifying?
            if len(set(v_ref.genotypes)) == 1:
                # Monoallelic sites in `ts_ref` are not imputed
                # TODO: Revisit
                continue

            assert v_ref.num_alleles == 2
            assert set(v_query_masked.genotypes) == set([-1])
            assert not np.any(v_query_imputed.genotypes == -1)

            # Note: A minor allele in `ts_ref` may be a major allele in `sd_query`
            freqs_ref = v_ref.frequencies()
            af_0 = freqs_ref[v_ref.alleles[0]]
            af_1 = freqs_ref[v_ref.alleles[1]]

            # Get MAF from `ts_ref`
            # Definition of a minor allele: < 0.50
            if af_1 < af_0:
                minor_allele_index = 1
                maf = af_1
            else:
                minor_allele_index = 0
                maf = af_0

            # Assess imputation performance
            total_concordance = measures.compute_concordance(
                genotypes_true=v_query_true.genotypes,
                genotypes_imputed=v_query_imputed.genotypes,
            )
            iqs = measures.compute_iqs(
                genotypes_true=v_query_true.genotypes,
                genotypes_imputed=v_query_imputed.genotypes,
            )

            # line.shape = (1, 4)
            line = np.array(
                [
                    [v_ref.site.position, maf, total_concordance, iqs],
                ]
            )
            if results is None:
                results = line
            else:
                results = np.append(results, line, axis=0)

    end_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    ### Write results
    out_results_file = out_prefix + "_" + str(index) + ".csv"

    ### Get parameter values from provenances
    prov = [p for p in ts_full.provenances()]
    prov_anc = json.loads(prov[0].record)
    prov_mut = json.loads(prov[1].record)
    assert prov_anc["parameters"]["recombination_rate"]
    assert prov_mut["parameters"]["rate"]

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
                "#" + "index" + "=" + f"{index}",
                "#" + "size_ref" + "=" + f"{ts_ref.num_samples}",
                "#" + "size_query" + "=" + f"{sd_query.num_samples}",
                "#" + "time_query" + "=" + f"{time_query}",
                "#" + "prop_missing_sites" + "=" + f"{prop_missing_sites}",
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

    header_text += ",".join(["position", "maf", "total_concordance", "iqs"])

    np.savetxt(
        out_results_file,
        results,
        fmt="%.10f",
        delimiter=",",
        newline="\n",
        comments="",
        header=header_text,
    )


if __name__ == "__main__":
    run_pipeline()
