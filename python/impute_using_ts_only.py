#!/usr/bin/env python
# coding: utf-8

import click
import gzip
from datetime import datetime

import numpy as np

import msprime
import tskit
import tsinfer
from tsinfer import make_ancestors_ts


### Helper functions
def count_sites_by_type(
    ts_or_sd
):
    """
    Iterate through the variants of a TreeSequence or SampleData object,
    and count the number of mono-, bi-, tri-, and quad-allelic sites.
    
    :param TreeSequence/SampleData ts_or_sd:
    """
    assert isinstance(ts_or_sd, tskit.TreeSequence) or\
        isinstance(ts_or_sd, tsinfer.SampleData)
    
    sites_mono = 0
    sites_bi = 0
    sites_bi_singleton = 0
    sites_tri = 0
    sites_quad = 0
    
    for v in ts_or_sd.variants():
        num_alleles = len(set(v.alleles) - {None})
        if num_alleles == 1:
            sites_mono += 1
        elif num_alleles == 2:
            sites_bi += 1
            if np.sum(v.genotypes) == 1:
                sites_bi_singleton += 1
        elif num_alleles == 3:
            sites_tri += 1
        else:
            sites_quad += 1
    
    sites_total = sites_mono + sites_bi + sites_tri + sites_quad
    
    print(f"\tsites mono : {sites_mono}")
    print(f"\tsites bi   : {sites_bi} ({sites_bi_singleton} singletons)")
    print(f"\tsites tri  : {sites_tri}")
    print(f"\tsites quad : {sites_quad}")
    print(f"\tsites total: {sites_total}")


def check_site_positions_ts_issubset_sd(
    tree_sequence,
    sample_data
):
    """
    Check whether the site positions in `TreeSequence` are a subset of
    the site positions in `SampleData`.
    
    :param TreeSequence tree_sequence:
    :param SampleData sample_data:
    :return bool:
    """
    ts_site_positions = np.empty(tree_sequence.num_sites)
    sd_site_positions = np.empty(sample_data.num_sites)
    
    i = 0
    for v in tree_sequence.variants():
        ts_site_positions[i] = v.site.position
        i += 1
        
    j = 0
    for v in sample_data.variants():
        sd_site_positions[j] = v.site.position
        j += 1
        
    assert i == tree_sequence.num_sites
    assert j == sample_data.num_sites
    
    if set(ts_site_positions).issubset(set(sd_site_positions)):
        return True
    else:
        return False


def compare_sites_sd_and_ts(
    sample_data,
    tree_sequence,
    is_common,
    check_matching_ancestral_state=True
):
    """
    If `include` is set to True, then get the ids of the sites
    found in `sample_data` AND in `tree_sequence`.
    
    if `exclude` is set to False, then get the ids of the sites
    found in `sample_data` but NOT in `tree_sequence`.
    
    :param TreeSequence tree_sequence:
    :param SampleData sample_data:
    :param include bool:
    :return np.array:
    """
    ts_site_positions = np.empty(tree_sequence.num_sites)
    
    i = 0
    for v in tree_sequence.variants():
        ts_site_positions[i] = v.site.position
        i += 1
        
    assert i == tree_sequence.num_sites
    
    sd_site_id = []
    for v in sample_data.variants():
        if is_common:
            if v.site.position in ts_site_positions:
                sd_site_id.append(v.site.id)
                if check_matching_ancestral_state:
                    ts_site = tree_sequence.site(position=v.site.position)
                    assert v.site.ancestral_state == ts_site.ancestral_state,\
                        f"Ancestral states at position {v.site.position} not identical, " +\
                        f"{v.site.ancestral_state} vs. {ts_site.ancestral_state}."
        else:
            if v.site.position not in ts_site_positions:
                sd_site_id.append(v.site.id)
    
    return(np.array(sd_site_id))


def make_compatible_sample_data(
    sample_data,
    ancestors_ts
):
    """
    Make an editable copy of a `sample_data` object, and edit it so that:
    (1) the derived alleles in the `sample_data` object not found in `ancestors_ts` are marked as MISSING;
    (2) the allele list in the new `sample_data` corresponds to the allele list in `ancestors_ts`.
    
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
    sd_var = next(sd_variants)
    for ts_site in ancestors_ts.sites():
        while sd_var.site.position != ts_site.position:
            # Sites in `samples_data` but not in `ancestors_ts` are not imputed.
            # Also, leave them as is in the `sample_data`, but keep track of them.
            sd_var = next(sd_variants)
            
        sd_site_id = sd_var.site.id # Site id in `sample_data`
        
        # CHECK that all the sites in `ancestors_ts` are biallelic.
        assert len(ts_site.alleles) == 2
        
        # Get the derived allele in `ancestors_ts` in nucleotide space
        ts_ancestral_allele = ts_site.ancestral_state
        ts_derived_allele = ts_site.alleles - {ts_ancestral_allele}
        assert len(ts_derived_allele) == 1 # CHECK
        ts_derived_allele = tuple(ts_derived_allele)[0]
        
        # CHECK that the ancestral allele should be the same
        # in both `ancestors_ts` and `sample_data`.
        assert ts_ancestral_allele == sd_var.alleles[0]
        
        if ts_derived_allele not in sd_var.alleles:
            # Case 1:
            # If the derived alleles in the `sample_data` are not in `ancestors_ts`,
            # then mark them as missing.
            #
            # The site in `sample_data` may be mono-, bi-, or multiallelic.
            #
            # We cannot determine whether the extra derived alleles in `sample_data`
            # are derived from 0 or 1 in `ancestors_ts` anyway.
            new_sample_data.sites_genotypes[sd_site_id] = np.where(
                sd_var.genotypes != 0, # Keep if ancestral
                tskit.MISSING_DATA, # Otherwise, flag as missing
                0,
            )
            print(f"Site {sd_site_id} has no matching derived alleles in the query samples.")
            # Update allele list
            new_sample_data.sites_alleles[sd_site_id] = [ts_ancestral_allele]
        else:
            # The allele lists in `ancestors_ts` and `sample_data` may be different.
            ts_derived_allele_index = sd_var.alleles.index(ts_derived_allele)
            
            if ts_derived_allele_index == 1:
                # Case 2:
                # Both the ancestral and derived alleles correspond exactly.
                if len(sd_var.alleles) == 2:
                    continue
                # Case 3:
                # The derived allele in `ancestors_ts` is indexed as 1 in `sample_data`,
                # so mark alleles >= 2 as missing.
                new_sample_data.sites_genotypes[sd_site_id] = np.where(
                    sd_var.genotypes > 1, # 0 and 1 should be kept "as is"
                    tskit.MISSING_DATA, # Otherwise, flag as missing
                    sd_var.genotypes,
                )
                print(f"Site {sd_site_id} has extra derived allele(s) in the query samples (set as missing).")
            else:
                # Case 4:
                #   The derived allele in `ancestors_ts` is NOT indexed as 1 in `sample_data`,
                #   so the alleles in `sample_data` needs to be reordered,
                #   such that the 1-indexed allele is also indexed as 1 in `ancestors_ts`.
                new_sample_data.sites_genotypes[sd_site_id] = np.where(
                    sd_var.genotypes == 0,
                    0, # Leave ancestral allele "as is"
                    np.where(
                        sd_var.genotypes == ts_derived_allele_index,
                        1, # Change it to 1 so that it corresponds to `ancestors_ts`
                        tskit.MISSING_DATA, # Otherwise, mark as missing
                    ),
                )
                print(f"Site {sd_site_id} has the target derived allele at a different index.")
            # Update allele list
            new_sample_data.sites_alleles[sd_site_id] = [ts_ancestral_allele, ts_derived_allele]
            
    new_sample_data.finalise()
    
    return(new_sample_data)


def pick_masked_sites_random(
    site_ids,
    prop_masked_sites
):
    """
    Draw N sites from `sites_ids` at random, where N is the number of sites to mask
    based on a specified proportion of masked sites `prop_masked_sites`.
    
    TODO: Specify random seed.
    
    :param np.array site_ids:
    :param float prop_masked_sites: float between 0 and 1
    :return np.array: list of site ids
    """
    assert prop_masked_sites >= 0
    assert prop_masked_sites <= 1
    
    rng = np.random.default_rng()
    
    num_masked_sites = int(np.floor(len(site_ids) * prop_masked_sites))
    
    masked_site_ids = np.sort(
        rng.choice(
            site_ids,
            num_masked_sites,
            replace=False,
        )
    )
    
    return(masked_site_ids)


def mask_sites_in_sample_data(
    sample_data,
    masked_sites=None
):
    """
    Create and return a `SampleData` object from an existing `SampleData` object,
    which contain masked sites as listed in `masked_sites` (site ids).
    
    :param SampleData sample_data:
    :param np.array masked_sites: list of site ids (NOT positions)
    :return SampleData:
    """
    new_sample_data = sample_data.copy()
    
    for v in sample_data.variants():
        if v.site.id in masked_sites:
            new_sample_data.sites_genotypes[v.site.id] = np.full_like(v.genotypes, tskit.MISSING_DATA)
    
    new_sample_data.finalise()
    
    return(new_sample_data)


def compute_iqs(
    genotypes_true,
    genotypes_imputed
):
    """
    Calculate the Imputation Quality Score between `genotypes_true` and `genotypes_imputed`.
    
    This specific formula is used to compute the IQS of imputed genotypes
    at biallelic sites in haploid genomes.
    """
    assert set(genotypes_true) == set([0, 1]),\
        f"True genotypes are non-biallelic {set(genotypes_true)}."
    assert set(genotypes_imputed) == set([0, 1]),\
        f"Imputed genotypes are non-biallelic {set(genotypes_imputed)}."
    
    # Allele 0 imputed correctly
    n00 = np.sum([y == 0 for x, y in zip(genotypes_imputed, genotypes_true) if x == 0])
    # Allele 1 imputed correctly
    n11 = np.sum([y == 1 for x, y in zip(genotypes_imputed, genotypes_true) if x == 1])
    # Allele 1 imputed wrongly
    n01 = np.sum([y == 1 for x, y in zip(genotypes_imputed, genotypes_true) if x == 0])
    # Allele 1 imputed wrongly
    n10 = np.sum([y == 0 for x, y in zip(genotypes_imputed, genotypes_true) if x == 1])
    
    # Marginal counts
    n0_ = n00 + n01
    n1_ = n10 + n11
    n_0 = n00 + n10
    n_1 = n01 + n11
    
    # Total genotypes imputed
    n__ = n00 + n10 + n01 + n11
    
    # Observed overall concordance
    Po = float(n00 + n11) / float(n__)
    
    # Chance agreement
    Pc = float(n0_ * n_0 + n1_ * n_1) / float(n__ * n__)
    
    assert Po >= 0 and Po <= 1
    assert Pc >= 0 and Pc <= 1
    
    iqs = (Po - Pc) / (1 - Pc)
    
    return(iqs)


@click.command()
@click.option(
    '--replicate_index', '-i',
    required=True,
    type=int,
    help="Replicate index"
    )
@click.option(
    '--sampling_time_query', '-t',
    required=True,
    type=float,
    help="Sampling time of query genomes"
    )
@click.option(
    '--prop_missing_sites', '-p',
    required=True,
    type=float,
    help="Proportion of missing sites"
    )
def run_pipeline(
    replicate_index,
    sampling_time_query,
    prop_missing_sites,
):
    ### Set simulation parameters
    # For simulations
    size_ref   = 1e4
    size_query = 1e3
    eff_pop_size = 10_000
    mutation_rate = 1e-8
    recombination_rate = 1e-8
    sequence_length = 1_000_000

    # For testing
    #size_ref   = 50
    #size_query = 50
    #eff_pop_size = 10_000
    #mutation_rate = 1e-6
    #recombination_rate = 1e-7
    #sequence_length = 10_000

    contig_id = '1'
    ploidy_level = 1

    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    ### Simulate genealogy and genetic variation
    # Uniform recombination rate
    recomb_rate_map = msprime.RateMap.uniform(
        sequence_length=sequence_length,
        rate=recombination_rate,
    )

    # Uniform mutation rate
    mut_rate_map = msprime.RateMap.uniform(
        sequence_length=sequence_length,
        rate=mutation_rate,
    )

    sample_set = [
        # Reference genomes
        msprime.SampleSet(num_samples=size_ref,
                        time=0,
                        ploidy=ploidy_level),
        # Query genomes
        msprime.SampleSet(num_samples=size_query,
                        time=sampling_time_query,
                        ploidy=ploidy_level),
    ]

    # A simulated tree sequence does not contain any monoallelic sites,
    # but there may be multiallelic sites.
    ts_full = msprime.sim_mutations(
        msprime.sim_ancestry(
            samples=sample_set,
            population_size=eff_pop_size,
            model="hudson",
            recombination_rate=recomb_rate_map,
            discrete_genome=True,
        ),
        rate=mut_rate_map,
        discrete_genome=True,
    )
    # Remove populations
    tables = ts_full.dump_tables()
    tables.populations.clear()
    tables.nodes.population = np.full_like(tables.nodes.population, tskit.NULL)
    ts_full = tables.tree_sequence()

    print("TS full")
    count_sites_by_type(ts_full)

    # The first `size_ref` individuals or `ploidy_level` * `size_ref` samples are taken as the reference panel.
    # The remaining individuals and samples are the query/target to impute into.
    individuals_ref = np.arange(size_ref, dtype=int)
    samples_ref = np.arange(ploidy_level * size_ref, dtype=int)

    individuals_query = np.arange(size_ref, size_ref + size_query, dtype=int)
    samples_query = np.arange(ploidy_level * size_ref, ploidy_level * (size_ref + size_query), dtype=int)

    ### Create an ancestor ts from the reference genomes
    # Remove all the branches leading to the query genomes
    ts_ref = ts_full.simplify(samples_ref, filter_sites=False)

    print(f"TS ref has {ts_ref.num_samples} sample genomes ({ts_ref.sequence_length} bp)")
    print(f"TS ref has {ts_ref.num_sites} sites and {ts_ref.num_trees} trees")
    print("TS ref")
    count_sites_by_type(ts_ref)

    # Multiallelic sites are automatically removed when generating an ancestor ts.
    # Sites which are biallelic in the full sample set but monoallelic in the ref. sample set are removed.
    # So, only biallelic sites are retained in the ancestor ts.
    ts_anc = make_ancestors_ts(ts=ts_ref, remove_leaves=True, samples=None)

    print(f"TS anc has {ts_anc.num_samples} sample genomes ({ts_anc.sequence_length} bp)")
    print(f"TS anc has {ts_anc.num_sites} sites and {ts_anc.num_trees} trees")
    print("TS anc")
    count_sites_by_type(ts_anc)

    ### Create a SampleData object holding the query genomes 
    sd_full = tsinfer.SampleData.from_tree_sequence(ts_full)
    sd_query = sd_full.subset(individuals_query)

    print(f"SD query has {sd_query.num_samples} sample genomes ({sd_query.sequence_length} bp)")
    print(f"SD query has {sd_query.num_sites} sites")
    print("SD query")
    count_sites_by_type(sd_query)

    assert check_site_positions_ts_issubset_sd(ts_anc, sd_query)

    new_sd_query = make_compatible_sample_data(
        sample_data=sd_query,
        ancestors_ts=ts_anc,
    )

    ### Create a SampleData object with masked sites
    # Identify sites in both `sd_query` and `ts_anc`.
    shared_sites = compare_sites_sd_and_ts(sd_query, ts_anc, is_common=True)
    print(f"Shared sites: {len(shared_sites)}")

    # Identify sites in `sd_query` but not in `ts_anc`, which are not to be imputed.
    exclude_sites = compare_sites_sd_and_ts(sd_query, ts_anc, is_common=False)
    print(f"Exclude sites: {len(exclude_sites)}")

    assert len(set(shared_sites).intersection(set(exclude_sites))) == 0

    masked_site_ids = pick_masked_sites_random(
        site_ids=shared_sites,
        prop_masked_sites=prop_missing_sites,
    )

    sd_query_masked = mask_sites_in_sample_data(
        new_sd_query,
        masked_sites=masked_site_ids,
    )

    ### Impute the query genomes
    ts_imp = tsinfer.match_samples(
        sample_data=sd_query_masked,
        ancestors_ts=ts_anc,
    )

    ### Evaluate imputation performance
    assert ts_ref.num_sites == ts_imp.num_sites

    ts_ref_site_pos = [site.position for site in ts_ref.sites()]
    ts_imp_site_pos = [site.position for site in ts_imp.sites()]

    assert len(set(ts_ref_site_pos).intersection(set(ts_imp_site_pos))) == len(ts_ref_site_pos)

    results = None
    for v_ref, v_imp in zip(ts_ref.variants(), ts_imp.variants()):
        if v_imp.site.id in shared_sites:
            assert v_ref.alleles[0] == v_imp.alleles[0]
            # TODO:
            #   Why doesn't `v.num_alleles` always reflect the number of genotypes
            #   after simplifying?
            if len(set(v_ref.genotypes)) == 1 or len(set(v_imp.genotypes)) == 1:
                # Monoallelic sites in `ts_ref` or `sd_query` are not imputed
                continue
            assert v_ref.num_alleles == 2
            assert v_imp.num_alleles == 2
            
            freqs_ref = v_ref.frequencies()
            freqs_imp = v_imp.frequencies()
            
            # Note: A minor allele in `ts_ref` may be a major allele in `sd_query`
            af_0 = freqs_ref[v_ref.alleles[0]]
            af_1 = freqs_ref[v_ref.alleles[1]]
            
            # Get MAF from `ts_ref`
            if af_1 < af_0:
                minor_allele_index = 1
                maf = af_1
            assert not np.any(v_imp.genotypes == -1)
            
            # Assess imputation performance
            total_concordance = np.sum(v_ref.genotypes == v_imp.genotypes) / len(v_ref.genotypes)
            iqs = compute_iqs(genotypes_true=v_ref.genotypes, genotypes_imputed=v_imp.genotypes)
            
            # line.shape = (1, 4)
            line = np.array([ [v_ref.site.position, maf, total_concordance, iqs], ])
            if results is None:
                results = line
            else:
                results = np.append(results, line, axis=0)

    end_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    ### Write results
    out_results_file = "sim" + "_" + str(replicate_index) + ".csv"

    header_text = "\n".join(
        [
            "#" + "start_timestamp" + "=" + f"{start_datetime}",
            "#" + "end_timestamp" + "=" + f"{end_datetime}",
            "#" + "msprime" + "=" + f"{msprime.__version__}",
            "#" + "tskit" + "=" + f"{tskit.__version__}",
            "#" + "tsinfer" + "=" + f"{tsinfer.__version__}",
            "#" + "replicate" + "=" + f"{replicate_index}",
            "#" + "size_ref" + "=" + f"{size_ref}",
            "#" + "size_query" + "=" + f"{size_query}",
            "#" + "eff_pop_size" + "=" + f"{eff_pop_size}",
            "#" + "mutation_rate" + "=" + f"{mutation_rate}",
            "#" + "recombination_rate" + "=" + f"{recombination_rate}",
            "#" + "contig_id" + "=" + f"{contig_id}",
            "#" + "ploidy_level" + "=" + f"{ploidy_level}",
            "#" + "sequence_length" + "=" + f"{sequence_length}",
            "#" + "sampling_time_query" + "=" + f"{sampling_time_query}",
            "#" + "prop_missing_sites" + "=" + f"{prop_missing_sites}",
        ]
    ) + "\n"

    header_text += ",".join(
        [
            "position",
            "maf",
            "total_concordance",
            "iqs",
        ]
    )

    np.savetxt(
        out_results_file,
        results,
        fmt='%.10f',
        delimiter=",",
        newline="\n",
        comments="",
        header=header_text,
    )


if __name__ == '__main__':
    run_pipeline()
