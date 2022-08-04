from re import I
import msprime
import tskit
import tsinfer

import numpy as np


# TODO: Use a HapMap recombination rate map

def simulate_ts(
    size_ref,
    size_query,
    eff_pop_size,
    mutation_rate,
    recombination_rate,
    sequence_length,
    ploidy_level=1,
    sampling_time_query=0
):
    ### Simulate genealogy and genetic variation
    # Uniform recombination rate
    recomb_rate_map = msprime.RateMap.uniform(
        sequence_length=sequence_length,
        rate=recombination_rate
    )

    # Uniform mutation rate
    mut_rate_map = msprime.RateMap.uniform(
        sequence_length=sequence_length,
        rate=mutation_rate
    )

    sample_set = [
        # Reference genomes
        msprime.SampleSet(num_samples=size_ref, time=0, ploidy=ploidy_level),
        # Query genomes
        msprime.SampleSet(
            num_samples=size_query, time=sampling_time_query, ploidy=ploidy_level
        )
    ]

    # A simulated tree sequence does not contain any monoallelic sites,
    # but there may be multiallelic sites.
    ts = msprime.sim_mutations(
        msprime.sim_ancestry(
            samples=sample_set,
            population_size=eff_pop_size,
            model="hudson",
            recombination_rate=recomb_rate_map,
            discrete_genome=True
        ),
        rate=mut_rate_map,
        discrete_genome=True
    )

    # Remove multi-allelic sites
    non_biallelic_sites = [v.site.id for v in ts.variants() if v.num_alleles != 2]
    ts = ts.delete_sites(site_ids=non_biallelic_sites)

    # Remove populations
    tables = ts.dump_tables()
    tables.populations.clear()
    tables.nodes.population = np.full_like(tables.nodes.population, tskit.NULL)
    ts = tables.tree_sequence()

    return(ts)


def get_ref_query_indices(size_ref, size_query, ploidy_level):
    inds_ref = np.arange(size_ref, dtype=int)
    samples_ref = np.arange(ploidy_level * size_ref, dtype=int)

    inds_query = np.arange(size_ref, size_ref + size_query, dtype=int)
    samples_query = np.arange(
        ploidy_level * size_ref, ploidy_level * (size_ref + size_query), dtype=int
    )

    return([inds_ref, samples_ref, inds_query, samples_query])


def get_ts_toy():
    """
    Simulate a simple `TreeSequence` object for doing test runs.

    :param None:
    :return TreeSequence:
    """
    ploidy_level = 1

    size_ref = 50
    size_query = 50
    eff_pop_size = 10_000
    mutation_rate = 1e-7
    recombination_rate = 1e-7
    sequence_length = 10_000

    return(
        [simulate_ts(
            size_ref=size_ref,
            size_query=size_query,
            eff_pop_size=eff_pop_size,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            sequence_length=sequence_length,
            ploidy_level=1,
            sampling_time_query=0
        )] +\
        get_ref_query_indices(
            size_ref,
            size_query,
            ploidy_level
        )
    )


def get_ts_single_panmictic(sampling_time_query):
    """
    Simulate a `TreeSequence` object under a single panmictic population.

    :param None:
    :return TreeSequence:
    """
    ploidy_level = 1

    size_ref = 1e4
    size_query = 1e3
    eff_pop_size = 10_000
    mutation_rate = 1e-8
    recombination_rate = 1e-8
    sequence_length = 1_000_000

    return(
        (
            simulate_ts(
                size_ref=size_ref,
                size_query=size_query,
                eff_pop_size=eff_pop_size,
                mutation_rate=mutation_rate,
                recombination_rate=recombination_rate,
                sequence_length=sequence_length,
                ploidy_level=1,
                sampling_time_query=sampling_time_query
            ),
            get_ref_query_indices(
                size_ref,
                size_query,
                ploidy_level
            )
        )
    )


def get_ts_seven_pop():
    """
    TODO: Use a more complex demographic model versus a single panmictic model
    """
    pass
