import demes
import msprime
import tskit
import tsinfer
import numpy as np


# TODO: Use a HapMap genetic map.

def simulate_ts(
    sample_set, demography, mutation_rate, recombination_rate, sequence_length
):
    """
    Simulate a tree sequence using `msprime` under a specified demographic model
    with genome-wide uniform mutation rate and recombination rate.

    :param msprime.Sample sample_set:
    :param msprime.Demography demography: If None, it defaults to a single population with constant size 1.
    :param float mutation_rate:
    :param float recombination_rate:
    :param float sequence_length:
    :return: A simulated tree sequence.
    :rtype: tskit.TreeSequence
    """
    # Set uniform recombination rate
    recomb_rate_map = msprime.RateMap.uniform(
        sequence_length=sequence_length, rate=recombination_rate
    )

    # Set uniform mutation rate
    mut_rate_map = msprime.RateMap.uniform(
        sequence_length=sequence_length, rate=mutation_rate
    )

    # A simulated tree sequence does not contain any monoallelic sites,
    # but there may be multiallelic sites.
    ts = msprime.sim_mutations(
        msprime.sim_ancestry(
            samples=sample_set,
            demography=demography,
            model="hudson", # TODO: Standard coalescent with recombination.
            recombination_rate=recomb_rate_map,
            discrete_genome=True,
        ),
        rate=mut_rate_map,
        discrete_genome=True,
    )

    # Remove multi-allelic sites
    multiallelic_sites = [v.site.id for v in ts.variants() if v.num_alleles > 2]
    ts = ts.delete_sites(site_ids=multiallelic_sites)   # Topology unaffected

    # Remove populations
    tables = ts.dump_tables()
    tables.populations.clear()
    tables.nodes.population = np.full_like(tables.nodes.population, tskit.NULL)
    ts = tables.tree_sequence()

    return ts


def generate_indices(size_ref, size_query, ploidy_level):
    inds_ref = np.arange(size_ref, dtype=int)
    samples_ref = np.arange(ploidy_level * size_ref, dtype=int)

    inds_query = np.arange(size_ref, size_ref + size_query, dtype=int)
    samples_query = np.arange(
        ploidy_level * size_ref, ploidy_level * (size_ref + size_query), dtype=int
    )

    return [inds_ref, samples_ref, inds_query, samples_query]


def get_ts_toy():
    """
    TODO

    :param None:
    :return:
    :rtype: list
    """
    size_ref = 90
    size_query = 10
    eff_pop_size = 1e4
    mutation_rate = 1e-7
    recombination_rate = 1e-7
    sequence_length = 1e4
    ploidy_level = 1

    demographic_model = msprime.Demography()
    demographic_model.add_population(
        name="A", description="toy", initial_size=eff_pop_size
    )

    sample_set = [
        msprime.SampleSet(num_samples=size_ref, ploidy=ploidy_level, time=0),
        msprime.SampleSet(num_samples=size_query, ploidy=ploidy_level, time=0),
    ]

    ts = simulate_ts(
        sample_set=sample_set,
        demography=demographic_model,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        sequence_length=sequence_length,
    )

    return [ts] + generate_indices(size_ref, size_query, ploidy_level)


def get_ts_single_panmictic(time_query):
    """
    TODO

    :param float time_query:
    :return:
    :rtype: list
    """
    size_ref = 1e4
    size_query = 1e3
    eff_pop_size = 1e4
    mutation_rate = 1e-8
    recombination_rate = 1e-8
    sequence_length = 1e6
    ploidy_level = 1

    demographic_model = msprime.Demography()
    demographic_model.add_population(
        name="A", description="single panmictic", initial_size=eff_pop_size
    )

    sample_set = [
        msprime.SampleSet(num_samples=size_ref, ploidy=ploidy_level, time=0),
        msprime.SampleSet(num_samples=size_query, ploidy=ploidy_level, time=time_query),
    ]

    ts = simulate_ts(
        sample_set=sample_set,
        demography=demographic_model,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        sequence_length=sequence_length,
    )

    return [ts] + generate_indices(size_ref, size_query, ploidy_level)


def get_ts_ten_pop(pop_ref, pop_query):
    """
    TODO

    Contemporary populations: YRI, CHB, and CEU.

    :param str pop_ref:
    :param str pop_query:
    :return:
    :rtype: list
    """
    yaml_file = "./assets/demes/jacobs_2019.yaml"
    ooa_graph = demes.load(yaml_file)
    demographic_model = msprime.Demography.from_demes(ooa_graph)

    size_ref = 1e4
    size_query = 1e3
    mutation_rate = 1e-8
    recombination_rate = 1e-8
    sequence_length = 1e6
    ploidy_level = 1

    # See https://tskit.dev/msprime/docs/stable/api.html?highlight=sampleset#msprime.SampleSet
    # population "May be either a string name or integer ID"
    sample_set = [
        msprime.SampleSet(
            num_samples=size_ref, population=pop_ref, ploidy=ploidy_level, time=0
        ),
        msprime.SampleSet(
            num_samples=size_query, population=pop_query, ploidy=ploidy_level, time=0
        ),
    ]

    ts = simulate_ts(
        sample_set=sample_set,
        demography=demographic_model,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        sequence_length=sequence_length,
    )

    return [ts] + generate_indices(size_ref, size_query, ploidy_level)
