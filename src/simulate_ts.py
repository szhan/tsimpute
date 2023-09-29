import demes
import msprime
import tskit
import numpy as np


# TODO: Use a HapMap genetic map.


def simulate_ts(
    sample_set, demography, recombination_rate, mutation_rate, sequence_length
):
    """
    Simulate a tree sequence using `msprime` under a specified demographic model
    with a genome-wide mutation rate and recombination rate.

    The standard coalescent with recombination is used (i.e. Hudson's model).

    Processing steps before returning the `TreeSequence` object:
    1. All multi-allelic sites are deleted.
    2. All populations are cleared.

    :param list sample_set: A list of msprime.SampleSet objects.
    :param msprime.Demography demography: If None, it defaults to a single population with constant Ne of 1.
    :param float recombination_rate: Recombination rate.
    :param float mutation_rate: Mutation rate.
    :param float sequence_length: Sequence length input to get msprime.RateMap objects.
    :return: Simulated tree sequence.
    :rtype: tskit.TreeSequence
    """
    # Set rate maps.
    recomb_rate_map = msprime.RateMap.uniform(
        sequence_length=sequence_length, rate=recombination_rate
    )
    mut_rate_map = msprime.RateMap.uniform(
        sequence_length=sequence_length, rate=mutation_rate
    )
    # Simulate a tree sequence.
    # Note a simulated ts contains no mono-allelic sites, but there may be multi-allelic sites.
    ts = msprime.sim_mutations(
        msprime.sim_ancestry(
            samples=sample_set,
            demography=demography,
            model="hudson",
            recombination_rate=recomb_rate_map,
            discrete_genome=True,
        ),
        rate=mut_rate_map,
        discrete_genome=True,
    )
    # Remove multi-allelic sites
    multiallelic_sites = [v.site.id for v in ts.variants() if v.num_alleles > 2]
    ts = ts.delete_sites(site_ids=multiallelic_sites)  # Topology unaffected
    # Remove populations
    tables = ts.dump_tables()
    tables.populations.clear()
    tables.nodes.population = np.full_like(tables.nodes.population, tskit.NULL)
    ts = tables.tree_sequence()
    # Check
    assert np.all(np.array([v.num_alleles for v in ts.variants()]) == 2)
    assert ts.num_populations == 0
    return ts


def get_ts_toy():
    """
    Simulate a tree sequence assuming a simple demographic model.

    :param None None:
    :return: Tree sequence and associated indices.
    :rtype: tuple
    """
    # Set parameters.
    num_ref_haps = 90
    num_query_haps = 10
    sequence_length = 1e4  # 10 kbp
    ne = 1e4  # 10k
    mutation_rate = 1e-7
    recombination_rate = 1e-7
    ploidy = 1  # Haploid individuals
    # Specify demographic model.
    demographic_model = msprime.Demography()
    demographic_model.add_population(name="A", description="toy", initial_size=ne)
    # Define sample set.
    sample_set = [
        msprime.SampleSet(num_samples=num_ref_haps, ploidy=ploidy, time=0),
        msprime.SampleSet(num_samples=num_query_haps, ploidy=ploidy, time=0),
    ]
    # Simulate trees.
    ts = simulate_ts(
        sample_set=sample_set,
        demography=demographic_model,
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        sequence_length=sequence_length,
    )
    return ts


def get_ts_single_panmictic(
    num_ref_haps=1e4,  # 10k
    num_query_haps=1e3,  # 1k
    sequence_length=1e6,  # 1 Mbp
    ne=1e4,  # 10k
    mutation_rate=1e-8,
    recombination_rate=1e-8,
    time_ref=0,
    time_query=0,
):
    """
    Simulate a tree sequence assuming a single panmictic population with constant Ne.

    :param int num_ref_haps: Number of reference haplotypes.
    :param int num_query_haps: Number of query haplotypes.
    :param float sequence_length: Sequence length.
    :param float ne: Effective population size.
    :param float mutation_rate: Mutation rate.
    :param float recombination_rate: Recomination rate.
    :param float time_ref: Time of sampling the reference haplotypes.
    :param float time_query: Time of sampling the query haplotypes.
    :return: Tree sequence and associated indices.
    :rtype: tuple
    """
    # Set parameters.
    ploidy = 1  # Haploid individuals
    # Specify demographic model.
    demographic_model = msprime.Demography()
    demographic_model.add_population(
        name="A", description="single panmictic", initial_size=ne
    )
    # Define sample set.
    sample_set = [
        msprime.SampleSet(num_samples=num_ref_haps, ploidy=ploidy, time=time_ref),
        msprime.SampleSet(num_samples=num_query_haps, ploidy=ploidy, time=time_query),
    ]
    # Simulate trees.
    ts = simulate_ts(
        sample_set=sample_set,
        demography=demographic_model,
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        sequence_length=sequence_length,
    )
    return ts


def get_ts_ten_pop(
    num_ref_inds,
    num_query_inds,
    sequence_length,
    pop_ref,
    pop_query,
    recombination_rate=1e-8,
    mutation_rate=1e-8,
    time_ref=0,
    time_query=0,
    yaml_dir="../assets/demes/"
):
    """
    Simulate a tree sequence under the Jacobs et al. (2019) demographic model.

    Choose the reference and query (target) populations from one of YRI, CHB, and CEU.

    :param str pop_ref: Name of the reference population.
    :param str pop_query: Name of the query population.
    :param int num_ref_inds: Number of reference individuals.
    :param int num_query_inds: Number of query individuals.
    :param int sequence_length: Sequence length.
    :param float mutation_rate: Mutation rate.
    :param float recombination_rate: Recomination rate.
    :param float time_ref: Time of sampling the reference haplotypes.
    :param float time_query: Time of sampling the query haplotypes.
    :param str yaml_dir: Directory containing demes YAML files.
    :return: Tree sequence and associated indices.
    :rtype: tuple
    """
    # Set parameters.
    ploidy = 2  # Diploid individuals
    # Get demographic model.
    yaml_file = yaml_dir + "/" + "jacobs_2019.yaml"
    ooa_graph = demes.load(yaml_file)
    demographic_model = msprime.Demography.from_demes(ooa_graph)
    # Define sample set.
    # See https://tskit.dev/msprime/docs/stable/api.html?highlight=sampleset#msprime.SampleSet
    # Population "May be either a string name or integer ID"
    sample_set = [
        msprime.SampleSet(
            num_samples=num_ref_inds, population=pop_ref, ploidy=ploidy, time=time_ref
        ),
        msprime.SampleSet(
            num_samples=num_query_inds,
            population=pop_query,
            ploidy=ploidy,
            time=time_query,
        ),
    ]
    # Simulate trees.
    ts = simulate_ts(
        sample_set=sample_set,
        demography=demographic_model,
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        sequence_length=sequence_length,
    )
    return ts
