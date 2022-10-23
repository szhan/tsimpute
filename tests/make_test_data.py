"""
Make toy tree sequences and samples for testing.
"""
import tskit
import tsinfer


def make_simple_ts():
    """
    Create a simple tree sequence with the following properties:
    a) 1 tree (binary);
    b) 5 sample nodes (4 non-sample internal nodes);
    c) 5 variant sites; and
    d) 5 mutations.

    :param: None
    :return: A simple tree sequence built from a table collection.
    :rtype: tskit.TreeSequence
    """
    tb = tskit.TableCollection(sequence_length=10)
    # Add individuals.
    for _ in range(3):
        tb.individuals.add_row()
    # Add sample nodes.
    tb.nodes.add_row(flags=1, time=0, individual=0)
    tb.nodes.add_row(flags=1, time=0, individual=1)
    tb.nodes.add_row(flags=1, time=0, individual=0)
    tb.nodes.add_row(flags=1, time=0, individual=1)
    tb.nodes.add_row(flags=1, time=0, individual=2)
    tb.nodes.add_row(flags=1, time=0, individual=2)
    # Add non-sample nodes.
    tb.nodes.add_row(flags=0, time=10)
    tb.nodes.add_row(flags=0, time=10)
    tb.nodes.add_row(flags=0, time=10)
    tb.nodes.add_row(flags=0, time=20)
    tb.nodes.add_row(flags=0, time=30)
    # Add edges.
    tb.edges.add_row(left=0, right=10, parent=6, child=0)
    tb.edges.add_row(left=0, right=10, parent=6, child=1)
    tb.edges.add_row(left=0, right=10, parent=7, child=2)
    tb.edges.add_row(left=0, right=10, parent=7, child=3)
    tb.edges.add_row(left=0, right=10, parent=8, child=4)
    tb.edges.add_row(left=0, right=10, parent=8, child=5)
    tb.edges.add_row(left=0, right=10, parent=9, child=7)
    tb.edges.add_row(left=0, right=10, parent=9, child=8)
    tb.edges.add_row(left=0, right=10, parent=10, child=6)
    tb.edges.add_row(left=0, right=10, parent=10, child=9)
    # Add sites.
    tb.sites.add_row(position=1, ancestral_state='A')
    tb.sites.add_row(position=3, ancestral_state='C')
    tb.sites.add_row(position=5, ancestral_state='G')
    tb.sites.add_row(position=7, ancestral_state='T')
    tb.sites.add_row(position=9, ancestral_state='A')
    # Add mutations.
    tb.mutations.add_row(site=0, node=6, derived_state='C')
    tb.mutations.add_row(site=1, node=7, derived_state='G')
    tb.mutations.add_row(site=2, node=8, derived_state='T')
    tb.mutations.add_row(site=3, node=9, derived_state='A')
    tb.mutations.add_row(site=4, node=4, derived_state='C')
    # Create tree sequence.
    ts = tb.tree_sequence()
    return(ts)


def make_simple_sd():
    """
    Create samples with two diploid genomes.

    This is intended to be used to test `util.make_compatible_sample_data()`
    with `make_simple_ts()`.

    :param: None
    :return: Samples.
    :rtype: tsinfer.SampleData
    """
    with tsinfer.SampleData(sequence_length=10) as sd:
        for _ in range(2):
            sd.add_individual(ploidy=2)
        # Position 1. Ref. marker and target marker aligned.
        sd.add_site(position=1, genotypes=[0, 1, 0, 1], alleles=['A', 'C'])
        # Position 3. Ref. marker and target marker unaligned.
        sd.add_site(position=3, genotypes=[0, 1, 0, 1], alleles=['G', 'C'])
        # Position 5. No variant site here, but there is in ref.
        # Position 7. Derive allele is not in ref.
        sd.add_site(position=7, genotypes=[1, 0, 1, 0], alleles=['T', 'G'])
        # Position 8. Unused marker in ts and anc ts, which is only in target.
        sd.add_site(position=8, genotypes=[1, 0, 1, 0], alleles=['A', 'T'])
        # Position 9. Unused marker in anc ts, which is only in target.
        sd.add_site(position=9, genotypes=[1, 0, 1, 0], alleles=['A', 'T'])
    return(sd)