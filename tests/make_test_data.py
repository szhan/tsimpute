"""
Make toy tree sequences and samples for testing.
"""
import tskit


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
    # Add sample nodes.
    for _ in range(5):
        tb.nodes.add_row(flags=1, time=0)
    # Add non-sample (internal) nodes.
    tb.nodes.add_row(flags=0, time=10)
    tb.nodes.add_row(flags=0, time=10)
    tb.nodes.add_row(flags=0, time=20)
    tb.nodes.add_row(flags=0, time=30)
    # Add edges.
    tb.edges.add_row(left=0, right=10, parent=5, child=0)
    tb.edges.add_row(left=0, right=10, parent=5, child=1)
    tb.edges.add_row(left=0, right=10, parent=6, child=2)
    tb.edges.add_row(left=0, right=10, parent=6, child=3)
    tb.edges.add_row(left=0, right=10, parent=7, child=4)
    tb.edges.add_row(left=0, right=10, parent=7, child=6)
    tb.edges.add_row(left=0, right=10, parent=8, child=5)
    tb.edges.add_row(left=0, right=10, parent=8, child=7)
    # Add sites.
    tb.sites.add_row(position=1, ancestral_state='A')
    tb.sites.add_row(position=3, ancestral_state='C')
    tb.sites.add_row(position=5, ancestral_state='G')
    tb.sites.add_row(position=7, ancestral_state='T')
    tb.sites.add_row(position=9, ancestral_state='A')
    # Add mutations.
    tb.mutations.add_row(site=0, node=5, derived_state='C')
    tb.mutations.add_row(site=1, node=7, derived_state='G')
    tb.mutations.add_row(site=2, node=6, derived_state='T')
    tb.mutations.add_row(site=3, node=6, derived_state='A')
    tb.mutations.add_row(site=4, node=4, derived_state='C')
    # Create tree sequence.
    ts = tb.tree_sequence()
    return(ts)
