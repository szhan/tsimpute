import pytest
import msprime
import numpy as np
import sys
sys.path.append('../src/')
import util


# Test for functions for adding a new individual to an existing tree sequence
def test_initialise_sample_path():
    s = util.SamplePath(
        individual="test",
        samples=np.repeat(2, 5),
        site_positions=np.arange(5)
    )
    assert s.individual == "test"
    assert np.array_equal(s.samples, np.repeat(2, 5))
    assert np.array_equal(s.site_positions, np.arange(5))
    assert len(s) == 5
    assert s.is_valid()


def test_initialise_sample_path_invalid():
    s = util.SamplePath(
        individual="test",
        samples=np.repeat(2, 4),
        site_positions=np.arange(5) # Offending
    )
    assert s.individual == "test"
    assert np.array_equal(s.samples, np.repeat(2, 4))
    assert np.array_equal(s.site_positions, np.arange(5))
    assert len(s) == 4
    assert not s.is_valid()


# TODO: More tests
# Individual name not specified.
# Samples not specified.
# Site positions not specified.


def test_get_switch_mask_no_switch():
    s = util.SamplePath(
        individual="test: no switch",
        samples=np.repeat(2, 5),
        site_positions=np.arange(5)
    )
    assert s.is_valid()
    actual = util.get_switch_mask(s)
    expected = np.repeat(False, 5)
    assert np.array_equal(actual, expected)


def test_get_switch_mask_one_switch():
    s = util.SamplePath(
        individual="test: one switch, middle",
        samples=np.array([2, 2, 3, 3, 3]),
        site_positions=np.arange(5)
    )
    assert s.is_valid()
    actual = util.get_switch_mask(s)
    expected = np.repeat(False, 5)
    expected[2] = True
    assert np.array_equal(actual, expected)


def test_get_switch_mask_one_switches_end():
    s = util.SamplePath(
        individual="test: one switch, end",
        samples=np.array([2, 2, 2, 2, 3]),
        site_positions=np.arange(5)
    )
    assert s.is_valid()
    actual = util.get_switch_mask(s)
    expected = np.repeat(False, 5)
    expected[4] = True
    assert np.array_equal(actual, expected)


def test_get_switch_mask_multiple_switches():
    s = util.SamplePath(
        individual="test: multiple switches",
        samples=np.array([2, 2, 3, 4, 4]),
        site_positions=np.arange(5)
    )
    assert s.is_valid()
    actual = util.get_switch_mask(s)
    expected = np.repeat(False, 5)
    expected[2] = expected[3] = True
    assert np.array_equal(actual, expected)


# TODO: More tests
# Empty sample path.


def test_add_individual_to_tree_sequence():
    # TODO: Hardcode an example ts.
    # e.g., sites_position = array([ 5.,  9., 35., 68., 77., 95.])
    ts = msprime.sim_mutations(
        msprime.sim_ancestry(
            5,
            sequence_length=100,
            random_seed=1234
        ),
        rate=1e-2,
        random_seed=1234
    )
    assert ts.num_individuals == 5
    assert ts.num_samples == 10
    assert ts.num_nodes == 19
    assert ts.num_sites == 6
    assert ts.sequence_length == 100

    # Examples of three different types of paths.
    individual_name = "Triploid test"
    path_1 = util.SamplePath(
        individual=individual_name,
        nodes=np.array([2, 2, 2, 2, 2, 2]), # No switch
        site_positions=ts.sites_position
    )
    path_2 = util.SamplePath(
        individual=individual_name,
        nodes=np.array([2, 2, 0, 0, 0, 0]), # One switch
        site_positions=ts.sites_position
    )
    path_3 = util.SamplePath(
        individual=individual_name,
        nodes=np.array([2, 2, 0, 0, 1, 1]), # Two switches
        site_positions=ts.sites_position
    )

    ind_id, new_ts = util.add_individual_to_tree_sequence(
        ts,
        paths=[path_1, path_2, path_3]
    )

    assert ind_id == 5
    assert new_ts.num_individuals == ts.num_individuals + 1
    assert new_ts.num_samples == ts.num_samples + 3
    assert new_ts.num_nodes == ts.num_nodes + 3
    assert np.sum(new_ts.edges_parent == 0) == 2
    assert np.sum(new_ts.edges_parent == 1) == 1
    assert np.sum(new_ts.edges_parent == 2) == 3
    assert np.sum(new_ts.edges_child == 19) == 1
    assert np.sum(new_ts.edges_child == 20) == 2
    assert np.sum(new_ts.edges_child == 21) == 3
