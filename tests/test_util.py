from dataclasses import FrozenInstanceError
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
        nodes=np.repeat(2, 5),
        site_positions=np.arange(5)
    )
    assert s.individual == "test"
    assert np.array_equal(s.nodes, np.repeat(2, 5))
    assert np.array_equal(s.site_positions, np.arange(5))
    assert len(s) == 5
    assert s.is_valid


def test_initialise_sample_path_invalid():
    s = util.SamplePath(
        individual="test",
        nodes=np.repeat(2, 4),
        site_positions=np.arange(5) # Offending
    )
    assert s.individual == "test"
    assert np.array_equal(s.nodes, np.repeat(2, 4))
    assert np.array_equal(s.site_positions, np.arange(5))
    assert len(s) == 4
    assert not s.is_valid


def test_frozen_instance_sample_path():
    s = util.SamplePath(
        individual="test",
        nodes=np.repeat(2, 5),
        site_positions=np.arange(5)
    )
    with pytest.raises(FrozenInstanceError):
        s.individual = "test2"
    with pytest.raises(FrozenInstanceError):
        s.nodes = np.repeat(3, 5)
    with pytest.raises(FrozenInstanceError):
        s.site_positions = np.arange(5)

# TODO: More tests
# Individual name not specified.
# Samples not specified.
# Site positions not specified.


def test_get_switch_mask_no_switch():
    s = util.SamplePath(
        individual="test: no switch",
        nodes=np.repeat(2, 5),
        site_positions=np.arange(5)
    )
    assert s.is_valid
    actual = util.get_switch_mask(s)
    expected = np.repeat(False, 5)
    assert np.array_equal(actual, expected)


def test_get_switch_mask_one_switch():
    s = util.SamplePath(
        individual="test: one switch, middle",
        nodes=np.array([2, 2, 3, 3, 3]),
        site_positions=np.arange(5)
    )
    assert s.is_valid
    actual = util.get_switch_mask(s)
    expected = np.repeat(False, 5)
    expected[2] = True
    assert np.array_equal(actual, expected)


def test_get_switch_mask_one_switches_end():
    s = util.SamplePath(
        individual="test: one switch, end",
        nodes=np.array([2, 2, 2, 2, 3]),
        site_positions=np.arange(5)
    )
    assert s.is_valid
    actual = util.get_switch_mask(s)
    expected = np.repeat(False, 5)
    expected[4] = True
    assert np.array_equal(actual, expected)


def test_get_switch_mask_multiple_switches():
    s = util.SamplePath(
        individual="test: multiple switches",
        nodes=np.array([2, 2, 3, 4, 4]),
        site_positions=np.arange(5)
    )
    assert s.is_valid
    actual = util.get_switch_mask(s)
    expected = np.repeat(False, 5)
    expected[2] = expected[3] = True
    assert np.array_equal(actual, expected)


# TODO: More tests
# Empty sample path.


def test_add_individuals_to_tree_sequence():
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

    # Examples of four different paths.
    individual_names = ["test_1", "test_2"]
    paths = np.array(
        [
            [2, 2, 2, 2, 2, 2], # No switch
            [2, 2, 0, 0, 0, 0], # One switch
            [2, 2, 0, 0, 1, 1], # Two switches
            [2, 2, 0, 0, 1, 2], # Three switches
        ],
        dtype=np.int32
    )

    new_ts = util.add_individuals_to_tree_sequence(
        ts=ts,
        paths=paths,
        individual_names=individual_names,
    )

    assert new_ts.num_individuals == ts.num_individuals + 2
    assert new_ts.num_samples == ts.num_samples + 4
    assert new_ts.num_nodes == ts.num_nodes + 4

    assert np.sum(new_ts.edges_parent == 0) == 3
    assert np.sum(new_ts.edges_parent == 1) == 2
    assert np.sum(new_ts.edges_parent == 2) == 5
    assert np.sum(new_ts.edges_child == 19) == 1
    assert np.sum(new_ts.edges_child == 20) == 2
    assert np.sum(new_ts.edges_child == 21) == 3
    assert np.sum(new_ts.edges_child == 22) == 4
