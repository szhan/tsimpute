import pytest

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
