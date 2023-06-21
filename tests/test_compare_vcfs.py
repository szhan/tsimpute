import pytest

import numpy as np
import xarray as xr

import sgkit as sg

import sys
sys.path.append('../src/')
import compare_vcfs


# Assumptions:
# 1. All individuals (i.e., samples) are diploid in both ds1 and ds2.
# 2. There is no missing data.

def make_test_case():
    """
    Create an xarray dataset that contains:
    1) two diploid samples (i.e., individuals); and
    2) two biallelic sites on contig 20 at positions 5 and 10.

    :return: Dataset for sgkit use.
    :rtype: xr.Dataset
    """
    contig_id = ['20']
    variant_contig = np.array([0, 0], dtype=np.int64)
    variant_position = np.array([5, 10], dtype=np.int64)
    variant_allele = np.array([
        ['A', 'C'],
        ['G', 'T'],
    ])
    sample_id = np.array([
        'tsk0',
        'tsk1',
    ])
    call_genotype = np.array(
        [
            [
                [0, 1],  # tsk0
                [1, 0],  # tsk1
            ],
            [
                [1, 0],  # tsk0
                [0, 1],  # tsk1
            ],
        ],
        dtype=np.int8
    )
    call_genotype_mask = np.zeros_like(call_genotype, dtype=bool)   # no mask

    ds = xr.Dataset(
        {
            "contig_id": ("contigs", contig_id),
            "variant_contig": ("variants", variant_contig),
            "variant_position": ("variants", variant_position),
            "variant_allele": (["variants", "alleles"], variant_allele),
            "sample_id": ("samples", sample_id),
            "call_genotype": (["variants", "samples", "ploidy"], call_genotype),
            "call_genotype_mask": (["variants", "samples", "ploidy"], call_genotype_mask),
        },
        attrs={
            "contigs": contig_id,
            "source": "sgkit" + "-" + str(sg.__version__),
        }
    )

    return ds


@pytest.mark.parametrize(
    "arr1, arr2, expected",
    [
        pytest.param([1, 5, 9], [5, 9, 15], [(1, 0), (2, 1)], id="sites shared"),
        pytest.param([1, 9], [5, 15], [], id="no sites shared"),
        pytest.param([1], [], [], id="one empty"),
        pytest.param([], [], [], id="both empty"),
    ]
)
def test_get_matching_indices(arr1, arr2, expected):
    actual = compare_vcfs.get_matching_indices(arr1, arr2)
    assert np.array_equal(actual, expected)


def test_both_biallelic_same_alleles_same_order():
    ds1 = make_test_case()
    ds2 = ds1.copy(deep=True)
    _, actual = compare_vcfs.remap_genotypes(ds1, ds2)
    expected = xr.DataArray(
        [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ],
        dims=["variants", "samples", "ploidy"]
    )
    assert np.array_equal(actual, expected)


def test_both_biallelic_same_alleles_different_order():
    ds1 = make_test_case()
    ds2 = ds1.copy(deep=True)
    for i in np.arange(ds2.variant_contig.size):
        ds2.variant_allele[i] = xr.DataArray(np.flip(ds2.variant_allele[i]))
        ds2.call_genotype[i] = xr.DataArray(np.where(ds2.call_genotype[i] == 0, 1, 0))
    _, actual = compare_vcfs.remap_genotypes(ds1, ds2)
    expected = xr.DataArray(
        [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ],
        dims=["variants", "samples", "ploidy"]
    )
    assert np.array_equal(actual, expected)


def test_both_biallelic_different_alleles():
    ds1 = make_test_case()
    ds2 = ds1.copy(deep=True)
    # At the first site, one allele is shared.
    ds2.variant_allele[0] = xr.DataArray(['C', 'G'])
    ds2.call_genotype[0] = xr.DataArray([[0, 1], [1, 0]])
    # At the second site, no allele is shared.
    ds2.variant_allele[1] = xr.DataArray(['A', 'C'])
    ds2.call_genotype[1] = xr.DataArray([[0, 1], [1, 0]])
    # Subtest 1
    _, actual = compare_vcfs.remap_genotypes(ds1, ds2)
    expected = xr.DataArray(
        [
            [[1, 2], [2, 1]],
            [[2, 3], [3, 2]],
        ],
        dims=["variants", "samples", "ploidy"]
    )
    assert np.array_equal(actual, expected)
    # Subtest 2
    _, actual = compare_vcfs.remap_genotypes(ds2, ds1)
    expected = xr.DataArray(
        [
            [[2, 0], [0, 2]],
            [[3, 2], [2, 3]],
        ],
        dims=["variants", "samples", "ploidy"]
    )
    assert np.array_equal(actual, expected)


def test_biallelic_monoallelic():
    ds1 = make_test_case()
    ds2 = ds1.copy(deep=True)
    # At the first site, one allele is shared.
    # At the second site, no allele is shared.
    for i in np.arange(ds2.variant_contig.size):
        ds2.variant_allele[i] = xr.DataArray(['C', ''])
        ds2.call_genotype[i] = xr.DataArray(np.zeros_like(ds2.call_genotype[i]))
    # Subtest 1
    _, actual = compare_vcfs.remap_genotypes(ds1, ds2)
    expected = xr.DataArray(
        [
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
        ],
        dims=["variants", "samples", "ploidy"]
    )
    assert np.array_equal(actual, expected)
    # Subtest 2
    _, actual = compare_vcfs.remap_genotypes(ds2, ds1)
    expected = xr.DataArray(
        [
            [[1, 0], [0, 1]],
            [[2, 1], [1, 2]],
        ],
        dims=["variants", "samples", "ploidy"]
    )
    assert np.array_equal(actual, expected)


def test_both_monoallelic():
    ds1 = make_test_case()
    ds2 = ds1.copy(deep=True)
    # Overwrite certain data variables in ds1 and ds2.
    # At the first site, one allele is shared.
    ds1.variant_allele[0] = xr.DataArray(['C', ''])
    ds1.call_genotype[0] = xr.DataArray(np.zeros_like(ds1.call_genotype[0]))
    ds2.variant_allele[0] = xr.DataArray(['C', ''])
    ds2.call_genotype[0] = xr.DataArray(np.zeros_like(ds2.call_genotype[0]))
    # At the second site, no allele is shared.
    ds1.variant_allele[1] = xr.DataArray(['C', ''])
    ds1.call_genotype[1] = xr.DataArray(np.zeros_like(ds1.call_genotype[1]))
    ds2.variant_allele[1] = xr.DataArray(['G', ''])
    ds2.call_genotype[1] = xr.DataArray(np.zeros_like(ds2.call_genotype[1]))
    _, actual = compare_vcfs.remap_genotypes(ds1, ds2)
    expected = xr.DataArray(
        [
            [[0, 0], [0, 0]],
            [[1, 1], [1, 1]],
        ],
        dims=["variants", "samples", "ploidy"]
    )
    assert np.array_equal(actual, expected)


def test_acgt_alleles_true():
    ds1 = make_test_case()
    ds2 = ds1.copy(deep=True)
    actual_alleles, actual_genotypes = compare_vcfs.remap_genotypes(ds1, ds2, acgt_alleles=True)
    expected_alleles = np.array([
        compare_vcfs._ACGT_ALLELES_,
        compare_vcfs._ACGT_ALLELES_,
    ])
    expected_genotypes = xr.DataArray(
        [
            [[0, 1], [1, 0]],
            [[3, 2], [2, 3]],
        ],
        dims=["variants", "samples", "ploidy"]
    )
    assert np.array_equal(actual_alleles, expected_alleles)
    assert np.array_equal(actual_genotypes, expected_genotypes)


def test_both_empty():
    """TODO"""
    raise NotImplementedError


def test_one_empty():
    """TODO"""
    raise NotImplementedError


def test_non_acgt():
    """TODO"""
    raise NotImplementedError


def test_make_compatible_genotypes():
    ds1 = make_test_case()
    ds2 = ds1.copy(deep=True)
    for i in np.arange(ds2.variant_contig.size):
        ds2.variant_allele[i] = xr.DataArray(np.flip(ds2.variant_allele[i]))
        ds2.call_genotype[i] = xr.DataArray(np.where(ds2.call_genotype[i] == 0, 1, 0))
    ds1_compat, ds2_compat = compare_vcfs.make_compatible_genotypes(ds1, ds2)
    assert np.array_equal(ds1_compat.variant_allele, ds2_compat.variant_allele)
    assert np.array_equal(ds1_compat.call_genotype, ds2_compat.call_genotype)


def test_make_compatible_genotypes_acgt_alleles_true():
    ds1 = make_test_case()
    ds2 = ds1.copy(deep=True)
    ds1_compat, ds2_compat = compare_vcfs.make_compatible_genotypes(ds1, ds2, acgt_alleles=True)
    expected_alleles = np.array([
        compare_vcfs._ACGT_ALLELES_,
        compare_vcfs._ACGT_ALLELES_,
    ])
    assert np.array_equal(ds1_compat.variant_allele, expected_alleles)
    assert np.array_equal(ds1_compat.variant_allele, ds2_compat.variant_allele)
    assert np.array_equal(ds1_compat.call_genotype, ds2_compat.call_genotype)
