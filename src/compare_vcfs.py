from tqdm import tqdm

import numpy as np
import xarray as xr

from numba import njit

import sgkit as sg


_ACGT_ALLELES = np.array(['A', 'C', 'G', 'T'])


@njit
def get_matching_indices(arr1, arr2):
    """
    Get the indices of `arr1` and `arr2`,
    where the values of `arr1` and `arr2` are equal.

    :param numpy.ndarray arr1: 1D array
    :param numpy.ndarray arr2: 1D array
    :return: Indices of `arr1` and `arr2`
    :rtype: numpy.ndarray
    """
    idx_pairs = []
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            idx_pairs.append((i, j))
            i += 1
            j += 1
        elif arr1[i] < arr2[j]:
            i += 1
        else:
            j += 1
    return np.array(idx_pairs)


def remap_genotypes(ds1, ds2, acgt_alleles=False):
    """
    Remap genotypes of `ds2` to `ds1` for each site, such that:
    1. There are only genotypes shared between `ds1` and `ds2`.
    2. The allele lists always have length 4 (full or padded).

    Assumptions:
    1. Only ACGT alleles.
    2. No mixed ploidy.

    Additionally, if `actg_alleles` is set to True,
    all allele lists in `ds2` are set to ACGT.

    :param xarray.Dataset ds1: sgkit-style dataset.
    :param xarray.Dataset ds2: sgkit-style dataset.
    :param bool acgt_alleles: All allele lists are set to ACGT (default = False).
    :return: Remapped genotypes of `ds2`.
    :rtype: xarray.DataArray
    """
    common_site_idx = get_matching_indices(
        ds1.variant_position.values,
        ds2.variant_position.values
    )

    remapped_ds2_variant_allele = xr.DataArray(
        np.empty([len(common_site_idx), 4], dtype='|S1'),   # ACGT
        dims=["variants", "alleles"]
    )
    remapped_ds2_call_genotype = xr.DataArray(
        np.zeros([len(common_site_idx), ds2.dims["samples"], ds2.dims["ploidy"]]),
        dims=["variants", "samples", "ploidy"]
    )

    i = 0
    for ds1_idx, ds2_idx in tqdm(common_site_idx):
        # Get the allele lists at matching positions
        ds1_alleles = _ACGT_ALLELES if acgt_alleles else \
            np.array([a for a in ds1.variant_allele[ds1_idx].values if a != ''])
        ds2_alleles = np.array([a for a in ds2.variant_allele[ds2_idx].values if a != ''])

        if not np.all(np.isin(ds1_alleles, _ACGT_ALLELES)) or \
            not np.all(np.isin(ds2_alleles, _ACGT_ALLELES)):
            raise ValueError(f"Alleles {ds1_alleles} and {ds2_alleles} are not all ACGT.")

        # Modify the allele lists such that both lists contain the same alleles
        ds1_uniq = np.setdiff1d(ds1_alleles, ds2_alleles)
        ds2_uniq = np.setdiff1d(ds2_alleles, ds1_alleles)
        ds1_alleles = np.append(ds1_alleles, ds2_uniq)
        ds2_alleles = np.append(ds2_alleles, ds1_uniq)

        if not np.array_equal(np.sort(ds1_alleles), np.sort(ds2_alleles)):
            raise ValueError("Allele lists are not the same.")

        # Get index map from the allele list of ds2 to the allele list of ds1
        ds1_sort_idx = np.argsort(ds1_alleles)
        ds2_sort_idx = np.argsort(ds2_alleles)
        index_array = np.argsort(ds2_sort_idx)[ds1_sort_idx]

        # Pad allele list so that it is length 4
        if len(ds1_alleles) < 4:
            ds1_alleles = np.append(ds1_alleles, np.full(4 - len(ds1_alleles), ''))

        # Remap genotypes 2 to genotypes 1
        remapped_ds2_variant_allele[i] = ds1_alleles
        ds2_genotype = ds2.call_genotype[ds2_idx].values
        remapped_ds2_call_genotype[i] = index_array[ds2_genotype].tolist()
        i += 1

    return (remapped_ds2_variant_allele, remapped_ds2_call_genotype)


def make_compatible_genotypes(ds1, ds2, acgt_alleles=False):
    """
    Make `ds2` compatible with `ds1` by remapping genotypes.

    Definition of compatibility:
    1. `ds1` and `ds2` have the same number of samples.
    2. `ds1` and `ds2` have the same ploidy.
    3. `ds1` and `ds2` have the same number of variable sites.
    4. `ds1` and `ds2` have the same allele list at each site.

    Assumptions:
    1. Only ACGT alleles.
    2. No mixed ploidy.

    Additionally, if `actg_alleles` is set to True,
    all allele lists in `ds1` and `ds2` are set to ACGT.

    :param xarray.Dataset ds1: sgkit-style dataset.
    :param xarray.Dataset ds2: sgkit-style dataset.
    :param bool acgt_alleles: All allele lists are set to ACGT (default = False).
    :return: Compatible `ds1` and `ds2`.
    :rtype: tuple(xarray.Dataset, xarray.Dataset)
    """
    # TODO: Refactor, routine run again when calling `remap_genotypes`
    common_site_idx = get_matching_indices(
        ds1.variant_position.values,
        ds2.variant_position.values
    )
    ds1_idx, ds2_idx = np.split(common_site_idx, 2, axis=1)
    ds1_idx = np.array(ds1_idx.flatten())
    ds2_idx = np.array(ds2_idx.flatten())
    assert len(ds1_idx) == len(ds2_idx) == len(common_site_idx)

    remap_ds2_alleles, remap_ds2_genotypes = remap_genotypes(ds1, ds2, acgt_alleles=acgt_alleles)
    assert remap_ds2_alleles.shape == (len(common_site_idx), 4)
    assert remap_ds2_genotypes.shape == (len(common_site_idx), ds2.dims["samples"], ds2.dims["ploidy"])

    # Subset `ds1` to common sites
    ds1_contig_id = ds1.contig_id.values
    ds1_variant_contig = ds1.variant_contig.isel(variants=ds1_idx).values
    ds1_variant_position = ds1.variant_position.isel(variants=ds1_idx).values
    if acgt_alleles:
        remap_ds1_alleles, remap_ds1_genotypes = remap_genotypes(ds2, ds1, acgt_alleles=acgt_alleles)
        assert remap_ds1_alleles.shape == (len(common_site_idx), 4)
        assert remap_ds1_genotypes.shape == (len(common_site_idx), ds1.dims["samples"], ds1.dims["ploidy"])
        ds1_variant_allele = remap_ds1_alleles.values
        ds1_call_genotype = remap_ds1_genotypes.values
    else:
        ds1_variant_allele = remap_ds2_alleles.values
        ds1_call_genotype = ds1.call_genotype.isel(variants=ds1_idx).values
    ds1_sample_id = ds1.sample_id.values
    ds1_call_genotype_mask = ds1.call_genotype_mask.isel(variants=ds1_idx).values

    remap_ds1 = xr.Dataset(
        {
            "contig_id": ("contigs", ds1_contig_id),
            "variant_contig": ("variants", ds1_variant_contig),
            "variant_position": ("variants", ds1_variant_position),
            "variant_allele": (["variants", "alleles"], ds1_variant_allele),
            "sample_id": ("samples", ds1_sample_id),
            "call_genotype": (["variants", "samples", "ploidy"], ds1_call_genotype),
            "call_genotype_mask": (["variants", "samples", "ploidy"], ds1_call_genotype_mask),
        },
        attrs={
            "contigs": ds1_contig_id,
            "source": "sgkit" + "-" + str(sg.__version__),
        }
    )

    # Subset `ds2` to common sites
    ds2_contig_id = ds2.contig_id.values
    ds2_variant_contig = ds2.variant_contig.isel(variants=ds2_idx).values
    ds2_variant_position = ds2.variant_position.isel(variants=ds2_idx).values
    ds2_variant_allele = remap_ds2_alleles.values
    ds2_sample_id = ds2.sample_id.values
    ds2_call_genotype = remap_ds2_genotypes.values
    ds2_call_genotype_mask = ds2.call_genotype_mask.isel(variants=ds2_idx).values

    remap_ds2 = xr.Dataset(
        {
            "contig_id": ("contigs", ds2_contig_id),
            "variant_contig": ("variants", ds2_variant_contig),
            "variant_position": ("variants", ds2_variant_position),
            "variant_allele": (["variants", "alleles"], ds2_variant_allele),
            "sample_id": ("samples", ds2_sample_id),
            "call_genotype": (["variants", "samples", "ploidy"], ds2_call_genotype),
            "call_genotype_mask": (["variants", "samples", "ploidy"], ds2_call_genotype_mask),
        },
        attrs={
            "contigs": ds2_contig_id,
            "source": "sgkit" + "-" + str(sg.__version__),
        }
    )

    return (remap_ds1, remap_ds2)
