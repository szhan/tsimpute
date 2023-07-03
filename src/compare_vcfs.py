from tqdm import tqdm

from numba import njit
import numpy as np
import xarray as xr

import sgkit as sg

import sys
sys.path.append('../')
import parallelise


_ACGT_ALLELES_ = np.array(['A', 'C', 'G', 'T'])


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


def remap_genotypes(ds1, ds2, acgt_alleles=False, num_workers=1):
    """
    Remap genotypes of `ds2` to `ds1` for each site, such that:
    1. There are only genotypes at sites shared between `ds1` and `ds2`.
    2. The allele lists always have length 4 (full or padded).

    Assumptions:
    1. Only ACGT alleles.
    2. No mixed ploidy.

    Additionally, if `actg_alleles` is set to True,
    all allele lists in `ds2` are set to ACGT.

    :param xarray.Dataset ds1: sgkit-style dataset.
    :param xarray.Dataset ds2: sgkit-style dataset.
    :param bool acgt_alleles: All allele lists are set to ACGT (default = False).
    :param int num_workers: Number of workers (default = 1).
    :return: Allele lists of `ds1` and remapped genotypes for `ds2`.
    :rtype: tuple(xarray.DataArray, xarray.DataArray)
    """
    common_site_idx = get_matching_indices(
        ds1.variant_position.values,
        ds2.variant_position.values
    )

    new_ds2_variant_allele = xr.DataArray(
        np.full([len(common_site_idx), 4], ''),   # ACGT
        dims=["variants", "alleles"]
    )

    new_ds2_call_genotype = xr.DataArray(
        np.zeros([len(common_site_idx), ds2.dims["samples"], ds2.dims["ploidy"]]),
        dims=["variants", "samples", "ploidy"]
    )

    def _remap_genotype_per_site(idx):
        assert len(idx) == 2
        ds1_idx, ds2_idx = idx

        # Get the allele lists at matching positions
        ds1_allel = _ACGT_ALLELES_ if acgt_alleles else \
            np.array([a for a in ds1.variant_allele[ds1_idx].values if a != ''])
        ds2_allel = np.array([a for a in ds2.variant_allele[ds2_idx].values if a != ''])

        if not np.all(np.isin(ds1_allel, _ACGT_ALLELES_)) or \
            not np.all(np.isin(ds2_allel, _ACGT_ALLELES_)):
            raise ValueError(f"Alleles {ds1_allel} and {ds2_allel} are not all ACGT.")

        # Modify the allele lists such that both lists contain the same alleles
        ds1_uniq = np.setdiff1d(ds1_allel, ds2_allel)
        ds2_uniq = np.setdiff1d(ds2_allel, ds1_allel)
        ds1_allel = np.append(ds1_allel, ds2_uniq)
        ds2_allel = np.append(ds2_allel, ds1_uniq)

        if not np.array_equal(np.sort(ds1_allel), np.sort(ds2_allel)):
            raise ValueError("Allele lists are not the same.")

        # Get index map from the allele list of ds2 to the allele list of ds1
        ds1_sort_idx = np.argsort(ds1_allel)
        ds2_sort_idx = np.argsort(ds2_allel)
        idx_arr = np.argsort(ds2_sort_idx)[ds1_sort_idx]

        # Pad allele list so that it is length 4
        if len(ds1_allel) < 4:
            ds1_allel = np.append(ds1_allel, np.full(4 - len(ds1_allel), ''))

        # Remap genotypes 2 to genotypes 1
        ds2_gt = ds2.call_genotype[ds2_idx].values
        new_ds2_gt = idx_arr[ds2_gt].tolist()

        return (ds1_allel, new_ds2_gt)

    # Parallelise this using a threaded map
    if num_workers > 1:
        results = parallelise.threaded_map(
            _remap_genotype_per_site,
            common_site_idx,
            num_workers=num_workers
        )
        i = 0
        for result in tqdm(results, total=len(common_site_idx)):
            new_ds2_variant_allele[i] = result[0]
            new_ds2_call_genotype[i] = result[1]
            i += 1
    else:
        i = 0
        for idx in tqdm(common_site_idx):
            result = _remap_genotype_per_site(idx)
            new_ds2_variant_allele[i] = result[0]
            new_ds2_call_genotype[i] = result[1]
            i += 1

    return (new_ds2_variant_allele, new_ds2_call_genotype)


def make_compatible(ds1, ds2, num_workers=1):
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

    :param xarray.Dataset ds1: sgkit-style dataset.
    :param xarray.Dataset ds2: sgkit-style dataset.
    :param int num_workers: Number of workers (default = 1).
    :return: New `ds2` compatible with `ds1`.
    :rtype: xarray.Dataset
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

    new_ds2_allel, new_ds2_gt = remap_genotypes(
        ds1,
        ds2,
        acgt_alleles=False,
        num_workers=num_workers
    )
    assert new_ds2_allel.shape == (len(common_site_idx), 4)
    assert new_ds2_gt.shape == (len(common_site_idx), ds2.dims["samples"], ds2.dims["ploidy"])

    # Subset `ds2` to common sites
    ds2_variant_contig = ds2.variant_contig.isel(variants=ds2_idx).values
    ds2_variant_position = ds2.variant_position.isel(variants=ds2_idx).values
    ds2_variant_allele = new_ds2_allel.values   # New allele lists
    ds2_call_genotype = new_ds2_gt.values   # New remappged genotypes
    ds2_call_genotype_mask = ds2.call_genotype_mask.isel(variants=ds2_idx).values

    remap_ds2 = xr.Dataset(
        {
            "contig_id": ("contigs", ds2.contig_id.values),
            #"contig_length": ("contigs", ds2.contig_length.values),
            "variant_contig": ("variants", ds2_variant_contig),
            "variant_position": ("variants", ds2_variant_position),
            "variant_allele": (["variants", "alleles"], ds2_variant_allele),
            "sample_id": ("samples", ds2.sample_id.values),
            "call_genotype": (["variants", "samples", "ploidy"], ds2_call_genotype),
            "call_genotype_mask": (["variants", "samples", "ploidy"], ds2_call_genotype_mask),
        },
        attrs={
            "contigs": ds2.contig_id.values,
            #"contig_lengths": ds2.contig_length.values,
            "vcf_header": ds2.vcf_header,
            "source": "sgkit" + "-" + str(sg.__version__),
        }
    )

    return remap_ds2


def remap_to_acgt(ds1, num_workers=1):
    """
    Remap `ds1` so that all sites have the same ACGT allele list.

    Assumptions:
    1. Only ACGT alleles.
    2. No mixed ploidy.

    :param xarray.Dataset ds1: sgkit-style dataset.
    :param int num_workers: Number of workers to use for parallelisation (default = 1).
    :return: New `ds1`.
    :rtype: xarray.Dataset
    """
    new_ds1_allel, new_ds1_gt = remap_genotypes(
        ds1,
        ds1,
        acgt_alleles=True,
        num_workers=num_workers
    )

    new_ds1 = xr.Dataset(
        {
            "contig_id": ("contigs", ds1.contig_id.values),
            #"contig_length": ("contigs", ds1.contig_length.values),
            "variant_contig": ("variants", ds1.variant_contig.values),
            "variant_position": ("variants", ds1.variant_position.values),
            "variant_allele": (["variants", "alleles"], new_ds1_allel.values),
            "sample_id": ("samples", ds1.sample_id.values),
            "call_genotype": (["variants", "samples", "ploidy"], new_ds1_gt.values),
            "call_genotype_mask": (["variants", "samples", "ploidy"], ds1.call_genotype_mask.values),
        },
        attrs={
            "contigs": ds1.contig_id.values,
            #"contig_lengths": ds1.contig_length.values,
            "vcf_header": ds1.vcf_header,
            "source": "sgkit" + "-" + str(sg.__version__),
        }
    )

    return new_ds1
