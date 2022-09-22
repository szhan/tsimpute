import numpy as np
import tskit


def pick_masked_sites_random(site_ids, prop_masked_sites, seed=None):
    """
    Draw N sites from `sites_ids` at random, where N is the number of sites to mask
    based on a specified proportion of masked sites `prop_masked_sites`.

    :param ndarray site_ids:
    :param float prop_masked_sites: value between 0 and 1
    :param int seed: integer to pass to np.random.rng (default = None)
    :return ndarray: list of site ids
    """
    assert prop_masked_sites >= 0
    assert prop_masked_sites <= 1

    rng = np.random.default_rng(seed=seed)

    num_masked_sites = int(np.floor(len(site_ids) * prop_masked_sites))

    masked_site_ids = np.sort(
        rng.choice(
            site_ids,
            num_masked_sites,
            replace=False,
        )
    )

    return masked_site_ids


def mask_sites_in_sample_data(sd, sites):
    """
    Create and return a `SampleData` object from an existing `SampleData` object,
    which contains masked sites (all genotypes marked as missing) specified by
    site IDs.

    :param tsinfer.SampleData sd: A SampleData object to mask.
    :param np.array sites: List of site IDs (NOT positions).
    :return: A copy of the SampleData object with masked sites.
    :rtype: tsinfer.SampleData
    """
    new_sd = sd.copy()

    for v in sd.variants():
        if v.site.id in sites:
            new_sd.sites_genotypes[v.site.id] = np.full_like(
                v.genotypes, tskit.MISSING_DATA
            )

    new_sd.finalise()

    return new_sd
