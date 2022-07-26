import numpy as np
import tskit


def pick_masked_sites_random(site_ids, prop_masked_sites):
    """
    Draw N sites from `sites_ids` at random, where N is the number of sites to mask
    based on a specified proportion of masked sites `prop_masked_sites`.

    TODO: Specify random seed.

    :param ndarray site_ids:
    :param float prop_masked_sites: value between 0 and 1
    :return ndarray: list of site ids
    """
    assert prop_masked_sites >= 0
    assert prop_masked_sites <= 1

    rng = np.random.default_rng()

    num_masked_sites = int(np.floor(len(site_ids) * prop_masked_sites))

    masked_site_ids = np.sort(
        rng.choice(
            site_ids,
            num_masked_sites,
            replace=False,
        )
    )

    return masked_site_ids


def mask_sites_in_sample_data(sample_data, masked_sites=None):
    """
    Create and return a `SampleData` object from an existing `SampleData` object,
    which contains masked sites as listed in `masked_sites` (site ids).

    :param SampleData sample_data:
    :param ndrray masked_sites: list of site ids (NOT positions)
    :return SampleData:
    """
    new_sample_data = sample_data.copy()

    for v in sample_data.variants():
        if v.site.id in masked_sites:
            new_sample_data.sites_genotypes[v.site.id] = np.full_like(
                v.genotypes, tskit.MISSING_DATA
            )

    new_sample_data.finalise()

    return new_sample_data
