import numpy as np
import tskit


def pick_masked_sites_random(sites, prop_masked_sites, seed=None):
    """
    Draw N sites from `sites` at random, where N is the number of sites to mask
    based on the specified proportion of masked sites.

    :param np.array sites: List of sites IDs.
    :param float prop_masked_sites: Proportion of masked sites [0, 1].
    :param int seed: Seed for np.random.rng (default = None).
    :return: List of site IDs.
    :rtype: np.array
    """
    assert prop_masked_sites >= 0 and prop_masked_sites <= 1, \
        f"{prop_masked_sites} is not between 0 and 1."

    rng = np.random.default_rng(seed=seed)

    num_masked_sites = int(np.floor(len(sites) * prop_masked_sites))

    masked_sites = np.sort(
        rng.choice(
            sites,
            num_masked_sites,
            replace=False,
        )
    )

    return masked_sites


def mask_sites_in_sample_data(sd, sites, site_type):
    """
    Create and return a `SampleData` object from an existing `SampleData` object,
    which contains masked sites (all genotypes marked as missing) specified by
    site IDs or positions.

    :param tsinfer.SampleData sd: A SampleData object to mask.
    :param np.array sites: A list of site IDs or positions.
    :param str site_type: IDs ("id") or positions ("position").
    :return: A copy of the SampleData object with masked sites.
    :rtype: tsinfer.SampleData
    """
    assert site_type in ["id", "position"], f"Site type {site_type} is invalid."

    new_sd = sd.copy()

    for v in sd.variants():
        site_ref = v.site.id if site_type == "id" else v.site.position
        if site_ref in sites:
            new_sd.sites_genotypes[v.site.id] = np.full_like(
                v.genotypes, tskit.MISSING_DATA
            )

    new_sd.finalise()

    return new_sd


def parse_site_position_file(in_file):
    """
    Read list of site positions from a plain tab-delimited text file.

    TODO: Consider sequence name, which is ignored now.

    :param in_file: A list of site positions.
    :return: A set of site positions.
    :rtype: set
    """
    site_pos = set()
    
    with open(in_file, "rt") as f:
        for line in f:
            chr, pos = line.rstrip().split("\t")
            site_pos.append(pos)
    
    return site_pos