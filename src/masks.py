import numpy as np
import pandas as pd
import tskit


def pick_mask_sites_random(sites, prop_mask_sites, seed=None):
    """
    Draw N sites from `sites` at random, where N is the number of sites to mask
    based on the specified proportion of masked sites.

    :param np.ndarray sites: A list of sites IDs.
    :param float prop_mask_sites: Proportion of mask sites [0, 1].
    :param int seed: Seed for np.random.rng (default = None).
    :return: A list of site IDs.
    :rtype: np.ndarray
    """
    assert 0 <= prop_mask_sites <= 1, f"{prop_mask_sites} is not a proportion."

    rng = np.random.default_rng(seed=seed)

    num_mask_sites = int(np.floor(len(sites) * prop_mask_sites))

    mask_sites = np.sort(
        rng.choice(
            sites,
            num_mask_sites,
            replace=False,
        )
    )

    return mask_sites


def mask_sites_in_sample_data(sample_data, sites, site_type, path=None):
    """
    Create a `SampleData` object from an existing `SampleData` object,
    which contains mask sites (all genotypes marked as missing) specified by
    site IDs or positions.

    :param tsinfer.SampleData sd: A SampleData object to mask.
    :param np.ndarray sites: A list of site IDs or positions.
    :param str site_type: IDs ("id") or positions ("position").
    :param str path: Output samples file (default = None).
    :return: A copy of the SampleData object with mask sites.
    :rtype: tsinfer.SampleData
    """
    assert site_type in ["id", "position"], f"Site type {site_type} is invalid."

    new_sd = sample_data.copy(path=path)

    for v in sample_data.variants():
        site_ref = v.site.id if site_type == "id" else v.site.position
        if site_ref in sites:
            new_sd.sites_genotypes[v.site.id] = np.full_like(
                v.genotypes, tskit.MISSING_DATA
            )

    new_sd.finalise()

    return new_sd


def parse_site_position_file(in_file):
    """
    Read list of site positions from a plain tab-delimited text file,
    which has these columns:
    1) chrom
    2) pos
    3) recomb_rate
    4) pos_cm

    TODO: Consider sequence name, which is ignored now.

    :param in_file: A tab-delimied file with site positions.
    :return: A list of site positions.
    :rtype: numpy.ndarray
    """
    expected_columns = ['chrom', 'pos', 'recomb_rate', 'pos_cm']
    df = pd.read_csv(in_file, sep="\t")
    assert np.all(df.columns == expected_columns)
    return df['pos'].to_numpy()
