import numpy as np
import tskit
import tsinfer


def count_sites_by_type(ts_or_sd):
    """
    Iterate through the variants of a `TreeSequence` or `SampleData` object,
    and count the number of mono-, bi-, tri-, and quad-allelic sites.

    :param tskit.TreeSequence/tsinfer.SampleData ts_or_sd:
    :return None:
    """
    assert isinstance(ts_or_sd, (tskit.TreeSequence, tsinfer.SampleData))

    num_sites_mono = 0
    num_sites_bi = 0
    num_sites_bi_singleton = 0
    num_sites_tri = 0
    num_sites_quad = 0

    for v in ts_or_sd.variants():
        num_alleles = len(set(v.alleles) - {None})
        if num_alleles == 1:
            num_sites_mono += 1
        elif num_alleles == 2:
            num_sites_bi += 1
            if np.sum(v.genotypes) == 1:
                num_sites_bi_singleton += 1
        elif num_alleles == 3:
            num_sites_tri += 1
        else:
            num_sites_quad += 1

    num_sites_total = num_sites_mono + num_sites_bi + num_sites_tri + num_sites_quad

    print(f"\tsites mono : {num_sites_mono}")
    print(f"\tsites bi   : {num_sites_bi} ({num_sites_bi_singleton} singletons)")
    print(f"\tsites tri  : {num_sites_tri}")
    print(f"\tsites quad : {num_sites_quad}")
    print(f"\tsites total: {num_sites_total}")

    return None


def check_site_positions_ts_issubse_sd(tree_sequence, sample_data):
    """
    Check whether the site positions in `tskit.TreeSequence` are a subset of
    the site positions in `tsinfer.SampleData`.

    :param tskit.TreeSequence tree_sequence:
    :param tsinfer.SampleData sample_data:
    :return:
    :rtype: bool
    """
    ts_site_positions = np.empty(tree_sequence.num_sites)
    sd_site_positions = np.empty(sample_data.num_sites)

    i = 0
    for v in tree_sequence.variants():
        ts_site_positions[i] = v.site.position
        i += 1

    j = 0
    for v in sample_data.variants():
        sd_site_positions[j] = v.site.position
        j += 1

    assert i == tree_sequence.num_sites
    assert j == sample_data.num_sites

    if set(ts_site_positions).issubset(set(sd_site_positions)):
        return True
    else:
        return False


def compare_sites_sd_and_ts(
    sample_data, tree_sequence, is_common, check_matching_ancestral_state=True
):
    """
    If `is_common` is set to True, then get the ids and positions of the sites
    found in `sample_data` AND in `tree_sequence`.

    if `is_common` is set to False, then get the ids and positions of the sites
    found in `sample_data` but NOT in `tree_sequence`.

    :param TreeSequence tree_sequence:
    :param SampleData sample_data:
    :param is_common bool:
    :param check_matching_ancestral_state bool: (default=True)
    :return:
    :rtype: 2-tuple of np.array
    """
    ts_site_positions = np.empty(tree_sequence.num_sites)

    i = 0
    for v in tree_sequence.variants():
        ts_site_positions[i] = v.site.position
        i += 1

    assert i == tree_sequence.num_sites

    sd_site_ids = []
    sd_site_positions = []
    for sd_v in sample_data.variants():
        if is_common:
            if sd_v.site.position in ts_site_positions:
                sd_site_ids.append(sd_v.site.id)
                sd_site_positions.append(sd_v.site.position)
                if check_matching_ancestral_state:
                    ts_site = tree_sequence.site(position=sd_v.site.position)
                    assert sd_v.site.ancestral_state == ts_site.ancestral_state, (
                        f"Ancestral states at {sd_v.site.position} not the same, "
                        + f"{sd_v.site.ancestral_state} vs. {ts_site.ancestral_state}."
                    )
        else:
            if sd_v.site.position not in ts_site_positions:
                sd_site_ids.append(sd_v.site.id)
                sd_site_positions.append(sd_v.site.position)

    return (
        np.array(sd_site_ids),
        np.array(sd_site_positions),
    )
