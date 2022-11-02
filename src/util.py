import json
import numpy as np
from tqdm import tqdm
import tskit
import tsinfer


def count_sites_by_type(ts_or_sd):
    """
    Iterate through the variants of a `TreeSequence` or `SampleData` object,
    and count the number of mono-, bi-, tri-, and quad-allelic sites.

    :param tskit.TreeSequence/tsinfer.SampleData ts_or_sd:
    :return: None
    :rtype: None
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


def check_site_positions_ts_issubset_sd(ts, sd):
    """
    Check whether the site positions in tree sequence are a subset of
    the site positions in samples.

    :param tskit.TreeSequence ts: A tree sequence.
    :param tsinfer.SampleData sd: Samples.
    :return: Are the site positions in the tree sequence a subset of the site positions in the samples?
    :rtype: bool
    """
    ts_site_positions = np.empty(ts.num_sites)
    sd_site_positions = np.empty(sd.num_sites)

    i = 0
    for v in ts.variants():
        ts_site_positions[i] = v.site.position
        i += 1

    j = 0
    for v in sd.variants():
        sd_site_positions[j] = v.site.position
        j += 1

    assert i == ts.num_sites
    assert j == sd.num_sites

    if set(ts_site_positions).issubset(set(sd_site_positions)):
        return True
    else:
        return False


def compare_sites_sd_and_ts(sd, ts, is_common, check_matching_ancestral_state=True):
    """
    If `is_common` is set to True, then get the IDs and positions of the sites
    found in `sd` AND in `ts`.

    if `is_common` is set to False, then get the IDs and positions of the sites
    found in `sd` but NOT in `ts`.

    :param tsinfer.SampleData sd:
    :param tskit.TreeSequence ts:
    :param bool is_common:
    :param bool check_matching_ancestral_state: (default=True)
    :return:
    :rtype: (np.array, np.array,)
    """
    ts_site_positions = np.empty(ts.num_sites)

    i = 0
    for v in ts.variants():
        ts_site_positions[i] = v.site.position
        i += 1

    assert i == ts.num_sites

    sd_site_ids = []
    sd_site_positions = []
    for sd_v in sd.variants():
        if is_common:
            if sd_v.site.position in ts_site_positions:
                sd_site_ids.append(sd_v.site.id)
                sd_site_positions.append(sd_v.site.position)
                if check_matching_ancestral_state:
                    ts_site = ts.site(position=sd_v.site.position)
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


def count_singletons(ts):
    """
    Count the number of singleton sites in a tree sequence.

    :param tskit.TreeSequence ts: A tree sequence.
    :return: Number of singleton sites.
    :rtype: int
    """
    num_singletons = 0

    for v in ts.variants():
        # 0 denotes ancestral allele.
        # 1 denotes derived allele.
        # -1 denotes missing genotypes, so it shouldn't be counted.
        if np.sum(v.genotypes == 1) == 1:
            num_singletons += 1

    return num_singletons


def count_inference_sites(ts):
    """
    Count number of sites used to infer a tree sequence.

    :param tskit.TreeSequence ts: A tree sequence.
    :return: Number of inference sites.
    :rtype: int
    """
    non_inference_sites = [
        s.id for s in ts.sites() if json.loads(s.metadata)["inference_type"] != "full"
    ]

    return ts.num_sites - len(non_inference_sites)


def is_biallelic(ts_or_sd):
    """
    Check whether all the sites in a `TreeSequence` or `SampleData` object
    are biallelic, disregarding missing alleles.

    :param tskit.TreeSequence/tsinfer.SampleData:
    :return: True if all sites are biallelic, otherwise False.
    :rtype: bool
    """
    assert isinstance(
        ts_or_sd, (tskit.TreeSequence, tsinfer.SampleData)
    ), f"Object type is invalid."

    for v in ts_or_sd.variants():
        num_alleles = len(set(v.alleles) - {None})
        if num_alleles != 2:
            return False

    return True


def make_compatible_sample_data(
    sample_data,
    ancestors_ts,
    skip_unused_markers=True,
    chip_site_pos=None,
    mask_site_pos=None,
    path=None,
):
    """
    Make a new `SampleData` object from an existing `SampleData` object such that:
    (1) the derived alleles in `sample_data` not in `ancestors_ts` are marked as `tskit.MISSING`;
    (2) the allele list in `new_sample_data` corresponds to the allele list in `ancestors_ts`.
    (3) sites in `ancestors_ts` but not in `sample_data` are added to `new_sd`
        with all the genotypes `tskit.MISSING`.

    These assumptions must be met:
    (1) All the sites in `sample_data` and `ancestors_ts` must be biallelic.
    (2) `sample_data` and `ancestors_ts` must have the same sequence length.

    Note: Two `SampleData` attributes `sites_alleles` and `sites_genotypes`,
    which are not explained in the tsinfer API doc, are used to facilitate the editing.

    :param tsinfer.SampleData sample_data: Samples which may be incompatible with ancestors_ts.
    :param tskit.TreeSequence ancestors_ts: Ancestors tree serving as the reference panel.
    :param bool skip_unused_markers: Skip markers only in samples (default = True).
    :param numpy.ndarray chip_site_pos: A list of chip site positions.
    :param numpy.ndarray mask_site_pos: A list of mask site positions.
    :param str path: Output samples file (default = None).
    :return: Samples compatible with the tree sequence.
    :rtype: tsinfer.SampleData
    """
    assert sample_data.sequence_length == ancestors_ts.sequence_length, (
        f"Samples has sequence length of {sample_data.sequence_length}, "
        + f"whereas ancestors ts has {ancestors_ts.sequence_length}."
    )

    ts_site_pos = ancestors_ts.sites_position
    sd_site_pos = sample_data.sites_position[:]
    all_site_pos = sorted(set(ts_site_pos).union(set(sd_site_pos)))

    print(f"INFO: Sites in TS = {len(ts_site_pos)}")
    print(f"INFO: Sites in SD = {len(sd_site_pos)}")
    print(f"INFO: Sites in ALL: {len(all_site_pos)}")

    # Keep track of properly aligned sites
    num_case_1 = 0
    num_case_2a = 0
    num_case_2b = 0
    num_case_2c = 0
    num_case_3 = 0

    # Keep track of markers
    num_chip_sites = 0  # In ref. and target
    num_mask_sites = 0  # In ref. and target
    num_unused_sites = 0  # Only in target

    with tsinfer.SampleData(
        sequence_length=ancestors_ts.sequence_length, path=path
    ) as new_sample_data:
        # TODO: Add populations.
        # Add individuals
        for ind in sample_data.individuals():
            new_sample_data.add_individual(
                ploidy=len(ind.samples),
                metadata=ind.metadata,
            )

        # Add sites
        for pos in tqdm(all_site_pos):
            # TODO: Append to existing metadata rather than overwriting it.
            metadata = {}
            if pos in chip_site_pos:
                metadata["marker"] = "chip"
                num_chip_sites += 1
            elif pos in mask_site_pos:
                metadata["marker"] = "mask"
                num_mask_sites += 1
            else:
                metadata["marker"] = ""
                num_unused_sites += 1

            if pos in ts_site_pos and pos not in sd_site_pos:
                # Case 1: Reference markers
                # Site in `ancestors_ts` (ref. panel) but not `sample_data` (target samples).
                # Add the site to `new_sample_data` with all genotypes `tskit.MISSING`.
                num_case_1 += 1

                ts_site = ancestors_ts.site(position=pos)
                assert (
                    len(ts_site.alleles) == 2
                ), f"Non-biallelic site at {pos} in ts: {ts_site.alleles}"
                ts_ancestral_state = ts_site.ancestral_state
                ts_derived_state = list(ts_site.alleles - {ts_ancestral_state})[0]

                new_sample_data.add_site(
                    position=pos,
                    genotypes=np.full(sample_data.num_samples, tskit.MISSING_DATA),
                    alleles=[ts_ancestral_state, ts_derived_state],
                    ancestral_allele=0,
                    metadata=metadata,
                )
            elif pos in ts_site_pos and pos in sd_site_pos:
                # Case 2: Target markers
                # Site in both `ancestors_ts` (ref. panel) and `sample_data` (target samples).
                # Align the allele lists and genotypes if unaligned.
                # Add the site to `new_sample_data` with (aligned) genotypes from `sample_data`.
                ts_site = ancestors_ts.site(position=pos)
                assert (
                    len(ts_site.alleles) == 2
                ), f"Non-biallelic site at {pos} in ts: {ts_site.alleles}"
                ts_ancestral_state = ts_site.ancestral_state
                ts_derived_state = list(ts_site.alleles - {ts_ancestral_state})[0]

                sd_site_id = sd_site_pos.tolist().index(pos)
                sd_site_alleles = sample_data.sites_alleles[sd_site_id]
                assert (
                    len(sd_site_alleles) == 2
                ), f"Non-biallelic site at {pos} in sd: {sd_site_alleles}"
                sd_site_gt = sample_data.sites_genotypes[sd_site_id]

                # Notes
                # `ts_site.alleles` is an unordered set of alleles (without None).
                # `sd_site_alleles` is an ordered list of alleles.
                if [ts_ancestral_state, ts_derived_state] == sd_site_alleles:
                    # Case 2a: Aligned target markers
                    # Both alleles are in `ancestors_ts` and `sample_data`.
                    # Already aligned, so no need to realign.
                    num_case_2a += 1

                    new_sample_data.add_site(
                        position=pos,
                        genotypes=sd_site_gt,
                        alleles=[ts_ancestral_state, ts_derived_state],
                        ancestral_allele=0,
                        metadata=metadata,
                    )
                elif [ts_derived_state, ts_ancestral_state] == sd_site_alleles:
                    # Case 2b: Unaligned target markers
                    # Both alleles are in `ancestors_ts` and `sample_data`.
                    # Align them by flipping the alleles in `sample_data`.
                    num_case_2b += 1

                    new_gt = np.where(
                        sd_site_gt == tskit.MISSING_DATA,
                        tskit.MISSING_DATA,
                        np.where(sd_site_gt == 0, 1, 0),  # Flip
                    )

                    new_sample_data.add_site(
                        position=pos,
                        genotypes=new_gt,
                        alleles=[ts_ancestral_state, ts_derived_state],
                        ancestral_allele=0,
                        metadata=metadata,
                    )
                else:
                    # Case 2c: At least one allele in `sample_data` is not found in `ancestor_ts`.
                    # Allele(s) in `sample_data` but not in `ancestor_ts` is always wrongly imputed.
                    # It is best to ignore these sites when assessing imputation performance.
                    # Also, if there are many such sites, then it should be a red flag.
                    num_case_2c += 1

                    new_allele_list = [ts_ancestral_state, ts_derived_state, None]
                    # Key: index in old allele list; value: index in new allele list
                    index_map = {}
                    for i, a in enumerate(sd_site_alleles):
                        index_map[i] = (
                            new_allele_list.index(a)
                            if a in new_allele_list
                            else tskit.MISSING_DATA
                        )

                    new_sample_data.add_site(
                        position=pos,
                        genotypes=np.vectorize(lambda x: index_map[x])(sd_site_gt),
                        alleles=new_allele_list,
                        ancestral_allele=0,
                        metadata=metadata,
                    )
            elif pos not in ts_site_pos and pos in sd_site_pos:
                # Case 3: Unused target-only markers
                # Site not in `ancestors_ts` but in `sample_data`.
                # Add the site to `new_sample_data` with the original genotypes from `sample_data`.
                num_case_3 += 1

                if skip_unused_markers:
                    continue

                sd_site_id = sd_site_pos.tolist().index(pos)
                sd_site_alleles = sample_data.sites_alleles[sd_site_id]
                assert (
                    len(sd_site_alleles) == 2
                ), f"Non-biallelic site at {pos} in sd: {sd_site_alleles}."

                new_sample_data.add_site(
                    position=pos,
                    genotypes=sample_data.sites_genotypes[sd_site_id],
                    alleles=sd_site_alleles,
                    ancestral_allele=0,
                    metadata=metadata,
                )
            else:
                raise ValueError(f"ERROR: Site at {pos} must be in the ts and/or sd.")

    print(f"INFO: Case 1  (ref.-only)       = {num_case_1}")
    print(f"INFO: Case 2a (both, aligned)   = {num_case_2a}")
    print(f"INFO: Case 2b (both, unaligned) = {num_case_2b}")
    print(f"INFO: Case 2c (flagged)         = {num_case_2c}")
    print(f"INFO: Case 3  (target-only)     = {num_case_3}")
    print(f"INFO: Chip sites                = {num_chip_sites}")
    print(f"INFO: Mask sites                = {num_mask_sites}")
    print(f"INFO: Unused sites              = {num_unused_sites}")

    assert (
        sum(num_case_1, num_case_2a, num_case_2b, num_case_2c, num_case_3)
        == sample_data.num_sites
    )
    assert (
        sum(num_chip_sites, num_mask_sites, num_unused_sites) == sample_data.num_sites
    )

    return new_sample_data
