import json
import logging
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


def make_compatible_samples(
    target_samples,
    ref_samples,
    chip_site_pos,
    mask_site_pos,
    skip_unused_markers=True,
    path=None,
):
    """
    Make `target_samples` compatible with `ref_samples`.

    Create a new `SampleData` object from an existing `SampleData` object such that:
    (1) the derived alleles in `target_samples` not in `ref_samples` are encoded
        as `tskit.MISSING` in `new_samples`;
    (2) the allele list in `new_samples` corresponds to the allele list in `ref_samples`;
    (3) the sites in `ref_samples` not in `target_samples` are added to `new_samples`
        with all the genotypes encoded as `tskit.MISSING`;
    (4) the sites in `target_samples` not in `ref_samples` may be kept as is,
        or skipped optionally.

    These assumptions must be met:
    (1) `ref_samples` and `target_samples` must have the same sequence length.
    (2) All the sites in `ref_samples` and `target_samples` must be biallelic.

    Note that this code uses two `SampleData` attributes `sites_alleles` and `sites_genotypes`,
    which are not explained in the tsinfer API doc.

    :param tsinfer.SampleData target_samples: Target samples incompatible with reference samples.
    :param tsinfer.SampleData ref_samples: Reference samples.
    :param numpy.ndarray chip_site_pos: Chip site positions.
    :param numpy.ndarray mask_site_pos: Mask site positions.
    :param bool skip_unused_markers: Skip markers only in target samples (default = True).
    :param str path: Output samples file (default = None).
    :return: Target samples compatible with reference samples.
    :rtype: tsinfer.SampleData
    """
    assert target_samples.sequence_length == ref_samples.sequence_length, (
        f"Target samples has sequence length of {target_samples.sequence_length}, "
        + f"whereas reference samples has {ref_samples.sequence_length}."
    )

    target_site_pos = target_samples.sites_position
    ref_site_pos = ref_samples.sites_position[:]
    all_site_pos = sorted(set(target_site_pos).union(set(ref_site_pos)))

    logging.info(f"sites in target samples = {len(target_site_pos)}")
    logging.info(f"sites in reference samples = {len(ref_site_pos)}")
    logging.info(f"site union = {len(all_site_pos)}")

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

    # Add helper to get allele list of a site by position.
    def get_site_id_alleles(samples, site_pos, query_pos):
        site_id = np.where(site_pos == query_pos)[0]
        assert len(site_id) == 1, f"Multiple site indices at {query_pos}"
        site_id = site_id[0]
        site_alleles = samples.sites_alleles[site_id]
        assert (
            len(site_alleles) == 2
        ), f"Non-biallelic site at {query_pos}: {site_alleles}"
        return site_id, site_alleles

    with tsinfer.SampleData(
        sequence_length=target_samples.sequence_length, path=path
    ) as new_samples:
        # TODO: Add populations.
        # Add individuals
        for ind in target_samples.individuals():
            new_samples.add_individual(
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

            if pos in ref_site_pos and pos not in target_site_pos:
                # Case 1: Reference markers
                # Site in `ref_samples` not `target_samples`.
                # Add the site to `new_samples` with all genotypes `tskit.MISSING`.
                num_case_1 += 1
                _, ref_alleles = get_site_id_alleles(
                    samples=ref_samples, site_pos=ref_site_pos, query_pos=pos
                )
                new_samples.add_site(
                    position=pos,
                    genotypes=np.full(target_samples.num_samples, tskit.MISSING_DATA),
                    alleles=ref_alleles,
                    ancestral_allele=0,
                    metadata=metadata,
                )
            elif pos in ref_site_pos and pos in target_site_pos:
                # Case 2: Target markers
                # Site in both `ref_samples` and `target_samples`.
                # Align the allele lists and genotypes if unaligned.
                # Add the site to `new_samples` with (aligned) genotypes from `target_samples`.
                _, ref_alleles = get_site_id_alleles(
                    samples=ref_samples, site_pos=ref_site_pos, query_pos=pos
                )
                target_site_id, target_alleles = get_site_id_alleles(
                    samples=target_samples, site_pos=target_site_pos, query_pos=pos
                )
                target_gt = target_samples.sites_genotypes[target_site_id]

                # `ref_alleles` and `target_alleles` are ordered lists.
                if ref_alleles == target_alleles:
                    # Case 2a: Aligned target markers
                    # Both alleles are in `ref_samples` and `target_samples`.
                    # Already aligned, so no need to realign.
                    num_case_2a += 1
                    new_samples.add_site(
                        position=pos,
                        genotypes=target_gt,
                        alleles=ref_alleles,
                        ancestral_allele=0,
                        metadata=metadata,
                    )
                elif ref_alleles[::-1] == target_alleles:
                    # Case 2b: Unaligned target markers
                    # Both alleles are in `ref_samples` and `target_samples`.
                    # Align them by flipping the alleles in `target_samples`.
                    num_case_2b += 1
                    realigned_gt = np.where(
                        target_gt == tskit.MISSING_DATA,
                        tskit.MISSING_DATA,
                        np.where(target_gt == 0, 1, 0),  # Flip
                    )
                    new_samples.add_site(
                        position=pos,
                        genotypes=realigned_gt,
                        alleles=ref_alleles,
                        ancestral_allele=0,
                        metadata=metadata,
                    )
                else:
                    # Case 2c: At least one allele in `target_samples` is not found in `ref_samples`.
                    # Allele(s) in `target_samples` but not in `ref_samples` is always wrongly imputed.
                    # It is best to ignore these sites when assessing imputation performance.
                    # Also, if there are many such sites, then it should be a red flag.
                    num_case_2c += 1
                    new_allele_list = [ref_alleles[0], ref_alleles[1], None]
                    # Key: index in old allele list; value: index in new allele list
                    index_map = {}
                    for i, a in enumerate(target_alleles):
                        index_map[i] = (
                            new_allele_list.index(a)
                            if a in new_allele_list
                            else tskit.MISSING_DATA
                        )
                    new_samples.add_site(
                        position=pos,
                        genotypes=np.vectorize(lambda x: index_map[x])(target_gt),
                        alleles=new_allele_list,
                        ancestral_allele=0,
                        metadata=metadata,
                    )
            elif pos not in ref_site_pos and pos in target_site_pos:
                # Case 3: Unused (target-only) markers
                # Site not in `ref_samples` but in `target_samples`.
                # Add the site to `new_samples` with the original genotypes.
                num_case_3 += 1
                if skip_unused_markers:
                    continue
                target_site_id, target_alleles = get_site_id_alleles(
                    samples=target_samples, site_pos=target_site_pos, query_pos=pos
                )
                target_gt = target_samples.sites_genotypes[target_site_id]

                new_samples.add_site(
                    position=pos,
                    genotypes=target_samples.sites_genotypes[target_site_id],
                    alleles=target_alleles,
                    ancestral_allele=0,
                    metadata=metadata,
                )
            else:
                logging.error(f"site at {pos} must be in the ref. or target samples.")

    logging.info(f"case 1 (ref.-only): {num_case_1}")
    logging.info(f"case 2a (both, aligned): {num_case_2a}")
    logging.info(f"case 2b (both, unaligned): {num_case_2b}")
    logging.info(f"case 2c (flagged): {num_case_2c}")
    logging.info(f"case 3 (target-only): {num_case_3}")
    logging.info(f"chip sites: {num_chip_sites}")
    logging.info(f"mask sites: {num_mask_sites}")
    logging.info(f"unused sites: {num_unused_sites}")

    assert num_case_1 + num_case_2a + num_case_2b + num_case_2c + num_case_3 == len(
        all_site_pos
    )

    if skip_unused_markers:
        assert num_chip_sites + num_mask_sites == len(ref_site_pos)
    else:
        assert num_chip_sites + num_mask_sites + num_unused_sites == len(all_site_pos)

    return new_samples
