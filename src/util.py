import json
import numpy as np
import tqdm
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


def make_compatible_sample_data_old(sample_data, ancestors_ts, path=None):
    """
    Make an editable copy of a SampleData object, and edit it so that:
    (1) the derived alleles in `sample_data` not in `ancestors_ts` are marked as MISSING;
    (2) the allele list in `new_sample_data` corresponds to the allele list in `ancestors_ts`.

    Note: Two `SampleData` attributes `sites_alleles` and `sites_genotypes`,
    which are not explained in the tsinfer API doc, are used to facilitate the editing.

    :param tsinfer.SampleData sample_data:
    :param tskit.TreeSequence ancestors_ts:
    :return: A SampleData object compatible with the ancestors TreeSequence.
    :rtype: tsinfer.SampleData
    """
    new_sample_data = sample_data.copy(path=path)

    # Iterate through the sites in `ancestors_ts` using one generator,
    # while iterating through the sites in `sample_data` using another generator,
    # letting the latter generator catch up.
    sd_variants = sample_data.variants()
    sd_v = next(sd_variants)
    for ts_site in tqdm.tqdm(ancestors_ts.sites()):
        print(ts_site.position)
        while sd_v.site.position != ts_site.position:
            # Sites in `samples_data` but not in `ancestors_ts` are not imputed.
            # Also, leave them as is in the `sample_data`, but keep track of them.
            print(sd_v.site.position)
            sd_v = next(sd_variants)

        sd_site_id = sd_v.site.id  # Site id in `sample_data`

        assert len(ts_site.alleles) == 2, f"Site {ts_site.position} is non-biallelic."

        # Get the derived allele in `ancestors_ts` in nucleotide space
        ts_ancestral_allele = ts_site.ancestral_state
        ts_derived_allele = ts_site.alleles - {ts_ancestral_allele}
        assert (
            len(ts_derived_allele) == 1
        ), f"Multiple derived alleles at site {ts_site.position}."
        ts_derived_allele = tuple(ts_derived_allele)[0]

        # CHECK that the ancestral allele should be the same
        # in both `ancestors_ts` and `sample_data`.
        assert (
            ts_ancestral_allele == sd_v.alleles[0]
        ), f"Ancestral alleles are different at site {ts_site.position}."

        if ts_derived_allele not in sd_v.alleles:
            # Case 1:
            # If the derived alleles in the `sample_data` are not in `ancestors_ts`,
            # then mark them as missing.
            #
            # The site in `sample_data` may be mono-, bi-, or multiallelic.
            #
            # We cannot determine whether the extra derived alleles in `sample_data`
            # are derived from 0 or 1 in `ancestors_ts` anyway.
            new_sample_data.sites_genotypes[sd_site_id] = np.where(
                sd_v.genotypes != 0,  # Keep if ancestral
                tskit.MISSING_DATA,  # Otherwise, flag as missing
                0,
            )
            print(
                f"Site {sd_site_id} has no matching derived alleles in the query samples."
            )
            # Update allele list
            new_sample_data.sites_alleles[sd_site_id] = [ts_ancestral_allele]
        else:
            # The allele lists in `ancestors_ts` and `sample_data` may be different.
            ts_derived_allele_index = sd_v.alleles.index(ts_derived_allele)

            if ts_derived_allele_index == 1:
                # Case 2:
                # Both the ancestral and derived alleles correspond exactly.
                if len(sd_v.alleles) == 2:
                    continue
                # Case 3:
                # The derived allele in `ancestors_ts` is indexed as 1 in `sample_data`,
                # so mark alleles >= 2 as missing.
                new_sample_data.sites_genotypes[sd_site_id] = np.where(
                    sd_v.genotypes > 1,  # 0 and 1 should be kept "as is"
                    tskit.MISSING_DATA,  # Otherwise, flag as missing
                    sd_v.genotypes,
                )
                print(
                    f"Site {sd_site_id} has extra derived allele(s) in the query samples (set as missing)."
                )
            else:
                # Case 4:
                #   The derived allele in `ancestors_ts` is NOT indexed as 1 in `sample_data`,
                #   so the alleles in `sample_data` needs to be reordered,
                #   such that the 1-indexed allele is also indexed as 1 in `ancestors_ts`.
                new_sample_data.sites_genotypes[sd_site_id] = np.where(
                    sd_v.genotypes == 0,
                    0,  # Leave ancestral allele "as is"
                    np.where(
                        sd_v.genotypes == ts_derived_allele_index,
                        1,  # Change it to 1 so that it corresponds to `ancestors_ts`
                        tskit.MISSING_DATA,  # Otherwise, mark as missing
                    ),
                )
                print(
                    f"Site {sd_site_id} has the target derived allele at a different index."
                )
            # Update allele list
            new_sample_data.sites_alleles[sd_site_id] = [
                ts_ancestral_allele,
                ts_derived_allele,
            ]

    new_sample_data.finalise()

    return new_sample_data


def make_compatible_sample_data(sample_data, ancestors_ts):
    """
    Make an editable copy of a `sample_data` object, and edit it so that:
    (1) the derived alleles in `sample_data` not in `ancestors_ts` are marked as MISSING;
    (2) the allele list in `new_sample_data` corresponds to the allele list in `ancestors_ts`.
    (3) sites in `ancestors_ts` but not in `sample_data` are added to `new_sample_data` with all the genotypes MISSING.

    All the sites in `sample_data` and `ancestors_ts` must be biallelic.

    Note. Two `SampleData` attributes `sites_alleles` and `sites_genotypes`,
    which are not explained in the tsinfer API doc, are used to facilitate the editing.

    :param tsinfer.SampleData sample_data:
    :param tskit.TreeSequence ancestors_ts:
    :return: An edited copy of sample_data.
    :rtype: tsinfer.SampleData
    """
    ts_site_pos = ancestors_ts.sites_position
    sd_site_pos = sample_data.sites_position[:]
    all_site_pos = sorted(set(ts_site_pos).union(set(sd_site_pos)))

    num_case_1 = 0
    num_case_2a = 0
    num_case_2b = 0
    num_case_2c = 0
    num_case_3 = 0

    with tsinfer.SampleData(sequence_length=ancestors_ts.sequence_length) as new_sd:
        # Add populations
        # TODO: Allow for more populations to be added.
        new_sd.add_population()

        # Add individuals
        for ind in sample_data.individuals():
            new_sd.add_individual(
                ploidy=len(ind.samples),
                population=ind.population,
                metadata=ind.metadata,
            )

        # Add sites
        for pos in tqdm.tqdm(all_site_pos):
            if pos in ts_site_pos and pos not in sd_site_pos:
                # Case 1:
                # Site found in `ancestors_ts` but not `sample_data`
                # Add the site to `new_sample_data` with all genotypes MISSING.
                num_case_1 += 1
                ts_site = ancestors_ts.site(position=pos)
                assert (
                    len(ts_site.alleles) == 2
                ), f"Non-biallelic site {ts_site.alleles}"
                ts_ancestral_state = ts_site.ancestral_state
                ts_derived_state = list(ts_site.alleles - {ts_ancestral_state})[0]
                new_sd.add_site(
                    position=pos,
                    genotypes=np.full(sample_data.num_samples, tskit.MISSING_DATA),
                    alleles=[ts_ancestral_state, ts_derived_state],
                )
            elif pos in ts_site_pos and pos in sd_site_pos:
                # Case 2:
                # Site found in both `ancestors_ts` and `sample_data`
                # Align the allele lists and genotypes if unaligned.
                # Add the site to `new_sample_data` with (aligned) genotypes from `sample_data`.
                ts_site = ancestors_ts.site(position=pos)
                sd_site_id = sd_site_pos.tolist().index(pos)
                sd_site_alleles = sample_data.sites_alleles[sd_site_id]
                assert (
                    len(ts_site.alleles) == 2
                ), f"Non-biallelic site {ts_site.alleles}"
                assert (
                    len(set(sd_site_alleles) - {None}) == 2
                ), f"Non-biallelic site {sd_site_alleles}"
                ts_ancestral_state = ts_site.ancestral_state
                ts_derived_state = list(ts_site.alleles - {ts_ancestral_state})[0]
                if list(ts_site.alleles) == sd_site_alleles:
                    # Case 2a:
                    # Both alleles are in `ancestors_ts` and `sample_data`.
                    # Already aligned, so no need to realign.
                    num_case_2a += 1
                    new_sd.add_site(
                        position=pos,
                        genotypes=sample_data.sites_genotypes[sd_site_id],
                        alleles=[ts_ancestral_state, ts_derived_state],
                    )
                elif set(ts_site.alleles) == set(sd_site_alleles):
                    # Case 2b:
                    # Both alleles are in `ancestors_ts` and `sample_data`.
                    # Align them by flipping the alleles in `sample_data`.
                    num_case_2b += 1
                    sd_site_gt = sample_data.sites_genotypes[sd_site_id]
                    new_gt = np.where(
                        sd_site_gt == tskit.MISSING_DATA,
                        tskit.MISSING_DATA,
                        np.where(sd_site_gt == 0, 1, 0),  # Flip
                    )
                    new_sd.add_site(
                        position=pos,
                        genotypes=new_gt,
                        alleles=[ts_ancestral_state, ts_derived_state],
                    )
                else:
                    # Case 2c:
                    # The allele(s) present in `sample_data` but absent in `ancestor_ts`
                    # is always incorrectly imputed.
                    # It is best to ignore these sites when assess imputation performance.
                    num_case_2c += 1
                    new_sd.add_site(
                        position=pos,
                        genotypes=np.full(sample_data.num_samples, tskit.MISSING_DATA),
                        alleles=[ts_ancestral_state, ts_derived_state],
                    )
            elif pos not in ts_site_pos and pos in sd_site_pos:
                # Case 3:
                # Site found in `sample_data` but not `ancestors_ts`
                # Add the site to `new_sample_data` with the original genotypes from `sample_data`.
                num_case_3 += 1
                sd_site_id = sd_site_pos.tolist().index(pos)
                sd_site_alleles = sample_data.sites_alleles[sd_site_id]
                assert (
                    len(set(sd_site_alleles) - {None}) == 2
                ), f"Non-biallelic site {sd_site_alleles}"
                new_sd.add_site(
                    position=pos,
                    genotypes=sample_data.sites_genotypes[sd_site_id],
                    alleles=sample_data.sites_alleles[sd_site_id],
                )
            else:
                raise ValueError(
                    f"Position {pos} must be in the tree sequence and/or sample data."
                )

    print(f"Case 1 : {num_case_1}")
    print(f"Case 2a: {num_case_2a}")
    print(f"Case 2b: {num_case_2b}")
    print(f"Case 2c: {num_case_2c}")
    print(f"Case 3 : {num_case_3}")
    
    return new_sd
