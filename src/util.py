import json
import logging
import numpy as np
import tqdm
import tskit
import tsinfer


def make_compatible_samples(
    sd,
    ts,
    skip_unused_markers=True,
    chip_site_pos=None,
    mask_site_pos=None,
    path=None,
):
    """
    Create `new_sd` from `sd` that is compatible with `ts`.

    Create a new `SampleData` object from an existing `SampleData` object such that:
    (1) the derived alleles in `sd` not in `ts` are marked as `tskit.MISSING`;
    (2) the allele list in `new_sd` corresponds to the allele list in `ts`;
    (3) sites in `ts` but not in `sd` are added to `new_sd`
        with all the genotypes `tskit.MISSING`;
    (4) sites in `sd` but not in `ts` are added to `new_sd` as is,
        but they can be optionally skipped.

    These assumptions must be met:
    (1) `sd` and `ts` must have the same sequence length.
    (2) All the sites in `sd` and `ts` must be biallelic.

    Note that this code uses two `SampleData` attributes `sites_alleles` and `sites_genotypes`,
    which are not explained in the tsinfer API doc.

    :param tsinfer.SampleData sd: Samples (possibly) incompatible with ts.
    :param tskit.TreeSequence ts: Tree sequence.
    :param bool skip_unused_markers: Skip markers only in samples (default = True).
    :param array-like chip_site_pos: Chip site positions.
    :param array-like mask_site_pos: Mask site positions.
    :param str path: Output samples file (default = None).
    :return: Samples compatible with ts.
    :rtype: tsinfer.SampleData
    """
    assert sd.sequence_length == ts.sequence_length, \
        f"Sequence length of samples and ts differ."

    sd_site_pos = sd.sites_position[:]
    ts_site_pos = ts.sites_position
    all_site_pos = sorted(set(sd_site_pos).union(set(ts_site_pos)))

    logging.info(f"Sites in samples = {len(sd_site_pos)}")
    logging.info(f"Sites in ts = {len(ts_site_pos)}")
    logging.info(f"Sites in both = {len(all_site_pos)}")

    # Keep track of properly aligned sites
    num_case_1 = 0
    num_case_2a = 0
    num_case_2b = 0
    num_case_2c = 0
    num_case_3 = 0

    # Keep track of markers
    num_chip_sites = 0  # In both ref. and target
    num_mask_sites = 0  # Only in ref.
    num_unused_sites = 0  # Only in target

    with tsinfer.SampleData(
        sequence_length=ts.sequence_length, path=path
    ) as new_sd:
        # Add populations.
        for pop in sd.populations():
            if not isinstance(pop.metadata, dict):
                metadata = json.loads(pop.metadata)
            else:
                metadata = pop.metadata
            new_sd.add_population(metadata=metadata)

        # Add individuals
        for ind in sd.individuals():
            if not isinstance(ind.metadata, dict):
                metadata = json.loads(ind.metadata)
            else:
                metadata = ind.metadata
            new_sd.add_individual(
                ploidy=len(ind.samples),
                metadata=metadata,
            )

        # Add sites
        for pos in tqdm.tqdm(all_site_pos):
            metadata = {}
            if chip_site_pos is not None and pos in chip_site_pos:
                metadata["marker"] = "chip"
                num_chip_sites += 1
            elif mask_site_pos is not None and pos in mask_site_pos:
                metadata["marker"] = "mask"
                num_mask_sites += 1
            else:
                metadata["marker"] = ""
                num_unused_sites += 1

            if pos in ts_site_pos and pos not in sd_site_pos:
                # Case 1: Reference markers.
                # Site in `ts` but not in `sd`.
                # Add the site to `new_sd` with all genotypes `tskit.MISSING`.
                num_case_1 += 1

                ts_site = ts.site(position=pos)
                assert (
                    len(ts_site.alleles) == 2
                ), f"Non-biallelic site at {pos} in ts: {ts_site.alleles}"
                ts_ancestral_state = ts_site.ancestral_state
                ts_derived_state = list(ts_site.alleles - {ts_ancestral_state})[0]

                if not isinstance(ts_site.metadata, dict):
                    metadata = json.loads(ts_site.metadata) | metadata

                new_sd.add_site(
                    position=pos,
                    genotypes=np.full(sd.num_samples, tskit.MISSING_DATA),
                    alleles=[ts_ancestral_state, ts_derived_state],
                    ancestral_allele=0,
                    metadata=metadata,
                )
            elif pos in ts_site_pos and pos in sd_site_pos:
                # Case 2: Target markers.
                # Site in both `ts` and `sd`.
                # Align the allele lists and genotypes if unaligned.
                # Add the site to `new_sd` with (aligned) genotypes from `sd`.
                ts_site = ts.site(position=pos)
                assert (
                    len(ts_site.alleles) == 2
                ), f"Non-biallelic site at {pos} in ts: {ts_site.alleles}"
                ts_ancestral_state = ts_site.ancestral_state
                ts_derived_state = list(ts_site.alleles - {ts_ancestral_state})[0]

                sd_site_id = np.where(sd_site_pos == pos)[0][0]
                sd_site_alleles = sd.sites_alleles[sd_site_id]
                assert (
                    len(sd_site_alleles) == 2
                ), f"Non-biallelic site at {pos} in sd: {sd_site_alleles}"
                sd_site_gt = sd.sites_genotypes[sd_site_id]

                if not isinstance(ts_site.metadata, dict):
                    metadata = json.loads(ts_site.metadata) | metadata

                # Notes
                # `ts_site.alleles` is an unordered set of alleles (without None).
                # `sd_site_alleles` is an ordered list of alleles.
                if [ts_ancestral_state, ts_derived_state] == sd_site_alleles:
                    # Case 2a: Aligned target markers
                    # Both alleles are in `ts` and `sd`.
                    # Already aligned, so no need to realign.
                    num_case_2a += 1

                    new_sd.add_site(
                        position=pos,
                        genotypes=sd_site_gt,
                        alleles=[ts_ancestral_state, ts_derived_state],
                        ancestral_allele=0,
                        metadata=metadata,
                    )
                elif [ts_derived_state, ts_ancestral_state] == sd_site_alleles:
                    # Case 2b: Unaligned target markers.
                    # Both alleles are in `ts` and `sd`.
                    # Align them by flipping the alleles in `sd`.
                    num_case_2b += 1

                    aligned_gt = np.where(
                        sd_site_gt == tskit.MISSING_DATA,
                        tskit.MISSING_DATA,
                        np.where(sd_site_gt == 0, 1, 0),  # Flip
                    )

                    new_sd.add_site(
                        position=pos,
                        genotypes=aligned_gt,
                        alleles=[ts_ancestral_state, ts_derived_state],
                        ancestral_allele=0,
                        metadata=metadata,
                    )
                else:
                    # Case 2c: At least one allele in `sd` is not found in `ts`.
                    # Allele(s) in `sd` but not in `ts` is always wrongly imputed.
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

                    new_sd.add_site(
                        position=pos,
                        genotypes=np.vectorize(lambda x: index_map[x])(sd_site_gt),
                        alleles=new_allele_list,
                        ancestral_allele=0,
                        metadata=metadata,
                    )
            elif pos not in ts_site_pos and pos in sd_site_pos:
                # Case 3: Unused (target-only) markers
                # Site not in `ts` but in `sd`.
                # Add the site to `new_sd` with the original genotypes.
                num_case_3 += 1

                if skip_unused_markers:
                    continue

                sd_site_id = np.where(sd_site_pos == pos)[0][0]
                sd_site_alleles = sd.sites_alleles[sd_site_id]
                assert (
                    len(sd_site_alleles) == 2
                ), f"Non-biallelic site at {pos} in sd: {sd_site_alleles}."

                new_sd.add_site(
                    position=pos,
                    genotypes=sd.sites_genotypes[sd_site_id],
                    alleles=sd_site_alleles,
                    ancestral_allele=0,
                    metadata=metadata,
                )
            else:
                logging.error(f"Site at {pos} must be in the ts and/or sd.")

    logging.info(f"Case 1 (ref.-only): {num_case_1}")
    logging.info(f"Case 2a (both, aligned): {num_case_2a}")
    logging.info(f"Case 2b (both, unaligned): {num_case_2b}")
    logging.info(f"Case 2c (flagged): {num_case_2c}")
    logging.info(f"Case 3 (target-only): {num_case_3}")
    logging.info(f"Chip sites: {num_chip_sites}")
    logging.info(f"Mask sites: {num_mask_sites}")
    logging.info(f"Unused sites: {num_unused_sites}")

    assert (
        num_case_1 + num_case_2a + num_case_2b + num_case_2c + num_case_3
        == len(all_site_pos)
    )

    return new_sd


def add_sample_to_tree_sequence(ts, path, metadata):
    assert ts.num_sites == len(path), \
        f"Sample path is of different length than tree sequence."
    assert np.isin(path, np.arange(ts.num_samples)), \
        f"Sample IDs in sample path are not found in tree sequence."

    tables = ts.dump_tables()

    # Add an individual to the individuals table
    # TODO: Add metadata
    ind_id = tables.individuals.add_row()

    # Add a sample node to the nodes table
    node_id = tables.nodes.add_row(
        flags=1, # Flag for a sample
        time=-1, # Arbitrarily set to be younger than samples in the existing ts
        population=0,
        individual=ind_id,
        metadata=metadata
    )

    # Add edges to the edges table
    for i in np.arange(ts.num_sites):
        if i == ts.num_sites - 1:
            break
        tables.edges.add_row(
            left=ts.sites_position[i],
            right=ts.sites_position[i + 1],
            parent=path[i],
            child=node_id
        )
    # TODO: The proper way is to add the full edges rather than squashing.
    tables.edges.squash()

    return(tables.tree_sequence())
