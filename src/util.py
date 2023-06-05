import json
import logging
import numpy as np
import tqdm
import tskit
import tsinfer


def print_samples_to_vcf(
    sd,
    ploidy,
    contig_name,
    out_file,
    site_mask=None,
    exclude_mask_sites=False,
    exclude_monoallelic_sites=False,
):
    """
    Print the contents of a samples file in VCF 4.2 format.

    Fields:
        CHROM contig_id
        POS
        ID .
        REF ancestral allele
        ALT derived allele(s)
        QUAL .
        FILTER PASS
        INFO
        FORMAT GT
            individual 0
            individual 1
            ...
            individual n - 1; n = number of individuals

    :param tsinfer.SampleData sd: Samples.
    :param int ploidy: 1 or 2.
    :param str contig_name: Contig name.
    :param array_like site_mask: Site positions to mask.
    :param bool exclude_mask_sites: Exclude masked sites.
    :param bool exclude_monoallelic_sites: Exclude monoallelic sites.
    :param click.Path out_file: Path to output VCF file.
    """
    CHROM = contig_name
    ID = "."
    QUAL = "."
    FILTER = "PASS"
    FORMAT = "GT"

    assert ploidy in [1, 2], f"Ploidy {ploidy} is not recognized."

    header = (
        "##fileformat=VCFv4.2\n" + \
        "##source=tskit " + tskit.__version__ + "\n" + \
        "##INFO=<ID=AA,Number=1,Type=String,Description=\"Ancestral Allele\">\n" + \
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n" + \
        "##contig=<ID=" + contig_name + "," + \
        "length=" + str(int(sd.sequence_length)) + ">\n"
    )
    header += "\t".join(
        ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        + [str(x.metadata["sample"]) for x in sd.individuals()]
    )

    with open(out_file, "w") as f:
        f.write(header + "\n")
        for v in tqdm.tqdm(sd.variants(), total=sd.num_sites):
            # Site positions are stored as float in tskit
            POS = int(v.site.position)
            # If the ts was produced by simulation,
            # there's no ref. sequence other than the ancestral sequence.
            REF = v.site.ancestral_state
            alt_alleles = list(set(v.alleles) - {REF})
            AA = v.site.ancestral_state
            ALT = ",".join(alt_alleles) if len(alt_alleles) > 0 else "."
            INFO = "AA" + "=" + AA

            record = np.array([CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT], dtype=str)

            if exclude_monoallelic_sites:
                is_monoallelic = len(np.unique(v.genotypes)) == 1
                if is_monoallelic:
                    continue

            if site_mask is not None and POS in site_mask:
                if exclude_mask_sites:
                    continue
                missing_gt = '.' if ploidy == 1 else '.|.'
                gt = np.repeat(missing_gt, sd.num_individuals)
            else:
                gt = v.genotypes.astype(str)
                if ploidy == 2:
                    a1 = gt[np.arange(0, sd.num_samples, 2)]
                    a2 = gt[np.arange(1, sd.num_samples, 2)]
                    gt = np.char.join('|', np.char.add(a1, a2))

            f.write("\t".join(np.concatenate([record, gt])) + "\n")


def is_compatible(tsdata_1, tsdata_2):
    """
    Check if `tsdata_1` is compatible with `tsdata_2`.

    Definition of compatibility:
    1. Same list of site positions.
    2. Same allele list at each site.

    If `None` (or `tskit.MISSING_DATA`) is present in the allele list,
    it is removed before comparison.

    :param tsinfer.SampleData/tskit.TreeSequence tsdata_1: Samples or tree sequence.
    :param tsinfer.SampleData/tskit.TreeSequence tsdata_2: Samples or tree sequence.
    :return: True if compatible, False otherwise.
    :rtype: bool
    """
    # Check condition 1
    if isinstance(tsdata_1, tsinfer.SampleData):
        site_pos_1 = tsdata_1.sites_position[:]
    elif isinstance(tsdata_1, tskit.TreeSequence):
        site_pos_1 = tsdata_1.sites_position
    else:
        raise TypeError(f"tsdata_1 must be a SampleData or TreeSequence object.")

    if isinstance(tsdata_2, tsinfer.SampleData):
        site_pos_2 = tsdata_2.sites_position[:]
    elif isinstance(tsdata_2, tskit.TreeSequence):
        site_pos_2 = tsdata_2.sites_position
    else:
        raise TypeError(f"tsdata_2 must be a SampleData or TreeSequence object.")

    is_site_pos_equal = np.array_equal(site_pos_1, site_pos_2)
    if not is_site_pos_equal:
        print(f"Site positions are not equal.")
        return False

    # Check condition 2
    iter_1 = tsdata_1.variants()
    iter_2 = tsdata_2.variants()
    var_1 = next(iter_1)
    var_2 = next(iter_2)
    for site_pos in tsdata_1.sites_position:
        while var_1.site.position != site_pos:
            var_1 = next(iter_1)
        while var_2.site.position != site_pos:
            var_2 = next(iter_2)
        alleles_1 = var_1.alleles[:-1] if var_1.alleles[-1] is None else var_1.alleles
        alleles_2 = var_2.alleles[:-1] if var_2.alleles[-1] is None else var_2.alleles
        if alleles_1 != alleles_2:
            print(f"Allele lists at {site_pos} are not equal.")
            print(f"tsdata_1: {var_1.alleles}")
            print(f"tsdata_2: {var_2.alleles}")
            return False

    return True


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


# Functions for adding new sample edges to an existing tree sequence
def get_switch_mask(path):
    is_switch = np.zeros(len(path), dtype=bool)
    is_switch[1:] = np.invert(np.equal(path[1:], path[:-1]))
    return(is_switch)


def get_switch_site_positions(path, site_positions):
    assert len(path) == len(site_positions), \
        f"Lengths of sample path and site positions are not the same."
    is_switch = get_switch_mask(path)
    return(site_positions[is_switch])


def get_num_switches(path):
    return(np.sum(get_switch_mask(path)))


def add_sample_to_tree_sequence(ts, path, metadata):
    assert ts.num_sites == len(path), \
        f"Sample path is of different length than tree sequence."
    assert np.all(np.isin(path, np.arange(ts.num_samples))), \
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
    is_switch = get_switch_mask(path)
    switch_pos = ts.sites_position[is_switch]
    parent_at_switch_pos = path[is_switch]
    # Add the first edge
    tables.edges.add_row(
        left=0,
        right=switch_pos[0],
        parent=path[0],
        child=node_id,
    )
    for i in np.arange(len(switch_pos) - 1):
        tables.edges.add_row(
            left=switch_pos[i],
            right=switch_pos[i + 1],
            parent=parent_at_switch_pos[i],
            child=node_id,
        )
    # Add last edge
    tables.edges.add_row(
        left=switch_pos[-1],
        right=ts.sequence_length - 1,
        parent=parent_at_switch_pos[-1],
        child=node_id,
    )
    tables.sort()

    # TODO: Return also individual ID.
    return(tables.tree_sequence())
