from dataclasses import dataclass, field
import json
import logging
import numpy as np
import tqdm
import tskit
import tsinfer


# Functions for writing and reading contents
def print_tsdata_to_vcf(
    tsdata,
    ploidy,
    contig_name,
    out_prefix,
    site_mask=None,
    exclude_mask_sites=False,
    exclude_monoallelic_sites=False,
):
    """
    Print the contents of a `SampleData` or `TreeSequence` object in VCF 4.2.

    Fields:
        CHROM contig_name
        POS 1-based
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
            individual n - 1

    :param tskit.TreeSequence/tsinfer.SampleData tsdata: Tree sequence or samples.
    :param int ploidy: 1 or 2.
    :param str contig_name: Contig name.
    :param str out_prefix: Output file prefix (*.vcf).
    :param array_like site_mask: Site positions to mask.
    :param bool exclude_mask_sites: Exclude masked sites.
    :param bool exclude_monoallelic_sites: Exclude monoallelic sites.
    """
    CHROM = contig_name
    ID = "."
    QUAL = "."
    FILTER = "PASS"
    FORMAT = "GT"

    if ploidy not in [1, 2]:
        raise ValueError(f"Ploidy {ploidy} is not recognized.")

    if isinstance(tsdata, tsinfer.SampleData):
        individual_names = [x.metadata["sample"] for x in tsdata.individuals()]
    elif isinstance(tsdata, tskit.TreeSequence):
        individual_names = [json.loads(x.metadata)["sample"] for x in tsdata.individuals()]
    else:
        raise TypeError(f"tsdata must be a SampleData or TreeSequence object.")

    header = (
        "##fileformat=VCFv4.2\n" + \
        "##source=tskit " + tskit.__version__ + "\n" + \
        "##INFO=<ID=AA,Number=1,Type=String,Description=\"Ancestral Allele\">\n" + \
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n" + \
        "##contig=<ID=" + contig_name + "," + \
        "length=" + str(int(tsdata.sequence_length)) + ">\n"
    )
    header += "\t".join(
        ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        + individual_names
    )

    out_file = out_prefix + ".vcf"
    with open(out_file, "w") as f:
        f.write(header + "\n")
        for v in tqdm.tqdm(tsdata.variants(), total=tsdata.num_sites):
            # Site positions are stored as float in tskit
            POS = int(v.site.position)
            # If the ts was produced by simulation,
            # there's no ref. sequence other than the ancestral sequence.
            REF = v.site.ancestral_state
            alt_alleles = list(set(v.alleles) - {REF} - {None})
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
                gt = np.repeat(missing_gt, tsdata.num_individuals)
            else:
                gt = v.genotypes.astype(str)
                if ploidy == 2:
                    a1 = gt[np.arange(0, tsdata.num_samples, 2)]
                    a2 = gt[np.arange(1, tsdata.num_samples, 2)]
                    gt = np.char.join('|', np.char.add(a1, a2))

            f.write("\t".join(np.concatenate([record, gt])) + "\n")


# Functions concerning object compatibility
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
    for site_pos in tqdm.tqdm(tsdata_1.sites_position):
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
    skip_unused_markers=None,
    chip_site_pos=None,
    mask_site_pos=None,
    path=None,
):
    """
    Create a new `SampleData` object (`new_sd`) from an existing one (`sd`) such that:
    (1) The derived alleles in `sd` not in `ts` are marked as `tskit.MISSING`.
    (2) The allele list in `new_sd` corresponds to the allele list in `ts` at each site.
    (3) Sites in `ts` but not in `sd` are added to `new_sd`
        with all the genotypes `tskit.MISSING`.
    (4) Sites in `sd` but not in `ts` are added to `new_sd` as is,
        but they can be optionally skipped.

    Note that this code uses two `SampleData` attributes `sites_alleles` and `sites_genotypes`,
    which are not explained in the tsinfer API doc.

    :param tsinfer.SampleData sd: Samples possibly incompatible with tree sequence.
    :param tskit.TreeSequence ts: Tree sequence.
    :param bool skip_unused_markers: Skip markers only in samples. If None, don't skip (default = None).
    :param array-like chip_site_pos: Chip site positions (default = None).
    :param array-like mask_site_pos: Mask site positions (default = None).
    :param str path: Output samples file (default = None).
    :return: Samples compatible with tree sequence.
    :rtype: tsinfer.SampleData
    """
    if not isinstance(sd, tsinfer.SampleData):
        raise TypeError(f"sd must be a SampleData object.")
    if not isinstance(ts, tskit.TreeSequence):
        raise TypeError(f"ts must be a TreeSequence object.")

    # Check all sites in sd are mono- or biallelic.    
    for v in sd.variants():
        if len(set(v.alleles) - {None}) > 2:
            raise ValueError(f"All sites in sd must be mono- or biallelic.")

    # Check all sites in ts are biallelic.
    for v in ts.variants():
        if len(set(v.alleles) - {None}) != 2:
            raise ValueError(f"All sites in ts must be biallelic.")

    sd_site_pos = sd.sites_position[:]
    ts_site_pos = ts.sites_position
    all_site_pos = sorted(set(sd_site_pos).union(set(ts_site_pos)))

    print(f"Sites in sd = {len(sd_site_pos)}")
    print(f"Sites in ts = {len(ts_site_pos)}")
    print(f"Sites in both = {len(all_site_pos)}")

    # Keep track of properly aligned sites
    num_case_1 = 0
    num_case_2a = 0
    num_case_2b = 0
    num_case_2c = 0
    num_case_2d = 0
    num_case_3 = 0

    # Keep track of types of markers
    num_chip_sites = 0
    num_mask_sites = 0

    with tsinfer.SampleData(
        sequence_length=ts.sequence_length, path=path
    ) as new_sd:
        # Add populations
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

            if pos in ts_site_pos and pos not in sd_site_pos:
                # Case 1: Reference markers.
                # Site in `ts` but not in `sd`.
                # Add the site to `new_sd` with all genotypes `tskit.MISSING`.
                num_case_1 += 1

                ts_site = ts.site(position=pos)
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
                ts_ancestral_state = ts_site.ancestral_state
                ts_derived_state = list(ts_site.alleles - {ts_ancestral_state})[0]

                sd_site_id = np.where(sd_site_pos == pos)[0][0]
                sd_site_alleles = sd.sites_alleles[sd_site_id]
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
                elif len(sd_site_alleles) == 2:
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
                elif len(sd_site_alleles) == 1:
                    # Case 2d: Only one allele in `sd`.
                    num_case_2d += 1

                    if sd_site_alleles[0] == ts_ancestral_state:
                        new_sd.add_site(
                            position=pos,
                            genotypes=np.full(sd.num_samples, 0),
                            alleles=[ts_ancestral_state, ts_derived_state],
                            ancestral_allele=0,
                            metadata=metadata,
                        )
                    elif sd_site_alleles[0] == ts_derived_state:
                        new_sd.add_site(
                            position=pos,
                            genotypes=np.full(sd.num_samples, 1),
                            alleles=[ts_ancestral_state, ts_derived_state],
                            ancestral_allele=0,
                            metadata=metadata,
                        )
                    else:
                        raise ValueError(f"Allele in sd not found in ts at {pos}.")
                else:
                    raise ValueError(f"Unexpected patterns of allele lists at {pos}.")
            elif pos not in ts_site_pos and pos in sd_site_pos:
                # Case 3: Unused (target-only) markers
                # Site not in `ts` but in `sd`.
                # Add the site to `new_sd` with the original genotypes.
                num_case_3 += 1

                if skip_unused_markers:
                    continue

                sd_site_id = np.where(sd_site_pos == pos)[0][0]
                sd_site_alleles = sd.sites_alleles[sd_site_id]

                new_sd.add_site(
                    position=pos,
                    genotypes=sd.sites_genotypes[sd_site_id],
                    alleles=sd_site_alleles,
                    ancestral_allele=0,
                    metadata=metadata,
                )
            else:
                raise ValueError(f"Site at {pos} must be in the ts and/or sd.")

    print(f"Case 1 (ref.-only): {num_case_1}")
    print(f"Case 2a (both, aligned): {num_case_2a}")
    print(f"Case 2b (both, unaligned): {num_case_2b}")
    print(f"Case 2c (flagged): {num_case_2c}")
    print(f"Case 2d (monoallelic in target): {num_case_2d}")
    print(f"Case 3 (target-only): {num_case_3}")
    print(f"Chip sites: {num_chip_sites}")
    print(f"Mask sites: {num_mask_sites}")

    assert (
        num_case_1 + num_case_2a + num_case_2b + num_case_2c + num_case_2d + num_case_3
        == len(all_site_pos)
    )

    return new_sd


# Functions for adding a new individual to an existing tree sequence
@dataclass
class SamplePath:
    """
    Convenience class for storing the sample paths of an individual.
    Each sample path is defined by a list of ids of nodes in a tree sequence.
    
    Definition of a valid `SamplePath` object:
    1. The sizes of the `nodes` and `site_positions` attributes are equal.
    2. The `site_positions` attribute is sorted in ascending order.

    individual: Name of individual.
    nodes: Sample path (list of node ids).
    site_positions: Site positions corresponding to the path.
    is_valid: Boolean indicating whether the path is valid.
    metadata: Metadata associated with the path (optional).
    """
    individual: str
    nodes: np.ndarray
    site_positions: np.ndarray
    metadata: dict = None
    is_valid: bool = field(init=False)

    def __len__(self):
        return(self.nodes.size)

    def __post_init__(self):
        self.is_valid = False
        is_nodes_site_positions_equal_length = self.nodes.size == self.site_positions.size
        is_site_positions_sorted = np.all(self.site_positions[:-1] < self.site_positions[1:])
        if is_nodes_site_positions_equal_length and is_site_positions_sorted:
            self.is_valid = True


def get_switch_mask(path):
    """
    Called by `get_switch_site_positions` and `get_num_switches`.
    
    :param SamplePath path: Sample path.
    :return: Boolean indicating whether the path switches at each site.
    :rtype: numpy.ndarray
    """
    is_switch = np.zeros(len(path), dtype=bool)
    is_switch[1:] = np.invert(np.equal(path.nodes[1:], path.nodes[:-1]))
    return(is_switch)


def get_switch_site_positions(path):
    """
    Get the positions of the sites at which the individual switches.

    :param SamplePath path: Sample path.
    :return: Site positions where the path switches.
    :rtype: numpy.ndarray
    """
    is_switch = get_switch_mask(path)
    return(path.site_positions[is_switch])


def get_num_switches(path):
    """
    :param SamplePath path: Sample path.
    :return: Number of switches in the path.
    :rtype: int
    """
    return(np.sum(get_switch_mask(path)))


def add_individual_to_tree_sequence(ts, paths, metadata=None):
    """
    Add an individual, which has at least one sample path,
    to an existing tree sequence.

    :param tskit.TreeSequence ts: Tree sequence to which the individual is added.
    :param list(util.SamplePath) paths: List of SamplePath objects.
    :param dict metadata: Metadata for the individual.
    """
    if not len(paths) > 0:
        raise ValueError("At least one sample path must be provided.")

    for i in np.arange(len(paths)):
        if ts.num_sites != len(paths[i]):
            raise ValueError("Lengths of path and ts are not equal.")
        if not np.all(np.isin(paths[i].nodes, np.arange(ts.num_nodes))):
            raise ValueError("Not all node ids in path are not in ts.")

    tables = ts.dump_tables()

    # Add an individual to the individuals table
    new_ind_id = tables.individuals.add_row(metadata=metadata)

    for i in np.arange(len(paths)):
        # Add a new sample node to the nodes table
        new_node_id = tables.nodes.add_row(
            flags=1, # Flag for a sample
            time=-1, # Arbitrarily set to be younger than samples in ts
            population=0,   # TODO: Associate it with a specific population
            individual=new_ind_id,
            metadata=paths[i].metadata,
        )

        # Add new edges to the edges table
        is_switch = get_switch_mask(paths[i])
        switch_pos = ts.sites_position[is_switch]
        parent_at_switch_pos = paths[i].nodes[is_switch]

        if len(switch_pos) == 0:
            # Add one edge when there is no switch
            assert len(np.unique(paths[i].nodes)) == 1
            edge_id = tables.edges.add_row(
                left=0,
                right=ts.sequence_length,
                parent=paths[i].nodes[0],
                child=new_node_id,
            )
        else:
            # Add multiple edges when there is at least one switch
            # Add first edge
            edge_id = tables.edges.add_row(
                left=0,
                right=switch_pos[0],
                parent=paths[i].nodes[0],
                child=new_node_id,
            )
            for j in np.arange(len(switch_pos)):
                if j == len(switch_pos) - 1:
                    # Add last edge
                    tmp_left = switch_pos[j]
                    tmp_right = ts.sequence_length
                    tmp_parent = parent_at_switch_pos[j]
                else:
                    # Add middle edge(s), if any
                    tmp_left = switch_pos[j]
                    tmp_right = switch_pos[j + 1]
                    tmp_parent = parent_at_switch_pos[j]
                edge_id = tables.edges.add_row(
                    left=tmp_left,
                    right=tmp_right,
                    parent=tmp_parent,
                    child=new_node_id,
                )

        tables.sort()

    return((new_ind_id, tables.tree_sequence()))
