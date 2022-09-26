from collections import OrderedDict
import warnings
from tqdm import tqdm
import numpy as np
import cyvcf2
import tskit
import tsinfer


def print_sample_data_to_vcf(
    sample_data, individuals, samples, ploidy_level, site_mask, contig_id, out_vcf_file
):
    """
    Fields:
    CHROM contig_id
    POS row index in genotype_matrix
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

    :param tsinfer.SampleData sample_data:
    :param array_like individuals: List of individual IDs
    :param array_like samples: List of sample IDs
    :param int ploidy_level: 1 (haploid) or 2 (diploid)
    :param array_like site_mask: List of booleans indicating to exclude the site (True) or not (False)
    :param str contig_id:
    :param click.Path out_vcf_file:
    """
    CHROM = contig_id
    ID = "."
    QUAL = "."
    FILTER = "PASS"
    FORMAT = "GT"

    assert (
        ploidy_level == 1 or ploidy_level == 2
    ), f"Ploidy {ploidy_level} is not recognized."

    assert ploidy_level * len(individuals) == len(
        samples
    ), f"Some individuals may not have the same ploidy of {ploidy_level}."

    # Assume that both sample and individual ids are ordered the same way.
    # individual_id_map = np.repeat(individuals, 2)

    header = (
        "##fileformat=VCFv4.2\n"
        + "##source=tskit "
        + tskit.__version__
        + "\n"
        + '##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">\n'
        + '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    )
    header += (
        "##contig=<ID="
        + contig_id
        + ","
        + "length="
        + str(int(sample_data.sequence_length))
        + ">\n"
    )
    header += "\t".join(
        ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        + ["s" + str(x) for x in individuals]
    )

    with open(out_vcf_file, "w") as f:
        f.write(header + "\n")
        for v in sample_data.variants():
            # Site positions are stored as float in tskit
            POS = int(np.round(v.site.position))
            # Since the tree sequence was produced using simulation,
            #    there's no reference sequence other than the ancestral sequence.
            REF = v.site.ancestral_state
            alt_alleles = list(set(v.alleles) - {REF})
            AA = v.site.ancestral_state
            ALT = ",".join(alt_alleles) if len(alt_alleles) > 0 else "."
            INFO = "AA" + "=" + AA
            record = [
                str(x) for x in [CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT]
            ]

            for j in individuals:
                # sample_ids = [samples[x]
                #              for x
                #              in np.where(individual_id_map == j)[0].tolist()]
                # genotype = "|".join([str(variant.genotypes[k])
                #                     for k
                #                     in sample_ids])
                if ploidy_level == 1:
                    genotype = str(v.genotypes[j])
                else:
                    genotype = (
                        str(v.genotypes[2 * j]) + "|" + str(v.genotypes[2 * j + 1])
                    )

                if site_mask is not None:
                    if site_mask.query_position(individual=j, position=POS) == True:
                        if ploidy_level == 1:
                            genotype = "."
                        else:
                            genotype = ".|."  # Or "./."
                record += [genotype]

            f.write("\t".join(record) + "\n")


def get_sequence_length(vcf, seq_name):
    """
    Get the sequence length by sequence name.

    :param cyvcf2.vcf vcf: VCF object containing header with sequence names and lengths.
    :param str seq_name: Sequence name.
    :return: Sequence length.
    :rtype: int
    """
    assert len(vcf.seqnames) == len(vcf.seqlens),\
        f"Number of sequence names different than number of sequence length."
    for seqname, seqlen in zip(vcf.seqnames, vcf.seqlens):
        if seqname == seq_name:
            return seqlen
    return None


def add_populations(vcf, sample_data):
    """
    Add populations to an existing `SampleData` object using data from an existing `VCF` object.

    Sourced and modified from:
    https://tsinfer.readthedocs.io/en/latest/tutorial.html#data-example

    :param cyvcf2.VCF vcf: A VCF object containing variant data
    :param tsinfer.SampleData sample_data:
    :return: Samples by population ID rather than sample names
    :rtype: list
    """
    # Here, the first letter of the sample name is the population name.
    pop_names = [sample_name[0] for sample_name in vcf.samples]

    pop_codes = np.unique(pop_names)
    pop_lookup = {}
    for p in pop_codes:
        pop_lookup[p] = sample_data.add_population(metadata={"name": p})

    pop_ids = [pop_lookup[p] for p in pop_names]

    return pop_ids


def add_individuals(vcf, sample_data, ploidy_level, populations):
    """
    Add individuals to an existing `SampleData` object using data from an existing `VCF` object.

    Sourced and modified from:
    https://tsinfer.readthedocs.io/en/latest/tutorial.html#data-example

    :param cyvcf2.VCF vcf: A VCF object containing variant data
    :param tsinfer.SampleData sample_data:
    :param int ploidy_level: 1 (haploid) or 2 (diploid)
    :param list populations: Population IDs
    :return: None
    :rtype: None
    """
    assert (
        ploidy_level == 1 or ploidy_level == 2
    ), f"Ploidy {ploidy_level} is not recognized."

    for sample_name, population in zip(vcf.samples, populations):
        sample_data.add_individual(
            ploidy=ploidy_level, metadata={"name": sample_name}, population=population
        )

    return None


def add_sites(
    vcf_iter, sample_data, ploidy_level, *, ancestral_alleles=None, show_warnings=False, help_debug=False
):
    """
    Add sites from a generator over a `VCF` object to an existing `SampleData` object,
    reordering the alleles to put the ancestral allele first, if available.

    If `ancestral_alleles=None`, then the reference allele is taken to be the ancestral allele.

    Sourced and modified from:
    https://tsinfer.readthedocs.io/en/latest/tutorial.html#data-example

    :param generator vcf_iter: Generator from cyvcf2.VCF object.
    :param tsinfer.SampleData sample_data: SampleData object.
    :param int ploidy_level: 1 (haploid) or 2 (diploid).
    :param collections.OrderedDict ancestral_alleles: Index map from old to new alleles (default = None).
    :param bool show_warnings: If True, show warnings (default = False).
    :param bool help_debug: If True, show debugging messages (default = False).
    :return: Number of sites with a matched ancestral allele.
    :rtype: int
    """
    assert (
        ploidy_level == 1 or ploidy_level == 2
    ), f"Ploidy {ploidy_level} is not recognized."

    num_sites_with_aa = 0 # Number of sites with a matched ancestral allele

    pos = 0
    for v in tqdm(vcf_iter):
        assert pos <= v.POS, f"Sites are not coordinate-sorted at {v.POS}"

        if pos == v.POS:
            if show_warnings:
                warnings.warn(f"Duplicate site position at {v.POS}")
            continue
        else:
            pos = v.POS

        if any([not phased for _, _, phased in v.genotypes]):
            raise ValueError(f"Unphased genotypes at {pos}")

        old_allele_list = [v.REF] + v.ALT

        if show_warnings:
            if len(set(old_allele_list) - {"."}) == 1:
                warnings.warn(f"Monomorphic site at {pos}")

        ancestral = v.REF
        if ancestral_alleles is not None:
            chr_pos = str(v.CHROM) + ":" + str(int(pos))
            if chr_pos in ancestral_alleles:
                ancestral = ancestral_alleles[chr_pos]
                num_sites_with_aa += 1

        # Ancestral state must be first in the allele list.
        new_allele_list = [ancestral] + list(set(old_allele_list) - {ancestral})

        # Create an index mapping from VCF (old) to SampleData (new).
        allele_index_map = {
            old_index: new_allele_list.index(allele) # new_index
            for old_index, allele in enumerate(old_allele_list)
        }

        # When genotype is missing...
        if v.num_unknown > 0:
            # cyvcf2 uses -1 to denote missing data.
            # tsinfer uses tskit.MISSING_DATA (-1) to denote missing data.
            allele_index_map[-1] = tskit.MISSING_DATA
            new_allele_list += [None]

        # Map old allele indices to the corresponding new allele indices.
        new_genotypes = [
            allele_index_map[old_index]
            for g in v.genotypes
            for old_index in g[0:ploidy_level]  # (allele 1, allele 2, phased?).
        ]

        if help_debug:
            print(f"VAR___: {v.ID}")
            print(f"MAP___: {allele_index_map}")
            print(f"OLD_AL: {old_allele_list}")
            print(f"NEW_AL: {new_allele_list}")
            print(f"OLD_GT: {v.genotypes}")
            print(f"NEW_GT: {new_genotypes}")

        sample_data.add_site(position=pos, genotypes=new_genotypes, alleles=new_allele_list)

    return num_sites_with_aa


def create_sample_data_from_vcf_file(
    vcf_file, samples_file, ploidy_level, seq_name, *, ancestral_alleles=None, num_threads=1
):
    """
    Create a `SampleData` object from a VCF file and store it in a `.samples` file.

    Sourced and modified from:
    https://tsinfer.readthedocs.io/en/latest/tutorial.html#data-example

    :param str vcf_file: VCF file.
    :param str samples_file: Output .samples file.
    :param int ploidy_level: 1 (haploid) or 2 (diploid).
    :param str seq_name: Sequence name.
    :param collections.OrderedDict ancestral_alleles: Map of ancestral alleles (default = None).
    :param int num_threads: Number of threads to use (default = 1).
    :return: SampleData object containing variants.
    :rtype: tsinfer.SampleData
    """
    # See https://brentp.github.io/cyvcf2/docstrings.html
    # strict_gt (bool) – if True, then any ‘.’ present
    # in a genotype will classify the corresponding element
    # in the gt_types array as UNKNOWN.
    vcf = cyvcf2.VCF(vcf_file, strict_gt=True, threads=num_threads)

    seq_len = get_sequence_length(vcf, seq_name)
    if seq_len is None:
        warnings.warn(
            f"VCF does not contain sequence length for {seq_name}, so setting it to 0."
            "This may cause internal checks to fail."
        )
        seq_len = 0

    vcf_iter = vcf(seq_name)

    with tsinfer.SampleData(
        path=samples_file, sequence_length=seq_len
    ) as sample_data:
        populations = add_populations(vcf, sample_data)
        add_individuals(vcf, sample_data, ploidy_level, populations)
        num_sites_with_aa = add_sites(vcf_iter, sample_data, ploidy_level, ancestral_alleles=ancestral_alleles)

    print(f"DATA: Sites with matched AA {num_sites_with_aa}")

    return sample_data


def extract_ancestral_alleles_from_vcf_file(vcf_file, *, seq_name_prefix=None, num_threads=1, show_warnings=False):
    """
    Extract ancestral alleles from a VCF file.
    Ancestral alleles (AA) should be provided in the INFO field. Note there may be indels.

    :param str vcf_file: VCF file.
    :param str seq_name_prefix: Prefix to prepend to sequence name (default = None).
    :param int num_threads: Number of threads to use (default = 1).
    :param bool show_warnings: If True, then show warnings (default = False).
    :return: Dict mapping site positions to ancestral allele.
    :rtype: collections.OrderedDict
    """
    vcf = cyvcf2.VCF(vcf_file, strict_gt=True, threads=num_threads)

    # Key is site position - (chr, coordinate,)
    # Value is ancestral allele - str
    map_aa = OrderedDict()
    stats = OrderedDict()

    stats["num_sites_total"] = 0
    stats["num_sites_dup"] = 0  # Duplicte site positions
    stats["num_sites_aa"] = 0  # Unique site positions with ancestral allele in INFO

    pos = 0
    for v in tqdm(vcf):
        stats["num_sites_total"] += 1

        assert pos <= v.POS, f"Sites are not coordinate-sorted at {v.POS}"

        if pos == v.POS:
            stats["num_sites_dup"] += 1
            if show_warnings:
                warnings.warn(f"Duplicate site position at {v.POS}")
        else:
            pos = v.POS

        if v.INFO.get("AA"):
            stats["num_sites_aa"] += 1
            chr_pos = str(v.CHROM) + ":" + str(int(pos)) # e.g. 20:123456
            if seq_name_prefix is not None:
                chr_pos = seq_name_prefix + chr_pos
            map_aa[chr_pos] = v.INFO.get("AA")

    return (map_aa, stats)