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


def get_sequence_length(vcf):
    """
    Helper function

    Sourced and modified from:
    https://tsinfer.readthedocs.io/en/latest/tutorial.html#data-example
    """
    assert len(vcf.seqlens) == 1
    return vcf.seqlens[0]


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


def add_sites(vcf, sample_data, ploidy_level, show_warnings=False):
    """
    Read the sites from an existing `VCF` object, and add them to an existing `SampleData` object,
    reordering the alleles to put the ancestral allele first, if it is available.

    Sourced and modified from:
    https://tsinfer.readthedocs.io/en/latest/tutorial.html#data-example

    :param cyvcf2.VCF vcf: A VCF object containing variant data
    :param tsinfer.SampleData sample_data:
    :param int ploidy_level: 1 (haploid) or 2 (diploid).
    :param bool show_warnings:
    :return: None
    :rtype: None
    """
    assert (
        ploidy_level == 1 or ploidy_level == 2
    ), f"Ploidy {ploidy_level} is not recognized."

    pos = 0
    for v in tqdm(vcf):
        assert pos <= v.POS, f"Sites are not coordinate-sorted at {v.POS}"

        if pos == v.POS:
            if show_warnings:
                warnings.warn(f"Duplicate site position at {v.POS}")
            continue
        else:
            pos = v.POS

        if any([not phased for _, _, phased in v.genotypes]):
            raise ValueError(f"Unphased genotypes at {pos}")

        alleles = [v.REF] + v.ALT

        if show_warnings:
            if len(set(alleles) - {"."}) == 1:
                warnings.warn(f"Monomorphic site at {pos}")

        # Dangerous action!!!
        # TODO: Provide ancestral alleles in a separate file.
        ancestral = v.INFO.get("AA", v.REF)

        # Ancestral state must be first in the allele list.
        ordered_alleles = [ancestral] + list(set(alleles) - {ancestral})

        # Create an index mapping from the input VCF to tsinfer input.
        allele_index = {
            old_index: ordered_alleles.index(allele)
            for old_index, allele in enumerate(alleles)
        }

        # When genotype is missing...
        if v.num_unknown > 0:
            allele_index[-1] = tskit.MISSING_DATA
            ordered_alleles += [None]

        # Map original allele indexes to their indexes in the new alleles list.
        genotypes = [
            allele_index[old_index]
            for row in v.genotypes  # cyvcf2 uses -1 to denote missing data.
            for old_index in row[
                0:ploidy_level
            ]  # 3-tuple (allele 1, allele 2, is phased?).
        ]

        sample_data.add_site(position=pos, genotypes=genotypes, alleles=ordered_alleles)

    return None


def create_sample_data_from_vcf_file(vcf_file, ploidy_level, samples_file):
    """
    Create a `SampleData` object from a VCF file and store it in a `.samples` file.

    Sourced and modified from:
    https://tsinfer.readthedocs.io/en/latest/tutorial.html#data-example

    :param str vcf_file: An input VCF file.
    :param int ploidy_level: 1 (haploid) or 2 (diploid).
    :return: A SampleData object containing variants from the VCF file.
    :rtype: tsinfer.SampleData
    """
    vcf = cyvcf2.VCF(vcf_file, strict_gt=True)

    try:
        sequence_length = get_sequence_length(vcf)
    except:
        warnings.warn(
            "VCF does not contain sequence length. Setting sequence length to 0."
        )
        sequence_length = 0

    with tsinfer.SampleData(
        path=samples_file, sequence_length=sequence_length
    ) as sample_data:
        populations = add_populations(vcf, sample_data)
        add_individuals(vcf, sample_data, ploidy_level, populations)
        add_sites(vcf, sample_data, ploidy_level)

    return sample_data
