import tskit
import tsinfer

import numpy as np

import cyvcf2


def print_sample_data_to_vcf(
    sample_data,
    individuals,
    samples,
    ploidy_level,
    mask,
    out_vcf_file,
    contig_id,
    sequence_length_max=1e12
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
    """
    CHROM = contig_id
    ID = '.'
    QUAL = '.'
    FILTER = 'PASS'
    FORMAT = 'GT'
    
    assert ploidy_level == 1 or ploidy_level == 2,\
        f"Specified ploidy_level {ploidy_level} is not recognized."
    
    assert ploidy_level * len(individuals) == len(samples),\
        f"Some individuals may not have the same ploidy level of {ploidy_level}."
    
    # Assume that both sample and individual ids are ordered the same way.
    #individual_id_map = np.repeat(individuals, 2)
    
    header  = "##fileformat=VCFv4.2\n"\
            + "##source=tskit " + tskit.__version__ + "\n"\
            + "##INFO=<ID=AA,Number=1,Type=String,Description=\"Ancestral Allele\">\n"\
            + "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
    header += "##contig=<ID=" + contig_id + "," + "length=" + str(int(sample_data.sequence_length)) + ">\n"
    header += "\t".join(
        ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']\
        + ["s" + str(x) for x in individuals]
        )
    
    with open(out_vcf_file, "w") as vcf:
        vcf.write(header + "\n")
        for i, variant in enumerate(sample_data.variants()):
            site_id = variant.site.id
            POS = int(np.round(variant.site.position))
            if POS > sequence_length_max:
                break
            # Since the tree sequence was produced using simulation,
            #    there's no reference sequence other than the ancestral sequence.
            REF = variant.site.ancestral_state
            alt_alleles = list(set(variant.alleles) - {REF})
            AA = variant.site.ancestral_state
            ALT = ",".join(alt_alleles) if len(alt_alleles) > 0 else "."
            INFO = "AA" + "=" + AA
            record = [str(x)
                      for x
                      in [CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT]]
            
            for j in individuals:
                #sample_ids = [samples[x]
                #              for x
                #              in np.where(individual_id_map == j)[0].tolist()]
                #genotype = "|".join([str(variant.genotypes[k])
                #                     for k
                #                     in sample_ids])
                if ploidy_level == 1:
                    genotype = str(variant.genotypes[j])
                else:
                    genotype = str(variant.genotypes[2 * j]) + "|" + str(variant.genotypes[2 * j + 1])
                    
                if mask is not None and mask.query_position(individual = j, position = POS) == True:
                    if ploidy_level == 1:
                        genotype = '.'
                    else:
                        genotype = '.|.' # Or "./."
                record += [genotype]
                
            vcf.write("\t".join(record) + "\n")


# Sourced and modified from:
# https://tsinfer.readthedocs.io/en/latest/tutorial.html#data-example
def get_chromosome_length(vcf):
    assert len(vcf.seqlens) == 1
    return vcf.seqlens[0]


def add_populations(
    vcf,
    samples
):
    """
    TODO
    """
    pop_ids = [sample_name[0] for sample_name in vcf.samples]
    pop_codes = np.unique(pop_ids)
    pop_lookup = {}
    for p in pop_codes:
        pop_lookup[p] = samples.add_population(metadata = {"name" : p})
    return [pop_lookup[pop_id] for pop_id in pop_ids]


def add_individuals(
    vcf,
    samples,
    ploidy_level,
    populations
):
    for name, population in zip(vcf.samples, populations):
        samples.add_individual(
            ploidy = ploidy_level,
            metadata = {"name": name},
            population = population
        )


def add_sites(
    vcf,
    samples,
    ploidy_level,
    warn_monomorphic_sites=False
):
    """
    Read the sites in the VCF and add them to the samples object,
    reordering the alleles to put the ancestral allele first,
    if it is available.
    """
    assert ploidy_level == 1 or ploidy_level == 2,\
        f"ploidy_level {ploidy_level} is not recognized."
    
    pos = 0
    for variant in vcf:
        # Check for duplicate site positions.
        if pos == variant.POS:
            raise ValueError("Duplicate positions for variant at position", pos)
        else:
            pos = variant.POS
        # Check that the genotypes are phased.
        #if any([not phased for _, _, phased in variant.genotypes]):
        #    raise ValueError("Unphased genotypes for variant at position", pos)
        alleles = [variant.REF] + variant.ALT # Exactly as in the input VCF file.
        if warn_monomorphic_sites:
            if len(set(alleles) - {'.'}) == 1:
                print(f"Monomorphic site at {pos}")

        ancestral = variant.INFO.get("AA", variant.REF) # Dangerous action!!!

        # Ancestral state must be first in the allele list.
        ordered_alleles = [ancestral] + list(set(alleles) - {ancestral})

        # Create an index mapping from the input VCF to tsinfer input.
        allele_index = {
            old_index: ordered_alleles.index(allele)
            for old_index, allele in enumerate(alleles)
        }

        # When genotype is missing...
        if variant.num_unknown > 0:
            allele_index[-1] = tskit.MISSING_DATA
            ordered_alleles += [None]

        # Map original allele indexes to their indexes in the new alleles list.
        genotypes = [
            allele_index[old_index]
            for row in variant.genotypes # cyvcf2 uses -1 to indicate missing data.
            for old_index in row[0:ploidy_level] # Each is a 3-tuple (allele 1, allele 2, is phased?).
        ]

        samples.add_site(pos,
                         genotypes = genotypes,
                         alleles = ordered_alleles)


def create_sample_data_from_vcf_file(vcf_file):
    vcf = cyvcf2.VCF(
        vcf_file,
        gts012=False, # 0=HOM_REF, 1=HET, 2=UNKNOWN, 3=HOM_ALT
        strict_gt=True
    )

    with tsinfer.SampleData(
        sequence_length = get_chromosome_length(vcf)
    ) as samples:
        populations = add_populations(vcf, samples)
        add_individuals(vcf, samples, ploidy_level, populations)
        add_sites(vcf, samples, ploidy_level)

    return(samples)