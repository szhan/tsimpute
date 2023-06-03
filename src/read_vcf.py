from tqdm import tqdm
import numpy as np
import tskit


def print_samples_to_vcf(
    sd,
    ploidy,
    out_file,
    site_mask=None,
    exclude_mask_sites=False,
    exclude_monoallelic_sites=False,
    contig_id=None,
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
    :param array_like site_mask: Site positions to mask (1-based).
    :param bool exclude_mask_sites: Exclude masked sites.
    :param bool exclude_monoallelic_sites: Exclude monoallelic sites.
    :param str contig_id: Contig name.
    :param click.Path out_file: Path to output VCF file.
    """
    CHROM = contig_id
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
        "##contig=<ID=" + contig_id + "," + \
        "length=" + str(int(sd.sequence_length)) + ">\n"
    )
    header += "\t".join(
        ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        + [str(x.metadata["name"]) for x in sd.individuals()]
    )

    with open(out_file, "w") as f:
        f.write(header + "\n")
        for v in sd.variants():
            # Site positions are stored as float in tskit
            POS = int(v.site.position)
            POS += 1  # VCF is 1-based, but tskit is 0-based.
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
