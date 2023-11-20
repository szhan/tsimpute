""" Functions for writing and reading contents. """
import json
import tqdm

import numpy as np

import tskit
import tsinfer


def print_tsdata_to_vcf(
    tsdata,
    contig_name,
    out_prefix,
    site_mask=None,
    exclude_mask_sites=False,
    exclude_monoallelic_sites=False,
):
    """
    Print the contents of a `SampleData` or `TreeSequence` object in VCF 4.2.

    Assume that:
    1. Individuals are diploid.
    2. Site positions are discrete.

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

    Parameters `site_mask` and `exclude_mask_sites` interact with each other.
    Site positions in `site_mask` are either printed as '.|.' (i.e. phased missing data)
    (if `exclude_mask_sites` is set to False) or excluded from the output file
    (if `exclude_mask_sites` is set to True).

    if `exclude_monoallelic_sites` is set to True, then invariant sites are excluded
    from the output file.

    :param tskit.TreeSequence/tsinfer.SampleData tsdata: Tree sequence or sample data.
    :param str contig_name: Contig name.
    :param str out_prefix: Output file prefix (*.vcf).
    :param array_like site_mask: Site positions to mask (default = None).
    :param bool exclude_mask_sites: Exclude masked sites (default = None).
    :param bool exclude_monoallelic_sites: Exclude monoallelic sites (default = None).
    """
    CHROM = contig_name
    ID = "."
    QUAL = "."
    FILTER = "PASS"
    FORMAT = "GT"

    if isinstance(tsdata, tsinfer.SampleData):
        individual_names = [x.metadata["name"] for x in tsdata.individuals()]
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
        ["#" + "CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        + individual_names
    )

    out_file = out_prefix + ".vcf"
    with open(out_file, "w") as f:
        f.write(header + "\n")
        for var in tqdm.tqdm(tsdata.variants(), total=tsdata.num_sites):
            # Site positions are stored as float in tskit.
            # WARN: This is totally wrong if the site positions are not discrete.
            POS = int(var.site.position)
            # If ts was simulated, there's no ref. sequence besides the ancestral sequence.
            REF = var.site.ancestral_state
            alt_alleles = list(set(v.alleles) - {REF} - {None})
            AA = var.site.ancestral_state
            ALT = ",".join(alt_alleles) if len(alt_alleles) > 0 else "."
            INFO = "AA" + "=" + AA
            record = np.array([CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT], dtype=str)
            if exclude_monoallelic_sites:
                if len(np.unique(var.genotypes)) == 1:
                    continue
            if site_mask is not None and POS in site_mask:
                if exclude_mask_sites:
                    continue
                gt = np.repeat('.|.', tsdata.num_individuals)
            else:
                gt = var.genotypes.astype(str)
                a1 = gt[np.arange(0, tsdata.num_samples, 2)]
                a2 = gt[np.arange(1, tsdata.num_samples, 2)]
                gt = np.char.join('|', np.char.add(a1, a2))
            f.write("\t".join(np.concatenate([record, gt])) + "\n")
