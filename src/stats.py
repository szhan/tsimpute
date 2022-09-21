from collections import OrderedDict
import csv
import warnings
from tqdm import tqdm
import numpy as np
import cyvcf2


def get_variant_statistics(vcf, *, show_warnings=False):
    """
    Get the following variant statistics from an existing `VCF` object:
        Total number of entries
        Number of duplicate site positions
        Number of multi-allelic sites
        Number of sites by variant type (unique site positions):
            SNVs
            Indels
            SVs
        Number of genotypes by zygosity:
            Hom-ref
            Het
            Hom-alt
        Number of missing / unknown genotypes (>= 1 allele is missing)
        Number of phased genotypes

    :param cyvcf2.VCF vcf: A VCF object.
    :param bool show_warnings: If True, show warnings (default = False).
    :return: A dict of variant statistics.
    :rtype: collections.OrderedDict
    """
    stats = OrderedDict()
    stats["num_entries"] = 0  # Unique and duplicate site positions
    stats["num_site_pos_dup"] = 0
    stats["num_multiallelic_sites"] = 0
    stats["num_snps"] = 0
    stats["num_indels"] = 0
    stats["num_svs"] = 0
    stats["num_others"] = 0  # Non-SNP/indel/SV
    stats["num_hom_ref"] = 0
    stats["num_het"] = 0
    stats["num_hom_alt"] = 0
    stats["num_unknown"] = 0
    stats["num_unphased"] = 0

    pos = 0
    for v in tqdm(vcf):
        assert pos <= v.POS, f"Sites are not coordinate-sorted at {v.POS}"

        stats["num_entries"] += 1

        # Check for duplicate site positions
        if v.POS == pos:
            stats["num_site_pos_dup"] += 1
            if show_warnings:
                warnings.warn(f"Duplicate site position at {v.POS}")
        else:
            pos = v.POS

        # Check for multiallelic sites
        if len(set(v.ALT) - {"."}) > 1:
            stats["num_multiallelic_sites"] += 1

        # Check type of variant
        if v.is_snp:
            stats["num_snps"] += 1
        elif v.is_indel:
            stats["num_indels"] += 1
        elif v.is_sv:
            stats["num_svs"] += 1
        else:
            stats["num_others"] += 1
            if show_warnings:
                warnings.warn(f"Unrecognized type of variant at {v.POS}")

        # Check properties of genotype
        stats["num_hom_ref"] += v.num_hom_ref
        stats["num_het"] += v.num_het
        stats["num_hom_alt"] += v.num_hom_alt
        stats["num_unknown"] += v.num_unknown
        stats["num_unphased"] += (v.num_called + v.num_unknown) - np.sum(v.gt_phases)

    return stats


def print_variant_statistics(stats, csv_file):
    """
    Write variant statistics to a CSV file.

    :param collections.OrderedDict stats: Output from `get_variant_statistics`.
    :param str csv_file: An output CSV file with variant statistics.
    :return: None
    :rtype: None
    """
    with open(csv_file, "w") as f:
        csv_writer = csv.writer(f)
        for k, v in stats.items():
            csv_writer.writerow([k, v])

    return None
