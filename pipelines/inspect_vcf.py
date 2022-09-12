from collections import OrderedDict
import csv
import warnings
import click
from tqdm import tqdm
import numpy as np
import cyvcf2


def get_variant_statistics(vcf_file, left_coordinate, right_coordinate, show_warnings=False):
    """
    Get the following variant statistics from a VCF file:
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

    :param str vcf_file: A VCF file to parse.
    :param int left_coordinate: 0-based left coordinate of the inclusion interval.
    :param int right_coordinate: 0-based right coordinate of the inclusion interval.
    :param bool show_warnings: If True, show warnings (default = False).
    :return: A dict of variant statistics.
    :rtype: collections.OrderedDict
    """
    stats = OrderedDict()
    stats["vcf_file"] = vcf_file
    stats["left_coordinate"] = left_coordinate
    stats["right_coordinate"] = right_coordinate
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

    vcf = cyvcf2.VCF(vcf_file, strict_gt=True)
    pos = 0
    for v in tqdm(vcf):
        assert pos <= v.POS, f"Sites are not sorted by coordinate starting at {v.POS}"

        if left_coordinate is not None:
            if v.POS < left_coordinate:
                continue

        if right_coordinate is not None:
            # Assuming oordinate-sorted entries
            if v.POS > right_coordinate:
                break

        stats["num_entries"] += 1
        # Check for duplicate site positions
        if v.POS == pos:
            stats["num_site_pos_dup"] += 1
            if verbose:
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
            if verbose:
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
    Dump variant statistics into a CSV file.

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


@click.command()
@click.option(
    "--in_vcf_file",
    "-i",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
    help="Input (gzipped) VCF file",
)
@click.option(
    "--out_csv_file",
    "-o",
    type=click.Path(file_okay=True, writable=True),
    required=True,
    help="Output CSV file with variant statistics",
)
@click.option(
    "--left_coordinate",
    "-l",
    type=int,
    default=None,
    help="Left 0-based coordinate of the inclusion interval (default = None)."
    + "If None, then set to 0.",
)
@click.option(
    "--right_coordinate",
    "-r",
    type=int,
    default=None,
    help="Right 0-based coordinate of the inclusion interval (default = None)."
    + "If None, then set to the last coordinate in the VCF file.",
)
@click.option("--verbose", "-v", is_flag=True, help="Show warnings")
def parse_vcf_file(
    in_vcf_file, out_csv_file, left_coordinate, right_coordinate, verbose
):
    stats = get_variant_statistics(
        in_vcf_file, left_coordinate, right_coordinate, verbose
    )
    print_variant_statistics(stats, out_csv_file)


if __name__ == "__main__":
    parse_vcf_file()
