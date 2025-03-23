from dataclasses import dataclass
import numpy as np
import tskit


__DATE__ = "TODO"
__VERSION__ = "TODO"


# Processing of results.
TUPLE_ACGT_ALLELES = (0, 1, 2, 3, tskit.MISSING_DATA)

@dataclass(frozen=True)
class ImpData:
    """
    Imputation data containing:
    * Individual names.
    * Positions of the imputed sites (bp).
    * Designated REF allele at each imputed site.
    * Designated ALT allele at each imputed site.
    * Imputed alleles at each imputed site.
    * Imputed allele probabilities at each imputed site.

    Assume that all the imputed sites are biallelic.

    Let x = number of imputed sites and q = number of query haplotypes.
    Since the query haplotypes are from diploid individuals, q is equal to
    twice the number of individuals.

    Imputed alleles is a matrix of size (q, x).
    Imputed allele probabilities is a matrix of size (q, x).
    """

    individual_names: list
    site_pos: np.ndarray
    refs: np.ndarray
    alts: np.ndarray
    alleles: np.ndarray
    allele_probs: np.ndarray

    def __post_init__(self):
        if len(self.individual_names) <= 0:
            raise ValueError("There must be at least one individual.")
        if len(self.site_pos) <= 0:
            raise ValueError("There must be at least one site.")
        if self.alleles.shape[0] / 2 != len(self.individual_names):
            raise ValueError("Unexpected number of query haplotypes.")
        if len(self.site_pos) != len(self.refs):
            raise ValueError("Unexpeced number of ref. alleles.")
        if len(self.site_pos) != len(self.alts):
            raise ValueError("Unexpeced number of alt. alleles.")
        if len(self.site_pos) != self.alleles.shape[1]:
            raise ValueError("Number of sites in alleles != number of site positions.")
        if self.alleles.shape != self.allele_probs.shape:
            raise ValueError("Incompatible alleles and allele probabilities.")
        for i in range(self.alleles.shape[1]):
            if ~np.all(np.isin(self.alleles[:, i], [self.refs[i], self.alts[i]])):
                raise ValueError("TODO")
        if ~np.all(np.isin(np.unique(self.refs), TUPLE_ACGT_ALLELES)):
            raise ValueError("Unrecognized alleles in REF.")
        if ~np.all(np.isin(np.unique(self.alts), TUPLE_ACGT_ALLELES)):
            raise ValueError("Unrecognized alleles in ALT.")
        if ~np.all(np.isin(np.unique(self.alleles), TUPLE_ACGT_ALLELES)):
            raise ValueError("Unrecognized alleles in alleles.")
        if np.array_equal(self.refs, self.alts):
            raise ValueError("Some REFs are identical to ALTs.")

    @property
    def num_sites(self):
        return len(self.site_pos)

    @property
    def num_samples(self):
        return self.alleles.shape[0]

    @property
    def num_individuals(self):
        return len(self.individual_names)

    def get_ref_allele_at_site(self, i):
        return self.refs[i]

    def get_alt_allele_at_site(self, i):
        return self.alts[i]

    def get_alleles_at_site(self, i):
        idx_hap1 = np.arange(0, self.num_samples, 2)
        idx_hap2 = np.arange(1, self.num_samples, 2)
        a1 = self.alleles[idx_hap1, i]
        ap1 = self.allele_probs[idx_hap1, i]
        a2 = self.alleles[idx_hap2, i]
        ap2 = self.allele_probs[idx_hap2, i]
        return a1, ap1, a2, ap2


def write_vcf(impdata, out_file, *, chr_name="1", print_gp=False, decimals=2):
    """
    Print imputation results in VCF format, following the output of BEAGLE 4.1.

    TODO: Print VCF records for genotyped sites.

    :param ImpData impdata: Object containing imputation data.
    :param str out_file: Path to output VCF file.
    :param str chr_name: Chromosome name (default = "1").
    :param bool print_gp: Print genotype probabilities if True (default = False).
    :param int decimals: Number of decimal places to print (default = 2).
    :return: None
    :rtype: None
    """
    _HEADER = [
        "##fileformat=VCFv4.2",
        f"##filedata={__DATE__}",
        f"##source=tsimpute (version {__VERSION__})",
        "##INFO=<ID=AF,Number=A,Type=Float,"
        + 'Description="Estimated ALT Allele Frequencies">',
        "##INFO=<ID=AR2,Number=1,Type=Float,"
        + 'Description="Allelic R-Squared: estimated squared correlation '
        + 'between most probable REF dose and true REF dose">',
        "##INFO=<ID=DR2,Number=1,Type=Float,"
        + 'Description="Dosage R-Squared: estimated squared correlation '
        + 'between estimated REF dose [P(RA) + 2*P(RR)] and true REF dose">',
        '##INFO=<ID=IMP,Number=0,Type=Flag,Description="Imputed marker">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "##FORMAT=<ID=DS,Number=A,Type=Float,"
        + 'Description="estimated ALT dose [P(RA) + P(AA)]">',
        "##FORMAT=<ID=GL,Number=G,Type=Float,"
        + 'Description="Log10-scaled Genotype Likelihood">',
        "##FORMAT=<ID=GP,Number=G,Type=Float,"
        + 'Description="Estimated Genotype Probability">',
    ]
    _COL_NAMES = [
        "CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        "FORMAT",
    ]
    with open(out_file, "w") as f:
        # Add header with metadata and definitions.
        for line in _HEADER:
            f.write(line + "\n")
        # Add column names.
        col_str = "#"
        col_str += "\t".join(_COL_NAMES)
        col_str += "\t"
        col_str += "\t".join(impdata.individual_names)
        f.write(col_str + "\n")
        # Add VCF records.
        is_imputed = True
        for i in range(impdata.num_sites):
            a1, ap1, a2, ap2 = impdata.get_alleles_at_site(i)
            gt_probs, dosages = compute_individual_scores(a1, ap1, a2, ap2)
            line_str = chr_name
            line_str += "\t"
            line_str += str(int(impdata.site_pos[i]))
            line_str += "\t"
            line_str += str(i)
            line_str += "\t"
            REF = impdata.get_ref_allele_at_site(i)
            ALT = impdata.get_alt_allele_at_site(i)
            line_str += tskit.ALLELES_ACGT[REF]
            line_str += "\t"
            line_str += tskit.ALLELES_ACGT[ALT]
            line_str += "\t"
            # QUAL field
            # '.' denotes missing.
            line_str += "."
            line_str += "\t"
            # FILTER field
            line_str += "PASS"
            line_str += "\t"
            # INFO field
            ar2 = compute_allelic_r_squared(gt_probs)
            dr2 = compute_dosage_r_squared(gt_probs)
            af = compute_allele_frequency(a1, ap1, a2, ap2, allele=1)
            ar2 = round(ar2, decimals)
            dr2 = round(dr2, decimals)
            af = round(af, decimals)
            info_str = f"AR2={ar2};DR2={dr2};AF={af}"
            if is_imputed:
                info_str += ";" + "IMP"
            line_str += info_str
            line_str += "\t"
            # FORMAT field
            line_str += "GT:DS"
            if print_gp:
                line_str += ":" + "GP"
            line_str += "\t"
            # DATA fields
            data_str = ""
            for j in range(impdata.num_individuals):
                gt_a1 = "0" if a1[j] == REF else "1"
                gt_a2 = "0" if a2[j] == REF else "1"
                data_str += gt_a1 + "|" + gt_a2 + ":"
                data_str += str(round(dosages[j], decimals))
                if print_gp:
                    data_str += ":"
                    data_str += ",".join(
                        [str(round(gt_probs[j, k], decimals)) for k in range(3)]
                    )
                if j < impdata.num_individuals - 1:
                    data_str += "\t"
            line_str += data_str
            f.write(line_str + "\n")
