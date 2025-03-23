"""Calculate metrics to assess imputation performance."""
import numpy as np


def compute_concordance(gt_true, gt_imputed, allele_state=None):
    """
    Calculate the total concordance (sometimes referred to as imputation accuracy) between
    `gt_true` and `gt_imputed`.

    Random agreement can inflate total concordance when MAF is low, so interpretation of
    total concordance should be done with caution.

    If `allele_state` is specified, then concordance is calculated based on the elements in
    `gt_true` and `gt_imputed` where `gt_true` is equal to `allele_state`.
    For example, this can be used to calculate non-reference disconcordance.

    This metric may be suitable for sample-wise (per genome) or site-wise (across genomes)
    comparisons of genotypes.

    WARNING: This assumes haploid genomes.

    :param numpy.ndarray gt_true: List of alleles from ground-truth genotypes.
    :param numpy.ndarray gt_imputed: List of alleles from imputed genotypes.
    :param int allele_state: Specify allele state to consider (default = None).
    :return: Total concordance.
    :rtype: float
    """
    assert isinstance(gt_true, np.ndarray), f"Not {np.ndarray}"
    assert isinstance(gt_imputed, np.ndarray), f"Not {np.ndarray}"
    assert len(gt_true) == len(gt_imputed), f"Genotype arrays differ in length."

    if allele_state != None:
        allele_match_bool = np.isin(gt_true, allele_state)
        assert np.any(allele_match_bool)
        gt_true = gt_true[allele_match_bool]
        gt_imputed = gt_imputed[allele_match_bool]

    gt_correct = np.sum(gt_true == gt_imputed)
    gt_total = len(gt_true)

    concordance = float(gt_correct) / float(gt_total)

    return concordance


def compute_concordance_haploid(gt_true, gt_imputed, allele_state=None):
    pass


def compute_concordance_diploid(gt_true, gt_imputed, allele_state=None):
    pass


def compute_iqs(gt_true, gt_imputed, ploidy):
    """
    Calculate the Imputation Quality Score (IQS) as proposed by Lin et al. (2010).
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009697

    Some notes on interpreting IQS:
    1) A value of 1 indicates perfect imputation;
    2) A value of 0 indicates that observed agreement rate is equal to chance agreement rate; and
    3) A negative value indicates that the method imputes poorly than by chance.

    Two formulas are used to compute the IQS of imputed genotypes at biallelic sites,
    one for haploid genomes and the other for diploid genomes.

    :param numpy.ndarray gt_true: A list of alleles from ground-truth genotypes.
    :param numpy.ndarray gt_imputed: A list of alleles from imputed genotypes.
    :ploidy int: Ploidy (1 or 2).
    :return: IQS.
    :rtype: float
    """
    assert ploidy in [1, 2], f"Ploidy {ploidy} is invalid."

    assert np.all(
        np.isin(np.unique(gt_true), [0, 1])
    ), f"Not all elements are 0 or 1 - {np.unique(gt_true)}."
    assert np.all(
        np.isin(np.unique(gt_imputed), [0, 1])
    ), f"Not all elements are 0 or 1 - {np.unique(gt_imputed)}."

    if ploidy == 1:
        iqs = compute_iqs_haploid(gt_true, gt_imputed)
    else:
        iqs = compute_iqs_diploid(gt_true, gt_imputed)

    return iqs


def compute_iqs_haploid(gt_true, gt_imputed):
    """
    Calculate the IQS between `gt_true` and `gt_imputed`.

    This specific formula is used to compute the IQS of imputed genotypes
    at biallelic sites in HAPLOID genomes.

    :param numpy.ndarray gt_true: A list of alleles from ground-truth genotypes.
    :param numpy.ndarray gt_imputed: A list of alleles from imputed genotypes.
    :return: IQS.
    :rtype: float
    """
    assert len(gt_true) == len(gt_imputed), f"Genotype arrays differ in size."

    # Allele 0 imputed correctly as allele 0
    n00 = np.sum([y == 0 for x, y in zip(gt_imputed, gt_true) if x == 0])
    # Allele 0 imputed wrongly as allele 1
    n01 = np.sum([y == 1 for x, y in zip(gt_imputed, gt_true) if x == 0])
    # Allele 1 imputed correctly as allele 1
    n11 = np.sum([y == 1 for x, y in zip(gt_imputed, gt_true) if x == 1])
    # Allele 1 imputed wrongly as allele 0
    n10 = np.sum([y == 0 for x, y in zip(gt_imputed, gt_true) if x == 1])

    # Marginal counts
    n0_ = n00 + n01
    n1_ = n10 + n11
    n_0 = n00 + n10
    n_1 = n01 + n11

    assert n0_ + n1_ == n_0 + n_1, f"Marginal counts do not add up."

    # Total count
    n__ = n00 + n10 + n01 + n11

    # Observed agreement (i.e. overall concordance)
    Po = float(n00 + n11) / float(n__)

    # Chance agreement
    Pc = float(n0_ * n_0 + n1_ * n_1) / float(n__**2)

    assert 0 <= Po <= 1, f"Po {Po} is not a proportion."
    assert 0 <= Pc <= 1, f"Pc {Pc} is not a proportion."

    iqs = float("nan") if Pc == 1 else (Po - Pc) / (1 - Pc)

    return iqs


def compute_iqs_diploid(gt_true, gt_imputed):
    """
    Calculate the IQS between `gt_true` and `gt_imputed`.

    This specific formula is used to compute the IQS of imputed genotypes
    at biallelic sites in DIPLOID genomes.

    TODO: Generalize to handle multiallelic sites.

    :param numpy.ndarray gt_true: A list of alleles from ground-truth genotypes.
    :param numpy.ndarray gt_imputed: A list of alleles from imputed genotypes.
    :return: IQS.
    :rtype: float
    """
    assert len(gt_true) == len(gt_imputed), f"Genotype arrays differ in size."
    assert len(gt_true) % 2 == 0, f"Not all genotypes are diploid."

    AA = [0, 0]  # shorthand, 1
    AB = [0, 1]  # shorthand, 2
    BA = [1, 0]  # shorthand, 3
    BB = [1, 1]  # shorthand, 4
    _POSSIBLE_GT_ = [AA, AB, BA, BB]
    num_possible_gt = len(_POSSIBLE_GT_)

    num_individuals = int(len(gt_true) / 2)
    gt_true_reshaped = np.reshape(gt_true, (num_individuals, 2))
    gt_imputed_reshaped = np.reshape(gt_imputed, (num_individuals, 2))

    # Ancestral allele is denoted by A, and derived allele by B.
    counts = np.empty(num_possible_gt**2)
    for i, gt_i in enumerate(_POSSIBLE_GT_):
        for j, gt_j in enumerate(_POSSIBLE_GT_):
            k = i * num_possible_gt + j
            counts[k] = np.sum(
                np.equal(gt_true_reshaped, gt_i).all(axis=1)
                & np.equal(gt_imputed_reshaped, gt_j).all(axis=1)
            )

    counts = np.reshape(
        counts,
        (
            num_possible_gt,
            num_possible_gt,
        ),
    )

    # Total count
    n_t = np.sum(counts)

    # Marginal counts
    n_c = np.sum(counts, axis=0)
    n_r = np.sum(counts, axis=1)
    assert np.sum(n_c) == np.sum(n_r), f"Marginal counts do not add up."

    # Observed agreement (i.e. overall concordance)
    Po = float(np.sum(counts.diagonal())) / float(n_t)

    # Chance agreement
    Pc = float(n_c.dot(n_r)) / float(n_t**2)

    assert 0 <= Po <= 1, f"Po {Po} is not a proportion."
    assert 0 <= Pc <= 1, f"Pc {Pc} is not a proportion."

    iqs = float("nan") if Pc == 1 else (Po - Pc) / (1 - Pc)

    return iqs


def computed_r_squared(genotypes_true, genotypes_imputed):
    """
    Calculate the squared correlation coefficient between `genotypes_true` and `genotypes_imputed`.

    :param numpy.ndarray genotypes_true: List of alleles from ground-truth genotypes.
    :param numpy.ndarray genotypes_imputed: List of alleles from imputed genotypes.
    :return: R-squared correlation coefficient.
    :rtype: float
    """
    r_squared = None
    return r_squared


""" Metrics used by BEAGLE. """


# Individual-level
def compute_individual_scores(
    alleles_1, allele_probs_1, alleles_2, allele_probs_2, ref
):
    """
    Compute genotype probabilities and allele dosages of diploid individuals
    at a position based on posterior marginal allele probabilities.

    Assume that all sites are biallelic. Otherwise, the calculation below is incorrect.
    Note 0 refers to the REF allele and 1 the ALT allele.

    Unphased genotype (or dosage) probabilities are: P(RR), P(RA or AR), P(AA).
    Dosages of the ALT allele are: RR = 0, RA or AR = 1, AA = 2.

    In BEAGLE 4.1 output,
    GP: "Estimated Genotype Probability", and
    DS: "Estimated ALT dose [P(RA) + P(AA)]".

    :param numpy.ndarray alleles_1: Imputed alleles for haplotype 1.
    :param numpy.ndarray allele_probs_1: Imputed allele probabilities for haplotype 1.
    :param numpy.ndarray alleles_2: Imputed alleles for haplotype 2.
    :param numpy.ndarray allele_probs_2: Imputed allele probabilities for haplotype 2.
    :param int ref: Specified REF allele (ACGT encoding).
    :return: Dosage probabilities and dosage scores.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    n = len(alleles_1)  # Number of individuals.
    assert len(alleles_2) == n, "Lengths of alleles differ."
    assert n > 0, "There must be at least one individual."
    assert len(allele_probs_1) == n, "Lengths of alleles and probabilities differ."
    assert len(allele_probs_2) == n, "Lengths of alleles and probabilities differ."
    dosage_probs = np.zeros((n, 3), dtype=np.float64)
    dosage_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ap_hap1_ref = (
            allele_probs_1[i] if alleles_1[i] == ref else 1 - allele_probs_1[i]
        )
        ap_hap1_alt = 1 - ap_hap1_ref
        ap_hap2_ref = (
            allele_probs_2[i] if alleles_2[i] == ref else 1 - allele_probs_2[i]
        )
        ap_hap2_alt = 1 - ap_hap2_ref
        dosage_probs[i, 0] = ap_hap1_ref * ap_hap2_ref  # P(RR)
        dosage_probs[i, 1] = ap_hap1_ref * ap_hap2_alt  # P(RA)
        dosage_probs[i, 1] += ap_hap1_alt * ap_hap2_ref  # P(AR)
        dosage_probs[i, 2] = ap_hap1_alt * ap_hap2_alt  # P(AA)
        dosage_scores[i] = dosage_probs[i, 1] + 2 * dosage_probs[i, 2]
    return (dosage_probs, dosage_scores)


# Site-level
def compute_allelic_r_squared(dosage_probs):
    """
    Compute the estimated allelic R^2 at a position from the unphased genotype
    (or dosage) probabilities of a set of diploid individuals.

    Assume that site is biallelic. Otherwise, the calculation below is incorrect.
    Note that 0 refers to REF allele and 1 the ALT allele.

    It is not the true allelic R^2, which needs access to true genotypes to compute.
    The true allelic R^s is the squared correlation between true and imputed ALT dosages.
    It has been shown the true and estimated allelic R-squared are highly correlated.

    In BEAGLE 4.1, it is AR2: "Allelic R-Squared: estimated squared correlation
    between most probable REF dose and true REF dose".
    See `allelicR2` in `R2Estimator.java` of the BEAGLE 4.1 source code.

    See the formulation in the Appendix 1 of Browning & Browning (2009).
    Am J Hum Genet. 84(2): 210–223. doi: 10.1016/j.ajhg.2009.01.005.

    :return: Dosage probabilities and dosage scores.
    :return: Estimated allelic R-squared.
    :rtype: float
    """
    _MIN_R2_DEN = 1e-8
    n = len(dosage_probs)  # Number of individuals.
    assert n > 0, "There must be at least one individual."
    assert dosage_probs.shape[1] == 3, "Three genotypes are considered."
    f = 1 / n
    z = np.argmax(dosage_probs, axis=1)  # Most likely imputed dosage.
    u = dosage_probs[:, 1] + 2 * dosage_probs[:, 2]  # E[X | y_i]
    w = dosage_probs[:, 1] + 4 * dosage_probs[:, 2]  # E[X^2 | y_i]
    cov = np.sum(z * u) - np.sum(z) * np.sum(u) * f
    var_best = np.sum(z**2) - np.sum(z) ** 2 * f
    var_exp = np.sum(w) - np.sum(u) ** 2 * f
    den = var_best * var_exp
    # Minimum of allelic R^2 is zero.
    allelic_rsq = 0 if den < _MIN_R2_DEN else cov**2 / den
    return allelic_rsq


def compute_dosage_r_squared(dosage_probs):
    """
    Compute the dosage R^2 for a position from the unphased genotype (or dosage)
    probabilities of a set of diploid individuals.

    Assume that site is biallelic. Otherwise, the calculation below is incorrect.
    Note that 0 refers to REF allele and 1 the ALT allele.

    In BEAGLE 4.1, DR2: "Dosage R-Squared: estimated squared correlation
    between estimated REF dose [P(RA) + 2 * P(RR)] and true REF dose".
    See `doseR2` in `R2Estimator.java` of the BEAGLE 4.1 source code.

    :return: Dosage probabilities and dosage scores.
    :return: Dosage R-squared.
    :rtype: float
    """
    _MIN_R2_DEN = 1e-8
    n = len(dosage_probs)  # Number of individuals.
    assert n > 0, "There must be at least one individual."
    assert dosage_probs.shape[1] == 3, "Three genotypes are considered."
    f = 1 / n
    u = dosage_probs[:, 1] + 2 * dosage_probs[:, 2]  # E[X | y_i].
    w = dosage_probs[:, 1] + 4 * dosage_probs[:, 2]  # E[X^2 | y_i].
    c = np.sum(u) ** 2 * f
    num = np.sum(u**2) - c
    if num < 0:
        num = 0
    den = np.sum(w) - c
    dosage_rsq = 0 if den < _MIN_R2_DEN else num / den
    return dosage_rsq


def compute_allele_frequency(
    alleles_1,
    allele_probs_1,
    alleles_2,
    allele_probs_2,
    allele,
):
    """
    Estimate the frequency of a specified allele at a position from allele probabilities
    of a set of diploid individuals.

    Assume that site is biallelic. Otherwise, the calculation below is incorrect.

    Input are the imputed alleles and their probabilities at a position.

    In BEAGLE 4.1, AF: "Estimated ALT Allele Frequencies".
    See `printInfo` in `VcfRecBuilder.java` of the BEAGLE 4.1 source code.

    See the note in "Standardized Allele-Frequency Error" in Browning & Browning (2009).
    Am J Hum Genet. 84(2): 210–223. doi: 10.1016/j.ajhg.2009.01.005.

    :param numpy.ndarray alleles_1: Imputed alleles for haplotype 1.
    :param numpy.ndarray allele_probs_1: Imputed allele probabilities for haplotype 1.
    :param numpy.ndarray alleles_2: Imputed alleles for haplotype 2.
    :param numpy.ndarray allele_probs_2: Imputed allele probabilities for haplotype 2.
    :param int allele: Specified allele (ACGT encoding).
    :return: Estimated allele frequency.
    :rtype: float
    """
    n = len(alleles_1)  # Number of individuals.
    assert len(alleles_2) == n, "Lengths of alleles differ."
    assert n > 0, "There must be at least one individual."
    assert len(allele_probs_1) == n, "Lengths of alleles and probabilities differ."
    assert len(allele_probs_2) == n, "Lengths of alleles and probabilities differ."
    cum_ap_hap1 = np.sum(allele_probs_1[alleles_1 == allele])
    cum_ap_hap2 = np.sum(allele_probs_2[alleles_2 == allele])
    # See `printInfo` in `VcfRecBuilder.java` in BEAGLE 4.1 source code.
    est_af = (cum_ap_hap1 + cum_ap_hap2) / (2 * n)
    return est_af
