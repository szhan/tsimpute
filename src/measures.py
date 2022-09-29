"""
Calculate various metrics to assess imputation performance.
"""
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

    :param np.ndarray gt_true: List of alleles from ground-truth genotypes.
    :param np.ndarray gt_imputed: List of alleles from imputed genotypes.
    :param int allele_state: Specify allele state to consider (default = None).
    :return: Total concordance.
    :rtype: float
    """
    assert isinstance(gt_true, np.ndarray), f"Not numpy.ndarray"
    assert isinstance(gt_imputed, np.ndarray), f"Not numpy.ndarray"
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

    :param np.ndarray gt_true: A list of alleles from ground-truth genotypes.
    :param np.ndarray gt_imputed: A list of alleles from imputed genotypes.
    :ploidy int: Ploidy (1 or 2).
    :return: IQS.
    :rtype: float
    """
    assert ploidy in [1, 2], f"Ploidy {ploidy} is invalid."

    assert np.all(np.isin(np.unique(gt_true), [0, 1])), \
        f"Not all elements are 0 or 1 - {np.unique(gt_true)}."
    assert np.all(np.isin(np.unique(gt_imputed), [0, 1])), \
        f"Not all elements are 0 or 1 - {np.unique(gt_imputed)}."

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

    :param np.ndarray gt_true: A list of alleles from ground-truth genotypes.
    :param np.ndarray gt_imputed: A list of alleles from imputed genotypes.
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

    :param np.ndarray gt_true: A list of alleles from ground-truth genotypes.
    :param np.ndarray gt_imputed: A list of alleles from imputed genotypes.
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

    :param np.ndarray genotypes_true: List of alleles from ground-truth genotypes.
    :param np.ndarray genotypes_imputed: List of alleles from imputed genotypes.
    :return: R-squared correlation coefficient.
    :rtype: float
    """
    r_squared = None
    return r_squared
