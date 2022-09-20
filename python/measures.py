import numpy as np


def compute_concordance(genotypes_true, genotypes_imputed, allele_state=None):
    """
    Calculate the total concordance (sometimes referred to as imputation accuracy) between
    `genotypes_true` and `genotypes_imputed`.

    Random agreement can inflate total concordance when MAF is low, so interpretation of
    total concordance should be done with caution.

    If `allele_state` is specified, then concordance is calculated based on the elements in
    `genotypes_true` and `genotypes_imputed` where `genotypes_true` is equal to `allele_state`.

    This metric may be suitable for sample-wise (per genome) or site-wise (across genomes)
    comparisons of genotypes.

    :param ndarray genotypes_true:
    :param ndarray genotypes_imputed:
    :param  allele: (default = None)
    :return float:
    """
    assert isinstance(genotypes_true, np.ndarray)
    assert isinstance(genotypes_imputed, np.ndarray)
    assert len(genotypes_true) == len(genotypes_imputed)

    if allele_state != None:
        allele_match_bool = np.isin(genotypes_true, allele_state)
        assert np.any(allele_match_bool)
        genotypes_true = genotypes_true[allele_match_bool]
        genotypes_imputed = genotypes_imputed[allele_match_bool]

    num_genotypes_correct = np.sum(genotypes_true == genotypes_imputed)
    num_genotypes_total = len(genotypes_true)
    concordance = float(num_genotypes_correct) / float(num_genotypes_total)

    return concordance


def compute_iqs(genotypes_true, genotypes_imputed, ploidy):
    assert ploidy == 1 or ploidy == 2, f"Ploidy {ploidy} is invalid."
    if ploidy == 1:
        iqs = compute_iqs_haploid(genotypes_true, genotypes_imputed)
    else:
        iqs = compute_iqs_diploid(genotypes_true, genotypes_imputed)
    return iqs


def compute_iqs_haploid(genotypes_true, genotypes_imputed):
    """
    Calculate the Imputation Quality Score between `genotypes_true` and `genotypes_imputed`.
    1. A value of 1 indicates perfect imputation;
    2. A value of 0 indicates that observed agreement rate is equal to chance agreement rate; and
    3. A negative value indicates that the method imputes poorly than by chance.

    This specific formula is used to compute the IQS of imputed genotypes
    at biallelic sites in HAPLOID genomes.

    :param np.array genotypes_true:
    :param np.array genotypes_imputed:
    :return: IQS
    :rtype: float
    """
    assert len(genotypes_true) == len(genotypes_imputed), \
        f"Arrays of genotype are not of the same length."

    # Allele 0 imputed correctly
    n00 = np.sum([y == 0 for x, y in zip(genotypes_imputed, genotypes_true) if x == 0])
    # Allele 1 imputed correctly
    n11 = np.sum([y == 1 for x, y in zip(genotypes_imputed, genotypes_true) if x == 1])
    # Allele 1 imputed wrongly
    n01 = np.sum([y == 1 for x, y in zip(genotypes_imputed, genotypes_true) if x == 0])
    # Allele 1 imputed wrongly
    n10 = np.sum([y == 0 for x, y in zip(genotypes_imputed, genotypes_true) if x == 1])

    # Marginal counts
    n0_ = n00 + n01
    n1_ = n10 + n11
    n_0 = n00 + n10
    n_1 = n01 + n11

    # Total genotypes imputed
    n__ = n00 + n10 + n01 + n11

    # Observed overall concordance
    Po = float(n00 + n11) / float(n__)

    # Chance agreement
    Pc = float(n0_ * n_0 + n1_ * n_1) / float(n__ * n__)

    assert Po >= 0 and Po <= 1
    assert Pc >= 0 and Pc <= 1

    iqs = float("nan") if Pc == 1 else (Po - Pc) / (1 - Pc)

    return iqs


def compute_iqs_diploid(genotypes_true, genotypes_imputed):
    """
    Calculate the Imputation Quality Score between `genotypes_true` and `genotypes_imputed`.
    1. A value of 1 indicates perfect imputation;
    2. A value of 0 indicates that observed agreement rate is equal to chance agreement rate; and
    3. A negative value indicates that the method imputes poorly than by chance.

    This specific formula is used to compute the IQS of imputed genotypes
    at biallelic sites in DIPLOID genomes.

    :param np.array genotypes_true: List of alleles from ground-truth genotypes
    :param np.array genotypes_imputed: List of alleles from imputed genotypes
    :return: IQS
    :rtype: float
    """
    assert len(genotypes_true) == len(genotypes_imputed), \
        f"Arrays of genotype are not of the same length."
    assert len(genotypes_true) % 2 == 0, \
        f"Not all genotypes are from diploid genomes."

    # Genotype AA correctly imputed as AA
    n11 = None
    # Genotype AA wrongly imputed as AB
    n12 = None
    # Genotype AA wrongly imputed as BB
    n13 = None
    # Genotype AB wrongly imputed as AA
    n21 = None
    # Genotype AB correctly imputed as AB
    n22 = None
    # Genotype AB wrongly imputed as BB
    n23 = None
    # Genotype BB wrongly imputed as AA
    n31 = None
    # Genotype BB wrongly imputed as AB
    n32 = None
    # Genotype BB correctly imputed as BB
    n33 = None
    
    # Marginal counts
    n_1 = None # Total number of cases having genotype AA
    n_2 = None # Total number of cases having genotype AB
    n_3 = None # Total number of cases having genotype BB
    n1_ = None # Total number of cases imputed as genotype AA
    n2_ = None # Total number of cases imputed as genotype AB
    n3_ = None # Total number of cases imputed as genotype BB

    assert n_1 + n_2 + n_3 == n1_ + n2_ + n3_, \
        f"Marginal counts do not add up."

    # Total count
    n__ = n_1 + n_2 + n_3

    # Observed agreement (i.e. overall concordance)
    Po = float(n11 + n22 + n33) / float(n__)

    # Chance agreement
    Pc = float(n1_ * n_1 + n2_ * n_2 + n3_ * n_3) / float(n__ * n__)

    assert Po >= 0 and Po <= 1, f"Po {Po} is not a proportion."
    assert Pc >= 0 and Pc <= 1, f"Pc {Pc} is not a proportion."

    iqs = float("nan") if Pc == 1 else (Po - Pc) / (1 - Pc)

    return iqs
