"""Re-implementation of the BEAGLE 4.1 algorithm"""
import warnings
from dataclasses import dataclass

import numpy as np
from numba import njit

import tskit


@dataclass(frozen=True)
class GeneticMap:
    """
    A genetic map containing:
    * Physical positions (bp).
    * Genetic map positions (cM).
    """

    base_pos: np.ndarray
    gen_pos: np.ndarray

    def __post_init__(self):
        assert len(self.base_pos) == len(
            self.gen_pos
        ), "Incompatible physical positions and genetic map positions."
        assert np.all(
            self.base_pos[1:] > self.base_pos[:-1]
        ), "Physical positions are not sorted in strict ascending order."


# Helper functions.
def check_data(ref_h, query_h):
    """
    For each position, check whether the alleles in the query haplotype
    are represented in the reference haplotypes.

    Missing data (i.e. -1) are ignored.

    :param numpy.ndarray ref_h: Reference haplotypes.
    :param numpy.ndarray query_h: One query haplotype.
    :return: True if alleles in query are in references at all the positions.
    :rtype: bool
    """
    m = ref_h.shape[0]  # Number of genotyped positions.
    assert m == len(query_h), "Reference and query haplotypes differ in length."
    for i in range(m):
        if query_h[i] != tskit.MISSING_DATA:
            ref_a = np.unique(ref_h[i, :])
            if query_h[i] not in ref_a:
                raise AssertionError(
                    f"Allele {query_h[i]} at the {i}-th position is not in reference."
                )
    return True


def convert_to_cm(site_pos, genetic_map=None):
    """
    Convert site positions (bp) to genetic map positions (cM).

    In BEAGLE 4.1, when a genetic map is not specified, it is assumed
    that the recombination rate is constant (1 cM / 1 Mb).

    If a genetic map is specified, then the genetic map positions are
    either taken directly from it or interpolated using it.

    The genetic map needs to contain site positions and their corresponding
    genetic map positions. For details, see `PlinkGenMap.java`
    in the BEAGLE 4.1 source code.

    :param numpy.ndarray site_pos: Site positions (bp).
    :param GeneticMap genetic_map: Genetic map.
    :return: Genetic map positions (cM).
    :rtype: numpy.ndarray
    """
    # See 'cumPos' in 'ImputationData.java' in BEAGLE 4.1.
    _MIN_CM_DIST = 1e-7
    if genetic_map is None:
        return site_pos / 1e6  # 1 cM / 1 Mb
    assert np.all(site_pos >= genetic_map.base_pos[0]) and np.all(
        site_pos < genetic_map.base_pos[-1]
    ), "Some site positions are outside of the genetic map."
    # Approximate genetic map distances by linear interpolation.
    # Note np.searchsorted(a, v, side='right') returns i s.t. a[i-1] <= v < a[i].
    right_idx = np.searchsorted(genetic_map.base_pos, site_pos, side="right")
    m = len(site_pos)
    est_cm = np.zeros(m, dtype=np.float64)  # BEAGLE 4.1 uses double in Java.
    for i in range(m):
        a = genetic_map.base_pos[right_idx[i] - 1]
        b = genetic_map.base_pos[right_idx[i]]
        fa = genetic_map.gen_pos[right_idx[i] - 1]
        fb = genetic_map.gen_pos[right_idx[i]]
        assert (
            site_pos[i] >= a
        ), f"Query position is not >= left-bound position: {site_pos[i]}, {a}."
        assert (
            fb >= fa
        ), f"Genetic map positions are not monotonically ascending: {fb}, {fa}."
        est_cm[i] = fa
        est_cm[i] += (fb - fa) * (site_pos[i] - a) / (b - a)
        # Ensure that adjacent positions are not identical in cM.
        if i > 0:
            if est_cm[i] - est_cm[i - 1] < _MIN_CM_DIST:
                est_cm[i] = est_cm[i - 1] + _MIN_CM_DIST
    return est_cm


# HMM probabilities.
def get_mismatch_probs(num_sites, error_rate):
    """
    Compute mismatch probabilities at genotyped positions.

    Mutation rates should be dominated by the rate of allele error,
    which should be the main source of mismatch between query and
    reference haplotypes.

    In BEAGLE 4.1/5.4, error rate is 1e-4 by default, and capped at 0.5.
    In IMPUTE5, the default value is also 1e-4.

    This corresponds to `mu` in `_tskit.LsHmm`.

    :param int num_sites: Number of sites.
    :param float error_rate: Allele error rate.
    :return: Mismatch probabilities.
    :rtype: numpy.ndarray
    """
    MAX_ERROR_RATE = 0.50
    if error_rate >= MAX_ERROR_RATE:
        error_rate = MAX_ERROR_RATE
    mismatch_probs = np.zeros(num_sites, dtype=np.float64) + error_rate
    return mismatch_probs


def get_transition_probs(cm, h, ne):
    """
    Compute probabilities of transitioning to a random state at genotyped positions.

    In BEAGLE 4.1, the default value of `ne` is set to 1e6,
    whereas in BEAGLE 5.4, the default value of `ne` is set to 1e5.
    In BEAGLE 4.1/5.4, this value was optimized empirically on datasets
    from large outbred human populations.

    In IMPUTE5, the default value of `ne` is set to 1e6.

    If `h` is small and `ne` is large, the transition probabilities are ~1.
    In such cases, it may be desirable to set `ne` to a small value
    to avoid switching between haplotypes too frequently.

    This corresponds to `rho` in `_tskit.LsHmm`.

    :param numpy.ndarray cm: Genetic map positions (cM).
    :param int h: Number of reference haplotypes.
    :param float ne: Effective population size.
    :return: Transition probabilities.
    :rtype: numpy.ndarray
    """
    # E(number of crossover events) at first site is always 0.
    r = np.zeros(len(cm), dtype=np.float64)
    r[1:] = np.diff(cm)
    c = -0.04 * (ne / h)
    trans_probs = -1 * np.expm1(c * r)
    return trans_probs


@njit
def compute_emission_probability(mismatch_prob, ref_a, query_a, num_alleles=2):
    """
    Compute the emission probability at a site based on whether the alleles
    carried by a query haplotype and a reference haplotype match at the site.

    Emission probability may be scaled according to the number of distinct
    segregating alleles.

    :param float mismatch_prob: Mismatch probability.
    :param int ref_a: Reference allele.
    :param int query_a: Query allele.
    :param int num_alleles: Number of distinct alleles (default = 2).
    :return: Emission probability.
    :rtype: float
    """
    if ref_a == query_a:
        return 1.0 - (num_alleles - 1) * mismatch_prob
    return mismatch_prob


# Replication of BEAGLE's implementation of LS HMM forward-backward algorithm.
@njit
def compute_forward_matrix_beaglelike(
    ref_h, query_h, trans_probs, mismatch_probs, *, num_alleles=2
):
    """
    Implement LS HMM forward algorithm as in BEAGLE.

    Reference haplotypes and query haplotype are subsetted to genotyped positions.
    So, they are a matrix of size (m, h) and an array of size m, respectively.

    This computes a forward probablity matrix of size (m, h).

    :param numpy.ndarray ref_h: Reference haplotypes.
    :param numpy.ndarray query_h: A query haplotype.
    :param numpy.ndarray trans_probs: Transition probabilities.
    :param numpy.ndarray mismatch_probs: Mismatch probabilities.
    :param int num_alleles: Number of distinct alleles (default = 2).
    :return: Forward probability matrix.
    :rtype: numpy.ndarray
    """
    h = ref_h.shape[1]  # Number of reference haplotypes.
    m = ref_h.shape[0]  # Number of genotyped positions.
    assert len(query_h) == m
    fwd_mat = np.zeros((m, h), dtype=np.float64)
    last_sum = 1.0  # Normalization factor.
    for i in range(m):
        # Get site-specific parameters.
        shift = trans_probs[i] / h
        scale = (1 - trans_probs[i]) / last_sum
        # Get allele at genotyped position i on query haplotype.
        query_a = query_h[i]
        for j in range(h):
            # Get allele at genotyped position i on reference haplotype j.
            ref_a = ref_h[i, j]
            # Get emission probability.
            em_prob = mismatch_probs[i]
            if query_a == ref_a:
                em_prob = 1.0 - (num_alleles - 1) * mismatch_probs[i]
            fwd_mat[i, j] = em_prob
            if i > 0:
                fwd_mat[i, j] *= scale * fwd_mat[i - 1, j] + shift
        site_sum = np.sum(fwd_mat[i, :])
        # Prior probabilities are multiplied when i = 0.
        last_sum = site_sum / h if i == 0 else site_sum
    return fwd_mat


@njit
def compute_forward_matrix(
    ref_h, query_h, trans_probs, mismatch_probs, *, num_alleles=2
):
    m, h = ref_h.shape
    assert len(query_h) == m
    fwd_mat = np.zeros((m, h), dtype=np.float64)
    last_sum = 1.0
    for i in range(m):
        shift = trans_probs[i] / h
        scale = (1 - trans_probs[i]) / last_sum
        for j in range(h):
            fwd_mat[i, j] = compute_emission_probability(
                mismatch_prob=mismatch_probs[i],
                ref_a=ref_h[i, j],
                query_a=query_h[i],
                num_alleles=num_alleles,
            )
            if i > 0:
                fwd_mat[i, j] *= scale * fwd_mat[i - 1, j] + shift
        site_sum = np.sum(fwd_mat[i, :])
        last_sum = site_sum / h if i == 0 else site_sum
    return fwd_mat


@njit
def compute_backward_matrix_beaglelike(
    ref_h, query_h, trans_probs, mismatch_probs, *, num_alleles=2
):
    """
    Implement LS HMM backward algorithm as in BEAGLE.

    Reference haplotypes and query haplotype are subsetted to genotyped positions.
    So, they are a matrix of size (m, h) and an array of size m, respectively.

    This computes a backward probablity matrix of size (m, h).

    In BEAGLE 4.1, the values are kept one position at a time.
    Here, we keep the values at all the positions.

    :param numpy.ndarray ref_h: Reference haplotypes.
    :param numpy.ndarray query_h: One query haplotype.
    :param numpy.ndarray trans_probs: Transition probabilities.
    :param numpy.ndarray mismatch_probs: Mismatch probabilities.
    :param int num_alleles: Number of distinct alleles (default = 2).
    :return: Backward probability matrix.
    :rtype: numpy.ndarray
    """
    h = ref_h.shape[1]  # Number of reference haplotypes.
    m = ref_h.shape[0]  # Number of genotyped positions.
    assert len(query_h) == m
    bwd_mat = np.zeros((m, h), dtype=np.float64)
    bwd_mat[-1, :] = 1.0 / h  # Initialise the last column.
    for i in range(m - 2, -1, -1):
        iP1 = i + 1
        query_a = query_h[iP1]
        for j in range(h):
            ref_a = ref_h[iP1, j]
            em_prob = mismatch_probs[iP1]
            if query_a == ref_a:
                em_prob = 1.0 - (num_alleles - 1) * mismatch_probs[iP1]
            bwd_mat[iP1, j] *= em_prob
        site_sum = np.sum(bwd_mat[iP1, :])
        scale = (1 - trans_probs[iP1]) / site_sum
        shift = trans_probs[iP1] / h
        bwd_mat[i, :] = scale * bwd_mat[iP1, :] + shift
    return bwd_mat


@njit
def compute_backward_matrix(
    ref_h, query_h, trans_probs, mismatch_probs, *, num_alleles=2
):
    m, h = ref_h.shape
    assert len(query_h) == m
    bwd_mat = np.zeros((m, h), dtype=np.float64)
    bwd_mat[-1, :] = 1.0 / h
    for i in range(m - 2, -1, -1):
        iP1 = i + 1
        for j in range(h):
            bwd_mat[iP1, j] *= compute_emission_probability(
                mismatch_prob=mismatch_probs[iP1],
                ref_a=ref_h[iP1, j],
                query_a=query_h[iP1],
                num_alleles=num_alleles,
            )
        site_sum = np.sum(bwd_mat[iP1, :])
        scale = (1 - trans_probs[iP1]) / site_sum
        shift = trans_probs[iP1] / h
        bwd_mat[i, :] = scale * bwd_mat[iP1, :] + shift
    return bwd_mat


@njit
def compute_state_prob_matrix(fwd_mat, bwd_mat):
    """
    Implementing this is simpler than faithfully reproducing BEAGLE 4.1.

    :param numpy.ndarray fwd_mat: Forward probability matrix.
    :param numpy.ndarray bwd_mat: Backward probability matrix.
    :return: Posterior state probability matrix.
    :rtype: numpy.ndarray
    """
    assert (
        fwd_mat.shape == bwd_mat.shape
    ), "Incompatible forward and backward matrices."
    m, h = fwd_mat.shape
    state_mat = np.zeros_like(fwd_mat)
    for i in range(m):
        site_sum = 0
        for j in range(h):
            state_mat[i, j] = fwd_mat[i, j] * bwd_mat[i, j]
            site_sum += state_mat[i, j]
        for j in range(h):
            state_mat[i, j] /= site_sum
    return state_mat


# Imputation.
@njit
def get_weights(typed_pos, untyped_pos, typed_cm, untyped_cm):
    """
    Compute weights for the ungenotyped positions in a query haplotype, which are used in
    linear interpolation of hidden state probabilities at the ungenotyped positions.

    In BB2016 (see below Equation 1), a weight between genotyped positions m and (m + 1)
    bounding ungenotyped position x is denoted lambda_m,x.

    lambda_m,x = [g(m + 1) - g(x)] / [g(m + 1) - g(m)],
    where g(i) = genetic map position of marker i.

    This looks for the right-bounding position instead of the left-bounding.

    :param numpy.ndarray typed_pos: Physical positions of genotyped markers (bp).
    :param numpy.ndarray untyped_pos: Physical positions of ungenotyped markers (bp).
    :param numpy.ndarray typed_cm: Genetic map positions of genotyped markers (cM).
    :param numpy.ndarray untyped_cm: Genetic map positions of ungenotyped markers (cM).
    :return: Weights for ungenotyped positions and indices of right-bounding positions.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    m = len(typed_pos)  # Number of genotyped positions.
    x = len(untyped_pos)  # Number of ungenotyped positions.
    # Identify genotype positions (m - 1) and m bounding ungenotyped position i.
    # Note np.searchsorted(a, v, side='right') returns i s.t. a[i-1] <= v < a[i].
    right_idx = np.searchsorted(typed_pos, untyped_pos, side="right")
    # Calculate weights for ungenotyped positions.
    weights = np.zeros(x, dtype=np.float64)
    for i in range(len(right_idx)):
        k = right_idx[i]
        if k == 0:
            # Left of the first genotyped position.
            weights[i] = 1.0
        elif k == m:
            # Right of the last genotyped position.
            weights[i] = 0.0
        else:
            # Between two genotyped positions.
            cm_m2x = typed_cm[k] - untyped_cm[i]
            # Avoid negative weights.
            if cm_m2x < 0:
                cm_m2x = 0
            cm_m2mM1 = typed_cm[k] - typed_cm[k - 1]
            weights[i] = cm_m2x / cm_m2mM1
    return (weights, right_idx)


@njit
def interpolate_allele_probs(
    state_mat,
    ref_h,
    pos_typed,
    pos_untyped,
    cm_typed,
    cm_untyped,
    *,
    use_threshold=False,
    return_weights=False,
):
    """
    Interpolate allele probabilities at the ungenotyped positions of a query haplotype
    following Equation 1 of BB2016.

    The interpolated allele probability matrix is of size (x, a),
    where a is the number of alleles.

    Note that this function takes:
    1. Hidden state probability matrix at genotyped positions of size (m, h).
    2. Reference haplotypes subsetted to ungenotyped positions of size (x, h).

    If thresholding is employed, it replicates BEAGLE's way to approximate calculations.
    See 'setFirstAlleleProbs', 'setAlleleProbs', and 'setLastAlleleProbs'
    in 'LSHapBaum.java' in BEAGLE 4.1 source code.

    :param numpy.ndarray state_mat: State probability matrix at genotyped positions.
    :param numpy.ndarray ref_h: Reference haplotypes subsetted to ungenotyped positions.
    :param numpy.ndarray pos_typed: Physical positions of genotyped markers (bp).
    :param numpy.ndarray pos_untyped: Physical positions of ungenotyped markers (bp).
    :param numpy.ndarray cm_typed: Genetic map positions at genotyped markers (cM).
    :param numpy.ndarray cm_untyped: Genetic map positions at ungenotyped markers (cM).
    :param bool use_threshold: Set trivial probabilities to 0 if True (default = False).
    :param bool return_weights: Return weights if True (default = False).
    :return: Imputed allele probabilities and weights.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    # TODO: Allow for biallelic site matrix. Work with `_tskit.lshmm` properly.
    alleles = np.arange(len(tskit.ALLELES_ACGT))
    m = state_mat.shape[0]  # Number of genotyped positions.
    x = ref_h.shape[0]  # Number of ungenotyped positions.
    # Set threshold to set trivial probabilities to zero.
    _MIN_THRESHOLD = 0
    weights, right_idx = get_weights(pos_typed, pos_untyped, cm_typed, cm_untyped)
    probs = np.zeros((x, len(alleles)), dtype=np.float64)
    for i in range(x):
        k = right_idx[i]
        w = weights[i]
        for a in alleles:
            is_a_in_ref_h = ref_h[i, :] == a
            if np.sum(is_a_in_ref_h) == 0:
                # This avoids division by zero when getting a threshold adaptively below.
                continue
            if use_threshold:
                # TODO: Check whether this is implemented correctly. Not used by default.
                # Threshold based on "the number of subsets in the partition Am of H".
                threshold_Am = 1 / np.sum(is_a_in_ref_h)
                _MIN_THRESHOLD = min(0.005, threshold_Am)
            if k == 0:
                # See 'setFirstAlleleProbs' in 'LSHapBaum.java'.
                assert w == 1.0, "Weight should be 1.0."
                sum_probs_a_k = np.sum(state_mat[k, is_a_in_ref_h])
                if sum_probs_a_k > _MIN_THRESHOLD:
                    probs[i, a] += sum_probs_a_k
            elif k == m:
                # See 'setLastAlleleProbs' in 'LSHapBaum.java'.
                assert w == 0.0, "Weight should be 0.0."
                sum_probs_a_kM1 = np.sum(state_mat[k - 1, is_a_in_ref_h])
                if sum_probs_a_kM1 > _MIN_THRESHOLD:
                    probs[i, a] += sum_probs_a_kM1
            else:
                # See 'setAlleleProbs' in 'LSHapBaum.java'.
                sum_probs_a_k = np.sum(state_mat[k, is_a_in_ref_h])
                sum_probs_a_kM1 = np.sum(state_mat[k - 1, is_a_in_ref_h])
                if max(sum_probs_a_k, sum_probs_a_kM1) > _MIN_THRESHOLD:
                    probs[i, a] += w * sum_probs_a_kM1
                    probs[i, a] += (1 - w) * sum_probs_a_k
    site_sums = np.sum(probs, axis=1)
    assert np.all(site_sums > 0), "Some site sums of allele probabilities is <= 0."
    probs_rescaled = probs / site_sums[:, np.newaxis]
    if return_weights:
        return (probs_rescaled, weights)
    return (probs_rescaled, None)


@njit
def get_map_alleles(allele_probs, num_alleles):
    """
    Compute maximum a posteriori (MAP) alleles at the ungenotyped positions
    of a query haplotype, based on posterior marginal allele probabilities.

    The imputed alleles and their probabilities are arrays of size x.

    WARN: If the allele probabilities are equal, then allele 0 is arbitrarily chosen.
    TODO: Investigate how often this happens and the effect of this arbitrary choice.

    :param numpy.ndarray allele_probs: Imputed allele probabilities.
    :param int num_alleles: Number of distinct alleles
    :return: Imputed alleles and their probabilities.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    x, a = allele_probs.shape
    assert a == num_alleles
    imputed_alleles = np.zeros(x, dtype=np.int32) - 1
    max_allele_probs = np.zeros(x, dtype=np.float64)
    for i in range(x):
        imputed_alleles[i] = 0
        max_allele_probs[i] = allele_probs[i, 0]
        for j in range(1, num_alleles):
            if allele_probs[i, j] > max_allele_probs[i]:
                imputed_alleles[i] = j
                max_allele_probs[i] = allele_probs[i, j]
    return (imputed_alleles, max_allele_probs)


def run_interpolation_beagle(
    ref_h,
    query_h,
    pos_all,
    *,
    ne=1e6,
    error_rate=1e-4,
    genetic_map=None,
    use_threshold=False,
):
    """
    Perform a simplified version of the procedure of interpolation-style imputation
    based on Equation 1 of BB2016.

    Reference haplotypes and query haplotype are of size (m + x, h) and (m + x).

    The physical positions of all the markers are an array of size (m + x).

    This produces imputed alleles and their probabilities at the ungenotyped positions
    of the query haplotype.

    The default values of `ne` and `error_rate` are taken from BEAGLE 4.1, not 5.4.
    In BEAGLE 5.4, the default value of `ne` is 1e5, and `error_rate` is data-dependent.

    :param numpy.ndarray ref_h: Reference haplotypes.
    :param numpy.ndarray query_h: One query haplotype.
    :param numpy.ndarray pos_all: Physical positions of all the markers (bp).
    :param int ne: Effective population size (default = 1e6).
    :param float error_rate: Allele error rate (default = 1e-4).
    :param GeneticMap genetic_map: Genetic map (default = None).
    :param bool use_threshold: Set trivial probabilities to 0 if True (default = False).
    :return: Imputed alleles and their probabilities.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    warnings.warn("This function is experimental and not fully tested.", stacklevel=1)
    warnings.warn(
        "Check the reference and query haplotypes use the same allele encoding.",
        stacklevel=1,
    )
    num_alleles = len(tskit.ALLELES_ACGT)
    h = ref_h.shape[1]  # Number of reference haplotypes.
    # Separate indices of genotyped and ungenotyped positions.
    idx_typed = np.where(query_h != tskit.MISSING_DATA)[0]
    idx_untyped = np.where(query_h == tskit.MISSING_DATA)[0]
    # Get physical positions of of genotyped and ungenotyped markers.
    pos_typed = pos_all[idx_typed]
    pos_untyped = pos_all[idx_untyped]
    # Get genetic map positions of of genotyped and ungenotyped markers.
    cm_typed = convert_to_cm(pos_typed, genetic_map=genetic_map)
    cm_untyped = convert_to_cm(pos_untyped, genetic_map=genetic_map)
    # Get HMM probabilities at genotyped positions.
    trans_probs = get_transition_probs(cm_typed, h=h, ne=ne)
    mismatch_probs = get_mismatch_probs(len(pos_typed), error_rate=error_rate)
    # Subset haplotypes.
    ref_h_typed = ref_h[idx_typed, :]
    ref_h_untyped = ref_h[idx_untyped, :]
    query_h_typed = query_h[idx_typed]
    # Compute matrices at genotyped positions.
    fwd_mat = compute_forward_matrix(
        ref_h_typed,
        query_h_typed,
        trans_probs,
        mismatch_probs,
        num_alleles=num_alleles,
    )
    bwd_mat = compute_backward_matrix(
        ref_h_typed,
        query_h_typed,
        trans_probs,
        mismatch_probs,
        num_alleles=num_alleles,
    )
    state_mat = compute_state_prob_matrix(fwd_mat, bwd_mat)
    # Interpolate allele probabilities.
    imputed_allele_probs, _ = interpolate_allele_probs(
        state_mat=state_mat,
        ref_h=ref_h_untyped,
        pos_typed=pos_typed,
        pos_untyped=pos_untyped,
        cm_typed=cm_typed,
        cm_untyped=cm_untyped,
        use_threshold=use_threshold,
        return_weights=False,
    )
    imputed_alleles, max_allele_probs = get_map_alleles(imputed_allele_probs, num_alleles=num_alleles)
    return (imputed_alleles, max_allele_probs)
