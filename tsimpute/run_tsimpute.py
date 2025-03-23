import warnings

import numpy as np

import _tskit
import tskit

from . import beagle_numba


def run_tsimpute(
    ref_ts,
    query_h,
    pos_all,
    *,
    ne=1e6,
    error_rate=1e-4,
    precision=10,
    genetic_map=None,
    use_threshold=False,
):
    """
    Perform interpolation-style imputation, except that the forward and backward
    probability matrices are computed from a tree sequence.

    Reference haplotypes and query haplotype are of size (m + x, h) and (m + x).

    The physical positions of all the markers are an array of size (m + x).

    This produces imputed alleles and their probabilities at the ungenotyped positions
    of the query haplotype.

    The default values for `ne` and `error_rate` are taken from BEAGLE 4.1.

    In an analysis comparing imputation accuracy from precision 6 to 24
    using the FinnGen SiSu dataset (~1k genotyped positions in query haplotypes),
    accuracy was highly similar from 8 to 24 and only drastically worsened at 6.
    Also, in an informal benchmark experiment, the runtime per query haplotype
    improved ~8x, going from precision 22 to 8. This indicates that there is
    a large boost in speed with very little loss in accuracy when precision is 8.
    To be on the safe side, the default value of precision is set to 10.

    Note that BEAGLE 4.1 uses Java float (32-bit) when calculating
    the forward, backward, and hidden state probability matrices.

    TODO: Handle `acgt_alleles` properly.

    :param numpy.ndarray ref_ts: Tree sequence with reference haplotypes.
    :param numpy.ndarray query_h: One query haplotype.
    :param numpy.ndarray pos_all: Physical positions of all the markers (bp).
    :param int ne: Effective population size (default = 1e6).
    :param float error_rate: Allelic error rate (default = 1e-4).
    :param int precision: Precision for running LS HMM (default = 10).
    :param GeneticMap genetic_map: Genetic map (default = None).
    :param bool use_threshold: Set trivial probabilities to 0 if True (default = False).
    :return: Imputed alleles and their probabilities.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    warnings.warn(
        "Check the reference and query haplotypes use the same allele encoding.",
        stacklevel=1,
    )
    h = ref_ts.num_samples  # Number of reference haplotypes.
    # Separate indices of genotyped and ungenotyped positions.
    idx_typed = np.where(query_h != tskit.MISSING_DATA)[0]
    idx_untyped = np.where(query_h == tskit.MISSING_DATA)[0]
    # Set physical positions of genotyped and ungenotyped markers.
    pos_typed = pos_all[idx_typed]
    pos_untyped = pos_all[idx_untyped]
    # Get genetic map positions of genotyped and ungenotyped markers.
    cm_typed = beagle_numba.convert_to_genetic_map_positions(pos_typed, genetic_map=genetic_map)
    cm_untyped = beagle_numba.convert_to_genetic_map_positions(pos_untyped, genetic_map=genetic_map)
    # Get HMM probabilities at genotyped positions.
    trans_probs = beagle_numba.get_transition_probs(cm_typed, h=h, ne=ne)
    mismatch_probs = beagle_numba.get_mismatch_probs(len(pos_typed), error_rate=error_rate)
    # Subset haplotypes.
    ref_ts_typed = ref_ts.delete_sites(site_ids=idx_untyped)
    ref_ts_untyped = ref_ts.delete_sites(site_ids=idx_typed)
    ref_h_untyped = ref_ts_untyped.genotype_matrix(alleles=tskit.ALLELES_ACGT)
    query_h_typed = query_h[idx_typed]
    # Get matrices at genotyped positions from tree sequence.
    fwd_mat = _tskit.CompressedMatrix(ref_ts_typed._ll_tree_sequence)
    bwd_mat = _tskit.CompressedMatrix(ref_ts_typed._ll_tree_sequence)
    # WARN: Be careful with argument `acgt_alleles`!!!
    ls_hmm = _tskit.LsHmm(
        ref_ts_typed._ll_tree_sequence,
        recombination_rate=trans_probs,  # Transition probabilities.
        mutation_rate=mismatch_probs,  # Mismatch probabilities.
        acgt_alleles=True,  # TODO: Handle allele encoding properly.
        precision=precision,
    )
    ls_hmm.forward_matrix(query_h_typed.T, fwd_mat)
    ls_hmm.backward_matrix(query_h_typed.T, fwd_mat.normalisation_factor, bwd_mat)
    # TODO: Check that these state probabilities align.
    state_mat = state_mat = np.multiply(fwd_mat.decode(), bwd_mat.decode())
    # Interpolate allele probabilities.
    imputed_allele_probs, _ = beagle_numba.interpolate_allele_probs(
        state_mat=state_mat,
        ref_h=ref_h_untyped,
        pos_typed=pos_typed,
        pos_untyped=pos_untyped,
        cm_typed=cm_typed,
        cm_untyped=cm_untyped,
        use_threshold=use_threshold,
        return_weights=False,
    )
    imputed_alleles, max_allele_probs = beagle_numba.get_map_alleles(imputed_allele_probs)
    return (imputed_alleles, max_allele_probs)
