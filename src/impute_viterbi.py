import logging
from tqdm import tqdm

import numpy as np

import _tskit
import tskit


_ACGT_INTEGERS_ = np.array([tskit.MISSING_DATA, 0, 1, 2, 3])


def prepare_input_matrix(ref_ts, target_ds):
    """
    Step 1.

    Prepare a matrix that has number of target sample genomes (rows)
    by number of variable sites in reference panel (columns).

    The elements in the matrix correspond to indices in `_ACGT_LETTERS_`
    Note that -1 (or None) is used to denote missing data.

    :param tsinfer.TreeSequence ref_ts: Tree sequence with ref. samples to match against.
    :param xarray.Dataset target_ds: Samples to impute into.
    :return: Matrix of samples (rows) x sites (columns).
    :rtype: numpy.ndarray
    """
    num_sites = ref_ts.num_sites
    num_samples = target_ds.dims["samples"] * target_ds.dims["ploidy"]

    H1 = np.full(
        (num_sites, num_samples),
        tskit.MISSING_DATA,
        dtype=np.int32
    )

    i = 0
    for pos in tqdm(target_ds.variant_position.values):
        assert pos in ref_ts.sites_position
        ref_site_idx = np.where(ref_ts.sites_position == pos)[0]
        assert len(ref_site_idx) == 1
        ref_site_idx = ref_site_idx[0]
        H1[ref_site_idx, :] = target_ds.call_genotype[i].values.flatten()
        i += 1

    return H1.T


def get_traceback_path(
    ts, sample_sequence, recombination_rates, mutation_rates, precision
):
    """
    Perform traceback on the HMM to get a path of sample IDs.

    :param tskit.TreeSequence ts: Tree sequence with samples to match against.
    :param numpy.ndarray sample_sequence: Sample sequence in ACGT space.
    :param numpy.ndarray recombination_rates: Site-specific recombination rates.
    :param numpy.ndarray mutation_rates: Site-specifc mutation rates.
    :param float precision: Precision of likelihood calculations.
    :return: List of sample IDs representing the HMM path.
    :rtype: numpy.ndarray
    """
    assert ts.num_sites == len(
        sample_sequence
    ), f"Lengths of tree sequence and sample sequence differ."
    assert len(sample_sequence) == len(
        recombination_rates
    ), f"Lengths of sample sequence and recombination rates differ."
    assert len(recombination_rates) == len(
        mutation_rates
    ), f"Lengths of recombination rates and mutation rates differ."
    assert np.all(
        np.isin(sample_sequence, _ACGT_INTEGERS_)
    ), f"Sample sequence has character(s) not in {_ACGT_INTEGERS_}."

    ll_ts = ts.ll_tree_sequence

    ls_hmm = _tskit.LsHmm(
        ll_ts,
        recombination_rate=recombination_rates,
        mutation_rate=mutation_rates,
        precision=precision,
        acgt_alleles=True,  # In ACGT space
    )

    vm = _tskit.ViterbiMatrix(ll_ts)
    ls_hmm.viterbi_matrix(sample_sequence, vm)
    path = vm.traceback()

    assert (
        len(path) == ts.num_sites
    ), f"Length of HMM path differs from number of sites."
    assert np.all(
        np.isin(path, ts.samples())
    ), f"Some IDs in the path are not sample IDs."

    return path


def perform_hmm_traceback(ts, H1, switch_prob, mismatch_prob, precision):
    """
    Step 2.

    Given a matrix of samples (rows) x sites (columns), trace through a Li & Stephens HMM,
    parameterised by per-site switch probabilities and mismatch probabilities,
    to obtain the Viteri HMM path of sample IDs.

    :param tskit.TreeSequence ts: Tree sequence with samples to match against.
    :param numpy.ndarray H1: Matrix of samples (rows) x sites (columns).
    :param numpy.ndarray switch_prob: Per-site switch probabilities.
    :param numpy.ndarray mismatch_prob: Per-site mutation rates.
    :param float precision: Precision of HMM likelihood calculations.
    :return: Sample IDs representing the HMM path.
    :rtype: numpy.ndarray
    """
    assert ts.num_sites == H1.shape[1], \
        f"Number of sites in ts differ from number of columns in H1."
    assert len(switch_prob) == len(mismatch_prob), \
        f"Length of switch prob. differs from length of mismatch prob."
    assert ts.num_sites == len(switch_prob), \
        f"Length of switch prob. and mismatch prob. differ from number of sites."

    # Get the Viterbi path for each sample.
    H2 = np.zeros_like(H1)
    for i in tqdm(np.arange(H1.shape[0])):
        H2[i, :] = get_traceback_path(
            ts=ts,
            sample_sequence=H1[i, :],
            recombination_rates=switch_prob,
            mutation_rates=mismatch_prob,
            precision=precision,
        )

    return H2


def impute_samples(ts, H2):
    """
    Step 3.

    Impute into query haplotypes represented by a matrix of size (h, m),
    where h = number of haplotypes and m = number of sites (reference markers).

    WARN: The imputed alleles are in ACGT encoding.

    :param tskit.TreeSequence ts: Tree sequence with reference samples.
    :param numpy.ndarray H2: Query haplotypes with missing data.
    :return: Query haplotypes with imputed alleles.
    :rtype: numpy.ndarray
    """
    assert (
        ts.num_sites == H2.shape[1]
    ), f"Number of sites in tree sequence and H2 differ."

    H3 = np.zeros_like(H2)
    i = 0
    for v in tqdm(ts.variants(alleles=tskit.ALLELES_ACGT), total=ts.num_sites):
        H3[:, i] = v.genotypes[H2[:, i]]
        i += 1

    return H3


def impute_by_sample_matching(
    ref_ts, target_ds, switch_prob, mismatch_prob, precision
):
    """
    Match samples to a tree sequence using an exact HMM implementation
    of the Li & Stephens model, and then impute into samples.

    :param tskit.TreeSequence ref_ts: Tree sequence with reference samples to match against.
    :param xarray.Dataset target_ds: Samples to impute into.
    :param numpy.ndarray switch_prob: Per-site switch probabilities.
    :param numpy.ndarray mismatch_prob: Per-site mismatch probabilities.
    :param float precision: Precision of likelihood calculations.
    :return: List of three matrices, one for each step of the imputation.
    :rtype: list
    """
    logging.info("Step 1: Prepare input matrix.")
    H1 = prepare_input_matrix(ref_ts, target_ds)

    logging.info("Step 2: Performing HMM traceback.")
    H2 = perform_hmm_traceback(
        ref_ts,
        H1,
        switch_prob=switch_prob,
        mismatch_prob=mismatch_prob,
        precision=precision,
    )

    logging.info("Step 3: Imputing into samples.")
    H3 = impute_samples(ref_ts, H2)

    return [H1, H2, H3]
