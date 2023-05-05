from datetime import datetime
import click
import logging
from pathlib import Path
import sys
from git import Repo
from tqdm import tqdm

import numpy as np
import msprime

import _tskit
import tskit
import tsinfer

sys.path.append("./src")
import masks
import util


_ACGT_INTEGERS_ = np.array([tskit.MISSING_DATA, 0, 1, 2, 3])
_ACGT_LETTERS_ = ["A", "C", "G", "T", None]


def create_index_map(x):
    """
    Prepare an allele map from ancestral/derived state space (i.e. 01) to ACGT space (i.e. 0123).

    :param numpy.ndarray x: Alleles in ancestral/derived state space.
    :return: Alleles in ACGT space.
    :rtype: numpy.ndarray
    """
    map_ACGT = np.array(
        [_ACGT_LETTERS_.index(x[i]) for i in np.arange(len(x))], dtype=np.int32
    )
    if None in x:
        # Otherwise, the last element is 4.
        map_ACGT[-1] = tskit.MISSING_DATA
    return map_ACGT


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
    :return: HMM path (list of sample IDs).
    :rtype: numpy.ndarray
    """
    assert ts.num_sites == len(sample_sequence), (
        f"Length of tree sequence {ts.num_sites} "
        f"differs from "
        f"length of sample sequence {len(sample_sequence)}."
    )
    assert len(sample_sequence) == len(recombination_rates), (
        f"Length of sample sequence {len(sample_sequence)} "
        f"differs from "
        f"length of recombination rate array {len(recombination_rates)}."
    )
    assert len(recombination_rates) == len(mutation_rates), (
        f"Length of recombination rate array {len(recombination_rates)}"
        f"differs from"
        f"length of mutation rate array {len(mutation_rates)}."
    )
    assert np.all(np.isin(sample_sequence, _ACGT_INTEGERS_)), (
        f"Sample sequence has character(s) not in {_ACGT_INTEGERS_},"
        f"{np.unique(sample_sequence)}."
    )

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

    assert len(path) == ts.num_sites, (
        f"Length of HMM path {len(path)} "
        "differs from "
        f"number of sites {ts.num_sites}."
    )
    assert np.all(
        np.isin(path, ts.samples())
    ), f"Some IDs in the path are not sample IDs."

    return path


def remap_state_space(ts, sd, samples=None):
    """
    Remap the alleles in the samples from ancestral/derived state space (01) to ACGT space.

    Automatically sort the sample IDs.

    :param tskit.TreeSequence ts: Tree sequence with samples to match against.
    :param tsinfer.SampleData sd: Samples to impute into.
    :return: Matrix of samples (rows) x sites (columns) in ACGT space.
    :rtype: numpy.ndarray
    """
    assert ts.num_sites == sd.num_sites, \
        f"Number of sites in tree sequence differs from that in sample data."

    if samples is None:
        samples = np.arange(sd.num_samples)
    else:
        assert np.all(np.isin(samples, np.arange(sd.num_samples))), \
            "Some IDs in the samples list are not sample IDs."
        samples = np.sort(np.array(samples))

    # Set up a matrix of sites (rows) x samples (columns).
    H1 = np.zeros((ts.num_sites, len(samples)), dtype=np.int32)
    i = 0
    for v in tqdm(sd.variants(), total=sd.num_sites):
        assert v.site.position in ts.sites_position
        H1[i, :] = create_index_map(v.alleles)[v.genotypes[samples]]
        i += 1
    # Transpose the matrix to get samples (rows) x sites (columns).
    H1 = H1.T

    return(H1)


def perform_hmm_traceback(ts, H1, switch_prob, mismatch_prob, precision):
    """
    Given a matrix of samples (rows) x sites (columns), trace through a Li & Stephens HMM, 
    parameterised by per-site switch probabilities and mismatch probabilities,
    to obtain the Viteri path of sample IDs.

    :param tskit.TreeSequence ts: Tree sequence with samples to match against.
    :param numpy.ndarray H1: Matrix of samples (rows) x sites (columns).
    :param numpy.ndarray switch_prob: Per-site switch probabilities (related to recombination rate).
    :param numpy.ndarray mismatch_prob: Per-site mutation rates (related to mutation rate).
    :param float precision: Precision of HMM likelihood calculations.
    :param samples: Optionally, list of samples specified by IDs.
    :return: Viteri HMM path (array of sample IDs).
    :rtype: numpy.ndarray
    """
    assert ts.num_sites == H1.shape[1], \
        f"Number of sites in tree sequence differ from number of columns of H1."
    assert len(switch_prob) == len(mismatch_prob), \
        f"Length of switch probabilities differs from length of mismatch probabilities."
    assert ts.num_sites == len(switch_prob), \
        f"Length of switch probabilities and mismatch probabilities differ from number of sites."
    
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
    
    return(H2)


def impute_samples(ts, H2):
    """
    Given a matrix of samples (rows) x sites (columns), impute into the samples.

    :param tskit.TreeSequence ts: Tree sequence with samples to match against.
    :param numpy.ndarray H2: Matrix of samples (rows) x sites (columns).
    :return: Matrix of samples (rows) x sites (columns) with imputed alleles.
    :rtype: numpy.ndarray
    """
    assert ts.num_sites == H2.shape[1], \
        f"Number of sites in tree sequence and H2 differ."
    
    H3 = np.zeros_like(H2)
    i = 0
    for v in tqdm(ts.variants(), total=ts.num_sites):
        H3[:, i] = v.genotypes[H2[:, i]]
        i += 1
    
    return(H3)


def impute_by_sample_matching(ts, sd, switch_prob, mismatch_prob, precision, samples=None):
    """
    Match samples to a tree sequence using an exact HMM implementation
    of the Li & Stephens model, and then impute into samples.

    The tree sequence and sample data must have the same site positions.

    :param tskit.TreeSequence ts: Tree sequence with samples to match against.
    :param tsinfer.SampleData sd: Samples to impute into.
    :param numpy.ndarray switch_prob: Per-site switch probabilities.
    :param numpy.ndarray mismatch_prob: Per-site mismatch probabilities.
    :param float precision: Precision of likelihood calculations.
    :param list samples: List of sample IDs to impute into. If None, impute into all samples.
    :return: List of three matrices, one for each step of the imputation.
    :rtype: list
    """
    assert ts.num_sites == sd.num_sites, \
        "Number of sites in tree sequence and sample data differ."
    assert np.all(np.equal(ts.sites_position, sd.sites_position)), \
        "Site positions in tree sequence and sample data do not match."

    logging.info("Step 1: Mapping samples to ACGT space.")
    H1 = remap_state_space(ts, sd, samples=samples)

    logging.info("Step 2: Performing HMM traceback.")
    H2 = perform_hmm_traceback(
        ts,
        H1,
        switch_prob=switch_prob,
        mismatch_prob=mismatch_prob,
        precision=precision,
    )

    logging.info("Step 3: Imputing into samples.")
    H3 = impute_samples(ts, H2)

    return [H1, H2, H3]


def write_genotype_matrix_to_samples(
    ts,
    genotype_matrix,
    out_file,
    mask_site_pos=None,
    chip_site_pos=None,
):
    """
    Write a genotype matrix to a sample data file.

    :param tskit.TreeSequence ts: Tree sequence with sites to match against.
    :param numpy.ndarray genotype_matrix: Genotype matrix in ancestral/derived state space.
    :param pathlib.Path out_file: Path to output samples file.
    :param array-like mask_site_pos: Site positions to mark as "mask".
    :param array-like chip_site_pos: Site positions to mark as "chip".
    """
    assert ts.num_sites == genotype_matrix.shape[1]
    assert (
        len(set(mask_site_pos).intersection(set(chip_site_pos))) == 0
    ), f"Mask and chip site positions are not mutually exclusive."

    if mask_site_pos is None:
        mask_site_pos = []
    if chip_site_pos is None:
        chip_site_pos = []

    i = 0
    with tsinfer.SampleData(path=str(out_file)) as sd:
        for ts_v in tqdm(ts.variants()):
            # TODO: Inherit site metadata from compatible samples.
            # Set site metadata
            metadata = {"marker": ""}
            if ts_v.site.position in mask_site_pos:
                metadata["marker"] = "mask"
            elif ts_v.site.position in chip_site_pos:
                metadata["marker"] = "chip"
            # Add site
            sd.add_site(
                position=ts_v.site.position,
                genotypes=genotype_matrix[:, i],
                alleles=ts_v.alleles,
                metadata=metadata,
            )
            i += 1


@click.command()
@click.option(
    "--in_reference_trees_file",
    "-r",
    required=True,
    help="Trees file with reference panel genomes.",
)
@click.option(
    "--in_target_samples_file",
    "-t",
    required=True,
    help="Samples file with target genomes.",
)
@click.option(
    "--in_chip_file",
    "-c",
    required=True,
    help="Tab-delimited file with chip site positions.",
)
@click.option(
    "--in_genetic_map_file",
    "-g",
    default=None,
    help="Genetic map file in HapMap3 format.",
)
@click.option(
    "--out_dir",
    "-o",
    type=click.Path(exists=True),
    required=True,
    help="Output directory.",
)
@click.option(
    "--out_prefix", "-p", type=str, required=True, help="Prefix of the output file."
)
@click.option(
    "--precision",
    type=int,
    default=10,
    help="Precision for computing likelihood values.",
)
def perform_imputation_by_sample_matching(
    in_reference_trees_file,
    in_target_samples_file,
    in_chip_file,
    in_genetic_map_file,
    out_dir,
    out_prefix,
    precision,
):
    out_dir = Path(out_dir)
    log_file = out_dir / f"{out_prefix}.log"
    compat_samples_file = out_dir / f"{out_prefix}.compat.samples"
    out_samples_file = out_dir / f"{out_prefix}.imputed.samples"

    logging.basicConfig(filename=str(log_file), encoding="utf-8", level=logging.INFO)

    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"start: {start_datetime}")
    logging.info(f"dep: tskit {tskit.__version__}")
    logging.info(f"dep: tsinfer {tsinfer.__version__}")
    repo = Repo(search_parent_directories=True)
    logging.info(f"dep: tsimpute URL {repo.remotes.origin.url}")
    logging.info(f"dep: tsimpute SHA {repo.head.object.hexsha}")

    logging.info(f"Loading reference trees file: {in_reference_trees_file}")
    ts_ref = tskit.load(in_reference_trees_file)
    ref_site_pos = ts_ref.sites_position

    logging.info(f"Loading target samples file: {in_target_samples_file}")
    sd_target = tsinfer.load(in_target_samples_file)

    logging.info(f"Loading chip site position file: {in_chip_file}")
    chip_site_pos_all = masks.parse_site_position_file(in_chip_file, one_based=False)

    # Human-like rates
    recombination_rate_constant = 1e-8
    mutation_rate_constant = 1e-20

    # Set genome-wide recombination rate.
    recombination_rates = np.repeat(recombination_rate_constant, len(ref_site_pos))
    # Apply genetic map, if supplied.
    if in_genetic_map_file is not None:
        logging.info(f"Loading genetic map file: {in_genetic_map_file}")
        genetic_map = msprime.RateMap.read_hapmap(in_genetic_map_file)
        # Coordinates must be discrete, not continuous.
        assert np.all(np.round(ref_site_pos) == ref_site_pos)
        recombination_rates = genetic_map.get_rate(ref_site_pos)
        # `_tskit.LsHmm()` cannot handle NaN, so replace them with a human-like rate.
        recombination_rates = np.where(
            np.isnan(recombination_rates),
            recombination_rate_constant,
            recombination_rates,
        )

    # Set genome-wide mutation rate.
    mutation_rates = np.repeat(mutation_rate_constant, len(ref_site_pos))

    logging.info("Defining chip and mask sites wrt the reference trees.")
    ref_sites_isin_chip = np.isin(
        ref_site_pos,
        chip_site_pos_all,
        assume_unique=True,
    )
    chip_site_pos = ref_site_pos[ref_sites_isin_chip]
    mask_site_pos = ref_site_pos[np.invert(ref_sites_isin_chip)]

    assert (
        len(set(chip_site_pos) & set(mask_site_pos)) == 0
    ), f"Chip and mask site positions are not mutually exclusive."

    logging.info("Making target samples compatible with the reference trees.")
    logging.info(f"Writing ref.-compatible target samples: {compat_samples_file}")
    sd_compat = util.make_compatible_samples(
        target_samples=sd_target,
        ref_tree_sequence=ts_ref,
        skip_unused_markers=True,
        chip_site_pos=chip_site_pos,
        mask_site_pos=mask_site_pos,
        path=str(compat_samples_file),
    )

    logging.info("Imputing into target samples.")
    _, _, gm_imputed = impute_by_sample_matching(
        ts=ts_ref,
        sd=sd_compat,
        recombination_rates=recombination_rates,
        mutation_rates=mutation_rates,
        precision=precision,
    )

    logging.info(f"Writing imputed samples: {out_samples_file}")
    write_genotype_matrix_to_samples(
        tree_sequence=ts_ref,
        genotype_matrix=gm_imputed,
        chip_site_pos=chip_site_pos,
        mask_site_pos=mask_site_pos,
        out_file=str(out_samples_file),
    )

    end_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"end: {end_datetime}")


if __name__ == "__main__":
    perform_imputation_by_sample_matching()
