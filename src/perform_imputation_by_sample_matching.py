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


def create_index_map(x):
    """
    Maps from ancestral/derived allele space (i.e. 01) to ACGT space (i.e. 0123).
    """
    alleles = ["A", "C", "G", "T", None]
    map_ACGT = [alleles.index(x[i]) for i in range(len(x))]
    if None in x:
        # Otherwise, the last element is 4.
        map_ACGT[-1] = tskit.MISSING_DATA
    map_ACGT = np.array(map_ACGT)
    return map_ACGT


def get_traceback_path(
    tree_sequence, haplotype, recombination_rates, mutation_rates, precision
):
    """
    :param tskit.TreeSequence tree_sequence: Tree sequence with samples to match against.
    :param numpy.ndarray haplotype: Haplotype in ACGT space.
    :param numpy.ndarray recombination_rates: Site-specific recombination rates.
    :param numpy.ndarray mutation_rates: Site-specifc mutation rates.
    :param float precision: Precision of likelihood calculations.
    :return: HMM path (a list of sample IDs).
    :rtype: numpy.ndarray
    """
    assert tree_sequence.num_sites == len(haplotype), (
        f"Length of haplotype array {tree_sequence.num_sites} "
        "differs from "
        f"number of sites {len(haplotype)}."
    )
    assert len(haplotype) == len(recombination_rates), (
        f"Length of haplotype array {len(haplotype)} "
        "differs from "
        f"length of recombination rate array {len(recombination_rates)}."
    )
    assert len(recombination_rates) == len(mutation_rates), (
        f"Length of recombination rate array {len(recombination_rates)}"
        f"differs from"
        f"length of mutation rate array {len(mutation_rates)}."
    )
    assert np.all(np.isin(haplotype, np.array([tskit.MISSING_DATA, 0, 1, 2, 3]))), (
        f"Haplotype array contains only {tskit.MISSING_DATA}, 0, 1, 2, and 3 "
        f"- {np.unique(haplotype)}."
    )

    ll_ts = tree_sequence.ll_tree_sequence

    ls_hmm = _tskit.LsHmm(
        ll_ts,
        recombination_rate=recombination_rates,
        mutation_rate=mutation_rates,
        precision=precision,
        acgt_alleles=True,  # Matrix is in ACGT (i.e. 0123) space
    )

    vm = _tskit.ViterbiMatrix(ll_ts)
    ls_hmm.viterbi_matrix(haplotype, vm)
    path = vm.traceback()

    assert len(path) == tree_sequence.num_sites, (
        f"Length of HMM path {len(path)} "
        "differs from "
        f"number of sites {tree_sequence.num_sites}."
    )
    assert np.all(
        np.isin(path, tree_sequence.samples())
    ), f"Some IDs in the path are not sample IDs."

    return path


def impute_by_sample_matching(
    tree_sequence, sample_data, recombination_rates, mutation_rates, precision
):
    assert tree_sequence.num_sites == sample_data.num_sites
    assert np.all(np.isin(tree_sequence.sites_position, sample_data.sites_position))

    logging.info("Mapping samples to ACGT space.")
    H1 = np.zeros((tree_sequence.num_sites, sample_data.num_samples), dtype=np.int32)
    i = 0
    for v in tqdm(sample_data.variants()):
        if v.site.position in tree_sequence.sites_position:
            H1[i, :] = create_index_map(v.alleles)[v.genotypes]
            i += 1
    H1 = H1.T

    logging.info("Performing traceback.")
    H2 = np.zeros((sample_data.num_samples, tree_sequence.num_sites), dtype=np.int32)
    for i in tqdm(np.arange(sample_data.num_samples)):
        H2[i, :] = get_traceback_path(
            tree_sequence=tree_sequence,
            haplotype=H1[i, :],
            recombination_rates=recombination_rates,
            mutation_rates=mutation_rates,
            precision=precision,
        )

    logging.info("Imputing into samples.")
    i = 0
    H3 = np.zeros((sample_data.num_samples, tree_sequence.num_sites), dtype=np.int32)
    for v in tqdm(tree_sequence.variants()):
        H3[:, i] = v.genotypes[H2[:, i]]
        i += 1

    return H3


def write_genotype_matrix_to_samples(
    tree_sequence,
    genotype_matrix,
    mask_site_pos,
    chip_site_pos,
    out_file,
):
    assert tree_sequence.num_sites == genotype_matrix.shape[1]

    out_file = str(out_file)
    ts_iter = tree_sequence.variants()

    i = 0  # Track iterating through `genotype_matrix`
    with tsinfer.SampleData(path=out_file) as sample_data:
        for ts_v in tqdm(ts_iter):
            # TODO: Inherit site metadata from compatible samples.
            # Set metadata
            metadata = {"marker": ""}
            if ts_v.site.position in mask_site_pos:
                metadata["marker"] = "mask"
            elif ts_v.site.position in chip_site_pos:
                metadata["marker"] = "chip"
            # Add site
            sample_data.add_site(
                position=ts_v.site.position,
                genotypes=genotype_matrix[:, i],
                alleles=ts_v.alleles,
                metadata=metadata,
            )
            i += 1


@click.command()
@click.option(
    "--in_reference_trees_file",
    "-i1",
    required=True,
    help="Trees file with reference genomes.",
)
@click.option(
    "--in_target_samples_file",
    "-i2",
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

    logging.info(f"Loading chip position file: {in_chip_file}")
    chip_site_pos_all = masks.parse_site_position_file(in_chip_file, one_based=False)

    # Set uniform genome-wide recombination rate
    recombination_rates = np.repeat(1e-8, len(ref_site_pos))
    # Apply genetic map if supplied.
    if in_genetic_map_file is not None:
        logging.info(f"Loading genetic map file: {in_genetic_map_file}")
        genetic_map = msprime.RateMap.read_hapmap(in_genetic_map_file)
        # Coordinates must be discrete, not continuous.
        assert np.all(np.round(ref_site_pos) == ref_site_pos)
        recombination_rates = genetic_map.get_rate(ref_site_pos)
        # `_tskit.LsHmm()` cannot handle NaN, so replace them with zero.
        recombination_rates = np.where(
            np.isnan(recombination_rates), 0, recombination_rates
        )
    # Set uniform genome-wide mutation rate
    mutation_rates = np.repeat(1e-8, len(ref_site_pos))

    logging.info("Defining chip and mask sites relative to the reference trees.")
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

    logging.info("Making samples compatible with the reference trees.")
    logging.info(f"Writing compatible samples to file: {compat_samples_file}")
    sd_compat = util.make_compatible_sample_data(
        sample_data=sd_target,
        ancestors_ts=ts_ref,
        skip_unused_markers=True,
        chip_site_pos=chip_site_pos,
        mask_site_pos=mask_site_pos,
        path=str(compat_samples_file),
    )

    logging.info("Imputing into target samples.")
    gm_imputed = impute_by_sample_matching(
        tree_sequence=ts_ref,
        sample_data=sd_compat,
        recombination_rates=recombination_rates,
        mutation_rates=mutation_rates,
        precision=precision,
    )

    logging.info(f"Writing imputed samples to file: {out_samples_file}")
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
