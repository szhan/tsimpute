import click
import logging
import sys
from tqdm import tqdm
import _tskit
import tskit
import tsinfer
import numpy as np
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


def get_traceback_path(tree_sequence, haplotype, recombination_rate_map, mutation_rate_map, precision):
    """
    :param tskit.TreeSequence tree_sequence: Tree sequence containing sample haplotypes to match against.
    :param numpy.ndarray haplotype: Haplotype in ACGT space.
    :param numpy.ndarray recombination_rate_map: Site-specific recombination rates.
    :param numpy.ndarray mutation_rate_map: Site-specifc mutation rates.
    :param float precision: Precision to calculate likelihood values.
    :return: HMM path (a list of sample IDs).
    :rtype: numpy.ndarray
    """
    assert tree_sequence.num_sites == len(haplotype)
    assert len(haplotype) == len(recombination_rate_map)
    assert len(recombination_rate_map) == len(mutation_rate_map)

    ll_ts = tree_sequence.ll_tree_sequence

    ls_hmm = _tskit.LsHmm(
        ll_ts,
        recombination_rate=recombination_rate_map,
        mutation_rate=mutation_rate_map,
        precision=precision,
        acgt_alleles=True,  # Matrix is in ACGT (i.e. 0123) space
    )

    vm = _tskit.ViterbiMatrix(ll_ts)
    ls_hmm.viterbi_matrix(haplotype, vm)
    path = vm.traceback()

    assert len(path) == tree_sequence.num_sites
    assert np.all(np.isin(path, tree_sequence.samples())), \
        f"Some IDs in the path are not sample IDs."

    return path


def impute_by_sample_matching(ts, sd, recombination_rate, mutation_rate, precision):
    assert ts.num_sites == sd.num_sites
    assert np.all(np.isin(ts.sites_position, sd.sites_position))

    logging.info("Mapping samples to ACGT space.")
    H1 = np.zeros((ts.num_sites, sd.num_samples), dtype=np.int32)
    i = 0
    for v in tqdm(sd.variants()):
        if v.site.position in ts.sites_position:
            H1[i, :] = create_index_map(v.alleles)[v.genotypes]
            i += 1
    H1 = H1.T

    logging.info("Performing traceback.")
    # TODO: Pass in arrays without creating them.
    recombination_rate_map = np.repeat(recombination_rate, ts.num_sites, dtype=np.int32)
    mutation_rate_map = np.repeat(mutation_rate, ts.num_sites, dtype=np.int32)

    H2 = np.zeros((sd.num_samples, ts.num_sites), dtype=np.int32)
    for i in tqdm(np.arange(sd.num_samples)):
        H2[i, :] = get_traceback_path(
            tree_sequence=ts,
            haplotype=H1[i, :],
            recombination_rate_map=recombination_rate_map,
            mutation_rate_map=mutation_rate_map,
            precision=precision,
        )

    logging.info("Imputing into samples.")
    i = 0
    H3 = np.zeros((sd.num_samples, ts.num_sites), dtype=np.int32)
    for v in tqdm(ts.variants()):
        H3[:, i] = v.genotypes[H2[:, i]]
        i += 1

    return H3


def write_genotype_matrix_to_samples(
    ts,
    genotype_matrix,
    mask_site_pos,
    chip_site_pos,
    out_file,
):
    assert ts.num_sites == genotype_matrix.shape[1]

    out_file = str(out_file)
    ts_iter = ts.variants()

    i = 0  # Track iterating through `genotype_matrix`
    with tsinfer.SampleData(path=out_file) as sd:
        for ts_v in tqdm(ts_iter):
            # Set metadata
            metadata = {'marker': ''}
            if ts_v.site.position in mask_site_pos:
                metadata['marker'] = 'mask'
            elif ts_v.site.position in chip_site_pos:
                metadata['marker'] = 'chip'

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
    "-i1",
    required=True,
    help="Input trees file containing reference genomes.",
)
@click.option(
    "--in_target_samples_file",
    "-i2",
    required=True,
    help="Input samples file containing target genomes.",
)
@click.option(
    "--in_chip_file",
    "-c",
    required=True,
    help="Input tab-delimited file with chip site positions.",
)
@click.option("--out_samples_file", "-o", required=True, help="Output samples file.")
@click.option("--tmp_samples_file", default=None, help="Temporary samples file")
@click.option("--precision", "-p", type=int, default=10, help="Precision to compute likelihood values.")
def perform_imputation_by_sample_matching(
    in_reference_trees_file,
    in_target_samples_file,
    in_chip_file,
    out_samples_file,
    tmp_samples_file,
    precision,
):
    logging.info(f"Loading reference trees file {in_reference_trees_file}")
    ts_ref = tskit.load(in_reference_trees_file)
    ts_ref = ts_ref.simplify()  # Needed? Remove unary nodes... what else?
    ts_ref_site_pos = ts_ref.sites_position

    logging.info(f"Loading target samples file {in_target_samples_file}")
    sd_target = tsinfer.load(in_target_samples_file)

    logging.info(f"Loading chip position file {in_chip_file}")
    chip_site_pos_all = masks.parse_site_position_file(in_chip_file, one_based=False)

    logging.info("Defining chip and mask sites relative to the reference trees")
    ts_ref_sites_isin_chip = np.isin(
        ts_ref_site_pos,
        chip_site_pos_all,
        assume_unique=True,
    )
    chip_site_pos = ts_ref_site_pos[ts_ref_sites_isin_chip]
    mask_site_pos = ts_ref_site_pos[np.invert(ts_ref_sites_isin_chip)]

    assert (
        len(set(chip_site_pos) & set(mask_site_pos)) == 0
    ), f"Chip and mask site positions are not mutually exclusive."

    logging.info("Making samples compatible with the reference trees")
    logging.info(f"Writing compatible samples to file: {tmp_samples_file}")
    sd_compat = util.make_compatible_sample_data(
        sd_target,
        ts_ref,
        skip_unused_markers=True,
        chip_site_pos=chip_site_pos,  # Site metadata
        mask_site_pos=mask_site_pos,  # Site metadata
        path=tmp_samples_file,
    )

    logging.info("Imputing into target samples")
    gm_imputed = impute_by_sample_matching(
        ts_ref,
        sd_compat,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        precision=precision,
    )

    print(f"Printing results to samples file: {out_samples_file}")
    write_genotype_matrix_to_samples(
        ts=ts_ref,
        genotype_matrix=gm_imputed,
        chip_site_pos=chip_site_pos,
        mask_site_pos=mask_site_pos,
        out_file=out_samples_file,
    )


if __name__ == "__main__":
    perform_imputation_by_sample_matching()
