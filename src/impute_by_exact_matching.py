import click
import sys
from tqdm import tqdm

import _tskit
import tskit
import tsinfer
import numpy as np

sys.path.append("./src")
import masks
import util


def closest_match(ts, h, recombination_rate, mutation_rate):
    rho = np.zeros(ts.num_sites) + recombination_rate
    mu = np.zeros(ts.num_sites) + mutation_rate
    ls_hmm = _tskit.LsHmm(
        ts.ll_tree_sequence,
        recombination_rate=rho,
        mutation_rate=mu,
        precision=6,  # Arbitrarily chose this, gives right answers on exact matching
        acgt_alleles=True,
    )
    vm = _tskit.ViterbiMatrix(ts.ll_tree_sequence)
    ls_hmm.viterbi_matrix(h, vm)
    path = vm.traceback()
    return path


def create_index_map(x):
    alleles = ["A", "C", "G", "T", None]
    map_ACGT = [alleles.index(x[i]) for i in range(len(x))]
    if None in x:
        map_ACGT[-1] = -1
    map_ACGT = np.array(map_ACGT)
    return map_ACGT


def impute_by_exact_matching(ts, sd, recombination_rate, mutation_rate):
    assert ts.num_sites == sd.num_sites
    assert np.all(np.isin(ts.sites_position, sd.sites_position))

    print("INFO: Mapping samples to ACGT space.")
    H1 = np.zeros((ts.num_sites, sd.num_samples), dtype=np.int32)
    i = 0
    for v in tqdm(sd.variants()):
        if v.site.position in ts.sites_position:
            H1[i, :] = create_index_map(v.alleles)[v.genotypes]
            i += 1
    H1 = H1.T

    print("INFO: Performing closest match.")
    H2 = np.zeros((sd.num_samples, ts.num_sites), dtype=np.int32)
    for i in tqdm(np.arange(sd.num_samples)):
        H2[i, :] = closest_match(
            ts,
            H1[i, :],
            recombination_rate=recombination_rate,
            mutation_rate=mutation_rate,
        )

    print("INFO: Imputing into samples.")
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
            marker_type = ""
            if ts_v.site.position in mask_site_pos:
                marker_type = "mask"
            elif ts_v.site.position in chip_site_pos:
                marker_type = "chip"
            metadata = {"marker": marker_type}
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
def perform_imputation_by_exact_matching(
    in_reference_trees_file,
    in_target_samples_file,
    in_chip_file,
    out_samples_file,
    tmp_samples_file,
):
    print("INFO: Loading trees file containing reference genomes")
    print(f"INFO: {in_reference_trees_file}")
    ts_ref = tskit.load(in_reference_trees_file)
    ts_ref = ts_ref.simplify()  # Needed? Remove unary nodes... what else?
    ts_ref_site_pos = ts_ref.sites_position

    print("INFO: Loading samples file containing target genomes")
    print(f"INFO: {in_target_samples_file}")
    sd_target = tsinfer.load(in_target_samples_file)

    print("INFO: Loading chip position file")
    print(f"INFO: {in_chip_file}")
    chip_site_pos = masks.parse_site_position_file(in_chip_file)

    print("INFO: Making samples compatible with the reference trees")
    print(f"INFO: {tmp_samples_file}")
    sd_compat = util.make_compatible_sample_data(
        sd_target, ts_ref, skip_unused_markers=True, path=tmp_samples_file
    )

    print("INFO: Defining chip and mask sites relative to the reference trees")
    chip_site_pos_all = masks.parse_site_position_file(in_chip_file, one_based=False)
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

    print("INFO: Imputing into target samples")
    gm_imputed = impute_by_exact_matching(
        ts_ref, sd_compat, recombination_rate=1e-8, mutation_rate=1e-8
    )

    print("INFO: Printing results to samples file")
    print(f"INFO: {out_samples_file}")
    write_genotype_matrix_to_samples(
        ts=ts_ref,
        genotype_matrix=gm_imputed,
        mask_site_pos=mask_site_pos,
        chip_site_pos=chip_site_pos,
        out_file=out_samples_file,
    )


if __name__ == "__main__":
    perform_imputation_by_exact_matching()
