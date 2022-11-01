from pathlib import Path
import sys

import _tskit
import tskit
import tsinfer
import numpy as np

sys.path.append("./src")
import masks
import util


def closest_match(ts, h, recombination_rate=1e-8, mutation_rate=1e-8):
    rho = np.zeros(ts.num_sites) + recombination_rate
    mu = np.zeros(ts.num_sites) + mutation_rate
    ls_hmm = _tskit.LsHmm(
        ts.ll_tree_sequence,
        recombination_rate=rho,
        mutation_rate=mu,
        precision=6, # Arbitrarily chose this, gives right answers on exact matching
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


def impute_by_exact_matching(ts, sd):
    assert np.all(np.isin(ts.sites_position, sd.sites_position))
    
    # Get genotype matrix from target genomes in ACGT space
    H1 = np.zeros((ts.num_sites, sd.num_samples), dtype=np.int32)
    for i, v in enumerate(sd.variants()):
        if v.site.position in ts.sites_position:
            H1[i, :] = create_index_map(v.alleles)[v.genotypes]
    H1 = H1.T

    # Get HMM paths
    H2 = np.zeros((sd.num_samples, ts.num_sites), dtype=np.int32)
    for i in np.arange(sd.num_samples):
        H2[i, :] = closest_match(ts, H1[i, :])

    # Get genotype matrix in 01 space by mapping HMM paths to reference tree
    # This is imputing from the reference genomes to the target genomes.
    H3 = np.zeros((sd.num_samples, ts.num_sites), dtype=np.int32)
    for i, v in enumerate(ts.variants()):
        H3[:, i] = v.genotypes[H2[:, i]]

    return H3


def write_genotype_matrix_to_samples(
    ts,
    gm,
    mask_site_pos,
    chip_site_pos,
    out_file,
):
    assert ts.num_sites == gm.shape[1]
    out_file = str(out_file)
    ts_iter = ts.variants()
    i = 0
    with tsinfer.SampleData(path=out_file) as sd:
        for ts_v in ts_iter:
            # Set metadata
            marker_type = ''
            if ts_v.site.position in mask_site_pos:
                marker_type = 'mask'
            elif ts_v.site.position in chip_site_pos:
                marker_type = 'chip'
            metadata = {'marker': marker_type}
            # Add site
            sd.add_site(
                position=ts_v.site.position,
                genotypes=gm[:, i],
                alleles=ts_v.alleles,
                metadata=metadata
            )
            i += 1


base_dir = Path("/Users/szhan/Projects/tsimpute/analysis/test")
in_reference_trees_file = base_dir / "single_panmictic_simulated.trees"
in_target_samples_file = base_dir / "single_panmictic_simulated.samples"
in_chip_file = base_dir / "chip.txt"
out_samples_file = base_dir / "test.samples"

print("INFO: Loading trees file containing reference genomes")
print(f"INFO: {in_reference_trees_file}")
ts_ref = tskit.load(in_reference_trees_file)
ts_ref = ts_ref.simplify()  # Needed? Remove unary nodes... what else?

print("INFO: Loading samples file containing target genomes")
print(f"INFO: {in_target_samples_file}")
sd_target = tsinfer.load(in_target_samples_file)

print("INFO: Loading chip position file")
print(f"INFO: {in_chip_file}")
chip_site_pos = masks.parse_site_position_file(in_chip_file)

print("INFO: Making samples compatible with the reference trees")
sd_compat = util.make_compatible_sample_data(sd_target, ts_ref)

print("INFO: Defining mask sites relative to the reference trees")
ts_ref_sites_isnotin_chip = np.isin(
    ts_ref.sites_position, chip_site_pos, assume_unique=True, invert=True
)
mask_site_pos = ts_ref.sites_position[ts_ref_sites_isnotin_chip]

assert (
    len(set(chip_site_pos) & set(mask_site_pos)) == 0
), f"Chip and mask site positions are not mutually exclusive."

print("INFO: Imputing into target samples")
gm_imputed = impute_by_exact_matching(ts_ref, sd_compat)

print("INFO: Printing results to samples file")
print(f"INFO: {out_samples_file}")
write_genotype_matrix_to_samples(
    ts=ts_ref,
    gm=gm_imputed,
    mask_site_pos=mask_site_pos,
    chip_site_pos=chip_site_pos,
    out_file=out_samples_file,
)