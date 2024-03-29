from datetime import datetime
import click
import logging
from pathlib import Path
import sys
from git import Repo
import numpy as np
import msprime
import tskit
import tsinfer

sys.path.append("./")
from src import masks
from src import util


@click.command()
@click.option(
    "--in_reference_trees_file",
    "-i1",
    type=click.Path(exists=True),
    required=True,
    help="Input tree sequence file with reference samples.",
)
@click.option(
    "--in_target_samples_file",
    "-i2",
    type=click.Path(exists=True),
    required=True,
    help="Input samples file with target samples.",
)
@click.option(
    "--in_chip_file",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Input list of positions to retain before imputation.",
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
    "--recombination_rate",
    "-r",
    type=float,
    default=None,
    help="Uniform recombination rate",
)
@click.option(
    "--genetic_map",
    "-g",
    type=click.Path(exists=True),
    default=None,
    help="Genetic map file in HapMap3 format",
)
@click.option(
    "--mmr_samples",
    "-s",
    type=float,
    default=None,
    help="Mismatch ratio used when matching sample haplotypes",
)
@click.option(
    "--remove_leaves",
    type=bool,
    default=False,
    help="Remove leaves when generate an ancestors tree?",
)
@click.option("--num_threads", "-t", type=int, default=1, help="Number of CPUs.")
def perform_imputation(
    in_reference_trees_file,
    in_target_samples_file,
    in_chip_file,
    out_dir,
    out_prefix,
    recombination_rate,
    genetic_map,
    mmr_samples,
    remove_leaves,
    num_threads,
):
    out_dir = Path(out_dir)
    log_file = out_dir / f"{out_prefix}.log"
    compat_samples_file = out_dir / f"{out_prefix}.compat.samples"
    out_trees_file = out_dir / f"{out_prefix}.imputed.trees"

    logging.basicConfig(filename=str(log_file), encoding="utf-8", level=logging.INFO)

    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"start: {start_datetime}")
    logging.info(f"dep: msprime {msprime.__version__}")
    logging.info(f"dep: tskit {tskit.__version__}")
    logging.info(f"dep: tsinfer {tsinfer.__version__}")
    repo = Repo(search_parent_directories=True)
    logging.info(f"dep: tsimpute URL {repo.remotes.origin.url}")
    logging.info(f"dep: tsimpute SHA {repo.head.object.hexsha}")

    logging.info(f"Loading reference trees file: {in_reference_trees_file}")
    ts_ref = tskit.load(in_reference_trees_file)

    logging.info(f"Loading target samples file: {in_target_samples_file}")
    sd_target = tsinfer.load(in_target_samples_file)

    if genetic_map is not None:
        logging.info(f"Loading genetic map: {genetic_map}")
        logging.info("Using these recombination rates instead of a uniform rate")
        recombination_rate = msprime.RateMap.read_hapmap(genetic_map)

    logging.info("Making ancestors trees from the reference trees")
    ts_anc = tsinfer.eval_util.make_ancestors_ts(ts=ts_ref, remove_leaves=remove_leaves)

    logging.info(f"Loading chip position file: {in_chip_file}")
    chip_site_pos_all = masks.parse_site_position_file(in_chip_file, one_based=False)

    logging.info("Defining chip and mask sites relative to the ancestors trees")
    ts_anc_sites_isin_chip = np.isin(
        ts_anc.sites_position,
        chip_site_pos_all,
        assume_unique=True,
    )
    chip_site_pos = ts_anc.sites_position[ts_anc_sites_isin_chip]
    mask_site_pos = ts_anc.sites_position[np.invert(ts_anc_sites_isin_chip)]

    logging.info("Making samples compatible with the ancestors trees")
    logging.info(f"Writing compatible samples to file: {compat_samples_file}")
    sd_compat = util.make_compatible_sample_data(
        sample_data=sd_target,
        ancestors_ts=ts_anc,
        skip_unused_markers=True,
        chip_site_pos=chip_site_pos,
        mask_site_pos=mask_site_pos,
        path=str(compat_samples_file),
    )

    assert (
        len(set(chip_site_pos) & set(mask_site_pos)) == 0
    ), f"Chip and mask site positions are not mutually exclusive."

    logging.info(f"Masking sites in target samples")
    # TODO: Check that sd_compat and sd_mask are identical if sd_target contains chip sites only.
    sd_masked = masks.mask_sites_in_sample_data(
        sample_data=sd_compat, sites=mask_site_pos, site_type="position"
    )

    logging.info(f"Imputing into target samples")
    ts_imputed = tsinfer.match_samples(
        sample_data=sd_masked,
        ancestors_ts=ts_anc,
        recombination_rate=recombination_rate,
        mismatch_ratio=mmr_samples,
        num_threads=num_threads,
    )

    logging.info(f"Writing imputed trees to file: {out_trees_file}")
    ts_imputed.dump(str(out_trees_file))

    end_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"end: {end_datetime}")


if __name__ == "__main__":
    perform_imputation()
