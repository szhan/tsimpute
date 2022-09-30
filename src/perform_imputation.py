import click
from datetime import datetime
import sys
import warnings
import numpy as np
import tskit
import tsinfer

sys.path.append("./src")
import masks
import util


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
def run_pipeline(
    in_reference_trees_file,
    in_target_samples_file,
    in_chip_file,
    out_dir,
    out_prefix,
    recombination_rate,
    mmr_samples,
    remove_leaves,
    num_threads,
):
    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"INFO: START {start_datetime}")

    print("INFO: Loading input files")
    ts_ref = tskit.load(in_reference_trees_file)
    sd_target = tsinfer.load(in_target_samples_file)

    chip_site_pos = masks.parse_site_position_file(in_chip_file)

    print("INFO: Making ancestors tree sequence")
    if tsinfer.__version__ == "0.2.4.dev27+gd61ae2f":
        ts_anc = tsinfer.eval_util.make_ancestors_ts(
            ts=ts_ref, remove_leaves=remove_leaves
        )
    else:
        # The samples argument is not actually used.
        ts_anc = tsinfer.eval_util.make_ancestors_ts(
            samples=None, ts=ts_ref, remove_leaves=remove_leaves
        )

    print("INFO: Making samples compatible with the ancestors tree sequence")
    sd_compat = util.make_compatible_sample_data(sd_target, ts_anc)

    warnings.warn("Defining masked site positions relative to the reference panel.")
    ts_anc_sites_isnotin_chip = np.isin(
        ts_anc.sites_position, chip_site_pos, assume_unique=True, invert=True
    )
    mask_site_pos = ts_anc.sites_position[ts_anc_sites_isnotin_chip]

    assert (
        len(set(chip_site_pos) & set(mask_site_pos)) == 0
    ), f"Chip and mask site positions are not mutually exclusive."

    print("INFO: Masking sites")
    sd_masked = masks.mask_sites_in_sample_data(
        sample_data=sd_compat, sites=mask_site_pos, site_type="position"
    )

    print("INFO: Imputing target samples")
    ts_imputed = tsinfer.match_samples(
        sample_data=sd_masked,
        ancestors_ts=ts_anc,
        recombination_rate=recombination_rate,
        mismatch_ratio=mmr_samples,
        num_threads=num_threads,
    )

    out_trees_file = out_dir + "/" + out_prefix + ".imputed.trees"
    ts_imputed.dump(out_trees_file)

    error_msg = f"Different number of sites in the tree sequences and sample data."
    assert ts_ref.num_sites == sd_compat.num_sites, error_msg
    assert ts_ref.num_sites == sd_masked.num_sites, error_msg
    assert ts_ref.num_sites == ts_imputed.num_sites, error_msg

    end_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"INFO: END {end_datetime}")


if __name__ == "__main__":
    run_pipeline()
