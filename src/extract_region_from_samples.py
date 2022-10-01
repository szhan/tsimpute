import click
import tsinfer
import numpy as np


@click.command()
@click.option(
    "--in_samples_file",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input samples file",
)
@click.option(
    "--out_samples_file",
    "-o",
    type=click.Path(exists=False),
    required=True,
    help="Output samples file",
)
@click.option("--end", type=int, required=True, help="Right coordinate (1-based)")
def extract_sites_from_region(in_samples_file, out_samples_file, end):
    sd = tsinfer.load(in_samples_file)
    sd_subset_site_pos = sd.sites_position[:][sd.sites_position[:] < end]
    sd_subset = sd.subset(
        sites=np.arange(len(sd_subset_site_pos)), path=out_samples_file
    )
    return sd_subset


if __name__ == "__main__":
    extract_sites_from_region()
