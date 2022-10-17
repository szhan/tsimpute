import click
import numpy as np
import tsinfer


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
@click.option(
    "--sequence_length",
    type=float,
    default=None,
    help="Set sequence length manually. If None, then set to 1 + end.",
)
def extract_samples_by_coordinates(in_samples_file, out_samples_file, end, sequence_length):
    """
    TODO: Subset using start coordinate.

    :return: Subsetted samples.
    :rtype: tsinfer.SampleData
    """
    if sequence_length is not None:
        assert end < sequence_length

    sd = tsinfer.load(in_samples_file)

    sd_subset_site_pos = sd.sites_position[:][sd.sites_position[:] < end]

    sd_subset = sd.subset(
        sites=np.arange(len(sd_subset_site_pos)),
        sequence_length=sequence_length,
        path=out_samples_file,
    )

    return sd_subset


if __name__ == "__main__":
    extract_samples_by_coordinates()
