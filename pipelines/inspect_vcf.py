import click
import sys

sys.path.append("./python")
import read_vcf
import stats


@click.command()
@click.option(
    "--in_vcf_file",
    "-i",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
    help="Input (gzipped) VCF file.",
)
@click.option(
    "--out_csv_file",
    "-o",
    type=click.Path(file_okay=True, writable=True),
    required=True,
    help="Output CSV file with variant statistics.",
)
@click.option(
    "--seq_name",
    "-s",
    type=str,
    default=None,
    help="Sequence name to query (default = None).",
)
@click.option(
    "--left_coord",
    "-l",
    type=int,
    default=None,
    help="Left 0-based coordinate of the inclusion interval (default = None)."
    + "If None, then set to 0.",
)
@click.option(
    "--right_coord",
    "-r",
    type=int,
    default=None,
    help="Right 0-based coordinate of the inclusion interval (default = None)."
    + "If None, then set to the last coordinate in the VCF file.",
)
@click.option(
    "--num_threads", "-t", type=int, default=1, help="Number of threads (default = 1)."
)
@click.option("--verbose", "-v", is_flag=True, help="Show warnings.")
def parse_vcf_file(
    in_vcf_file, out_csv_file, seq_name, left_coord, right_coord, num_threads, verbose
):
    vcf = read_vcf.get_vcf(
        in_vcf_file,
        seq_name=seq_name,
        left_coord=left_coord,
        right_coord=right_coord,
        num_threads=num_threads,
    )
    var_stats = stats.get_variant_statistics(vcf, show_warnings=verbose)
    stats.print_variant_statistics(var_stats, out_csv_file)


if __name__ == "__main__":
    parse_vcf_file()
