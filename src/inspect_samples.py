import click
import tsinfer


@click.command()
@click.option(
    "--in_samples_file",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input samples file",
)
def get_variant_statistics_from_samples_file(in_samples_file):
    # Multiallelic sites
    num_sites_multiallelic = 0
    # Biallelic sites where the AA is an indel, but the derived allele is a SNV.
    num_sites_ancestral_state_indel = 0
    # Biallelic sites where the derived allele is an indel, but the AA is a SNV.
    num_sites_derived_state_indel = 0
    # Biallelic sites where both the ancestral and derived alleles are indels.
    num_sites_both_states_indel = 0
    # Biallelic sites with no indels
    num_sites_only_snvs = 0

    sd = tsinfer.load(in_samples_file)
    for v in sd.variants():
        if len(v.alleles) > 2:
            num_sites_multiallelic += 1
        else:
            if len(v.alleles[0]) > 1 and len(v.alleles[1]) > 1:
                num_sites_both_states_indel += 1
            elif len(v.alleles[0]) > 1:
                num_sites_ancestral_state_indel += 1
            elif len(v.alleles[1]) > 1:
                num_sites_derived_state_indel += 1
            else:
                num_sites_only_snvs += 1

    print(f"Sites multiallelic: {num_sites_multiallelic}")
    print(f"Sites ancestral indel: {num_sites_ancestral_state_indel}")
    print(f"Sites derived indel: {num_sites_derived_state_indel}")
    print(f"Sites both indels {num_sites_both_states_indel}")
    print(f"Sites only SNVs: {num_sites_only_snvs}")


if __name__ == "__main__":
    get_variant_statistics_from_samples_file()
