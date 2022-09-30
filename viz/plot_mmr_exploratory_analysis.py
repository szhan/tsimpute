from importlib.metadata import requires
from typing_extensions import Required
import click
import pandas as pd
import numpy as np


def parse_imputation_results_file(in_file):
    """
    :param str in_file:
    :return: IQS averaged across sites.
    :rtype: float
    """
    df = pd.read_csv(in_file, comment="#")
    df = df[df["iqs"].notna()]

    df_subset_0 = df[df["ref_minor_allele_freq"] == 0]
    df_subset_1 = df[df["ref_minor_allele_freq"] == 1]
    assert df_subset_0.shape[0] == 0, (
        f"Some {df_subset_0.shape[0]} sites have no minor allele, "
        f"but {df.shape[0]} sites should have a minor allele."
    )
    assert df_subset_1.shape[0] == 0, (
        f"Some {df_subset_1.shape[0]} sites have no minor allele, "
        f"but {df.shape[0]} sites should have a minor allele."
    )

    mean_iqs = df["iqs"].mean()
    median_iqs = df["iqs"].median()
    perc_et_100_iqs = float(np.sum(df["iqs"] == 1)) / float(df.shape[0])
    perc_gt_90_iqs = float(np.sum(df["iqs"] >= 0.90)) / float(df.shape[0])
    perc_gt_80_iqs = float(np.sum(df["iqs"] >= 0.80)) / float(df.shape[0])
    perc_gt_70_iqs = float(np.sum(df["iqs"] >= 0.70)) / float(df.shape[0])

    return [
        mean_iqs,
        median_iqs,
        perc_et_100_iqs,
        perc_gt_90_iqs,
        perc_gt_80_iqs,
        perc_gt_70_iqs,
    ]


@click.command()
@click.option(
    "--in_dir",
    type=click.Path(exists=True),
    required=True,
    help="Input directory containing imputation result CSV files.",
)
@click.option(
    "--out_file", type=click.Path(exists=False), required=True, help="Output CSV file."
)
def aggregate_imputation_results(in_dir, out_file):
    mmr_log10 = np.arange(-7, 5)
    results = []
    for i in mmr_log10:
        for j in mmr_log10:
            prefix = "a" + str(mmr_log10[i]) + "s" + str(mmr_log10[j])
            in_file = in_dir + "/" + prefix + ".1Mb.imputation.csv"
            (
                mean_iqs,
                median_iqs,
                perc_et_100_iqs,
                perc_gt_90_iqs,
                perc_gt_80_iqs,
                perc_gt_70_iqs,
            ) = parse_imputation_results_file(in_file)
            results.append(
                (
                    mmr_log10[i],
                    mmr_log10[j],
                    mean_iqs,
                    median_iqs,
                    perc_et_100_iqs,
                    perc_gt_90_iqs,
                    perc_gt_80_iqs,
                    perc_gt_70_iqs,
                )
            )
    results = pd.DataFrame(
        results,
        columns=[
            "index",
            "mmr_ancestors",
            "mmr_samples",
            "mean_iqs",
            "median_iqs",
            "perc_et_100_iqs",
            "perc_gt_90_iqs",
            "perc_gt_80_iqs",
            "perc_gt_70_iqs",
        ],
    )
    results.to_csv(out_file)


if __name__ == "__main__":
    aggregate_imputation_results()
