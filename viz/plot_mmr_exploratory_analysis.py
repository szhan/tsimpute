import pandas as pd
import matplotlib.pyplot as plt


def parse_imputation_results_file(in_file):
    """
    :param str in_file:
    :return: IQS averaged across sites.
    :rtype: float
    """
    df_full = pd.read_csv(in_file, comment="#")
    df_full = df_full[df_full["iqs"].notna()]
    df_subset = df_full[df_full["ref_minor_allele_freq"] == 0]
    assert df_subset.shape[0] == 0, \
        f"Some {df_subset.shape[0]} sites have no minor allele, " \
        f"but {df_full.shape[0]} sites should have a minor allele."
    mean_iqs = df_full["iqs"].mean(skipna=True)
    return mean_iqs


in_dir = "../analysis/tuning/imputed/"

mmr_log10 = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
results = []
for i in mmr_log10:
    for j in mmr_log10:
        prefix = "a" + str(mmr_log10[i]) + "s" + str(mmr_log10[j])
        in_file = in_dir + "/" + prefix + ".1Mb.imputation.csv"
        print(in_file)
        mean_iqs = parse_imputation_results_file(in_file)
        results.append((mmr_log10[i], mmr_log10[j], mean_iqs))

results = pd.DataFrame(results, columns=["mmr_ancestors", "mmr_samples", "mean_iqs"])

results.to_csv("imputation.aggregated.csv")
