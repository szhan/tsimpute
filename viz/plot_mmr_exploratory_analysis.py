from unittest import skip
import pandas as pd
import matplotlib.pyplot as plt


def parse_imputation_results_file(in_file):
    """
    :param str in_file:
    :return: IQS averaged across sites.
    :rtype: float
    """
    df = pd.read_csv(in_file)
    assert df[df["ref_minor_allele_freq"] > 0].shape[0] == df.shape[0]
    mean_iqs = df["iqs"].mean(skipna=True)
    return mean_iqs


in_dir = "../analysis/tuning/imputed/"

mmr_log10 = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
results = []
for i in mmr_log10:
    for j in mmr_log10:
        prefix = "a" + mmr_log10[i] + "s" + mmr_log10[j]
        in_file = in_dir + "/" + prefix + ".imputation.csv"
        print(in_file)
        mean_iqs = parse_imputation_results_file(in_file)
        results.append((str(mmr_log10[i]), str(mmr_log10[j]), mean_iqs))

results = pd.DataFrame(results, columns=["mmr_ancestors", "mmr_samples", "mean_iqs"])
print(results)

#grid_data = mean_vals.pivot("mismatch_ancestors", "mismatch_samples", "mean_iqs")
#plt.contourf(grid_data.columns, grid_data.index, grid_data, 20, cmap='viridis')
#plt.xscale("log")
#plt.xlabel(grid_data.columns.name)
#plt.yscale("log")
#plt.ylabel(grid_data.index.name)
#plt.show()
