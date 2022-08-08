import demes
import demesdraw
import msprime


in_yaml_file = "jacobs_2019.yaml"
out_svg_file = "jacobs_2019.svg"

ooa_graph = demes.load(in_yaml_file)
demographic_model = msprime.Demography.from_demes(ooa_graph)

ax = demesdraw.tubes(ooa_graph, log_time=True, labels="xticks-legend");
ax.figure.savefig(out_svg_file)

