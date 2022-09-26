import tsinfer
import math
import numpy as np


in_samples_file = "in.samples"
out_full_samples_file = "out.region.full.samples"
out_query_samples_file = "out.region.query.samples"
out_ref_samples_file = "out.region.ref.samples"

sd = tsinfer.load(in_samples_file)

# Subset by region
# cytoband 20.p12.1
chr = "20"
length_focal_region = 5e6 # 5 Mbp
start = 11_900_000 # 1-based, inclusive
end = start + length_focal_region # 1-based, inclusive

included_site_ids = []
for s in sd.sites():
    if s.position >= (start - 1) and s.position <= (end - 1):
        included_site_ids.append(s.id)

sd_region = sd.subset(sites=included_site_ids, path=out_full_samples_file)

print(f"Region: {chr}:{start}-{end}")
print(f"Sites: {len(included_site_ids)}")

# Subset by individuals
query_size = math.ceil(sd.num_individuals * 0.10)
query_ids = np.random.choice(sd.num_individuals, query_size, replace=False)
ref_ids = list(set(np.arange(sd.num_individuals)) - set(query_ids))

print(f"Full: {len(sd.num_individuals)}")
print(f"Query: {len(query_ids)}")
print(f"Ref: {len(ref_ids)}")

sd_region_query = sd_region.subset(individuals=query_ids, path=out_query_samples_file)
sd_region_ref = sd_region.subset(individuals=ref_ids, path=out_ref_samples_file)
