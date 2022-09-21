import tsinfer


samples_file = "test.samples"
sd = tsinfer.load(samples_file)

# cytoband 20.p12.1
chr = "20"
start = 11_900_000 # 1-based, inclusive
end = 17_800_000 # 1-based, inclusive

included_site_ids = []
for s in sd.sites():
    if s.position >= (start - 1) and s.position <= (end - 1):
        included_site_ids.append(s.id)

sd_region = sd.subset(included_site_ids)

print(f"Region: {chr}:{start}-{end}")
print(f"Sites: {len(sd_region)}")
