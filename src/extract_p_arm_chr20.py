import tsinfer
import numpy as np


#chr20 p-arm
#0-25700000

sd = tsinfer.load("../analysis/sisu42_ensembl_aa_high/chr20.samples")

sd_p_site_pos = sd.sites_position[:][sd.sites_position[:] <= 25700000]
len(sd_p_site_pos)                         

sd_p = sd.subset(sites=np.arange(len(sd_p_site_pos)))
