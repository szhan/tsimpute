from itertools import product
import tsinfer


samples_file_ref = "sisu42_ensembl_aa_high/chr20.samples"
samples_file_target = "sisu3/sisu3_affylike_chr20.samples.report"
samples_file_imputed = "sisu3/sisu3_imputed_merged_nochip_info_chr20.samples.report" 
samples_file_true = "sisu3/sisu3_merged_nochip_chr20.samples.report"

# Multi-allelic sites should be removed.
sd_ref = tsinfer.load(samples_file_ref)
sd_target = tsinfer.load(samples_file_target)
sd_imputed = tsinfer.load(samples_file_imputed)
sd_true = tsinfer.load(samples_file_true)

pos_ref = set(sd_ref.sites_position)
pos_target = set(sd_target.sites_position)
pos_imputed = set(sd_imputed.sites_position)
pos_true = set(sd_true.sites_position)

pos_all = pos_ref | pos_target | pos_imputed | pos_true

upset_members = []
upset_size = []

# - denotes substraction
# & denotes intersection
for op_1, op_2, op_3, op_4 in product('-&', repeat=4):
    set_members = []
    if op_1 == '&':
        set_members.append("Reference markers")
    if op_2 == '&':
        set_members.append("Target markers")
    if op_3 == '&':
        set_members.append("Imputed markers")
    if op_4 == '&':
        set_members.append("Ground-truth markers")
    set_size = eval(f"{pos_all} {op_1} {pos_ref} {op_2} {pos_target} {op_3} {pos_imputed} {op_4} {pos_true}")
    upset_members.append(set_members)
    upset_size.append(set_size)

print(set_members)
print(set_size)
