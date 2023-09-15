# Default settings
#time python src/run_tsinfer.py -i ../analysis/sisu42_ensembl_aa_high/chr20_p.samples -o ../analysis/default/ -p chr20_p -t 16
# Uniform recombination rate
#time python src/run_tsinfer.py -i ../analysis/sisu42_ensembl_aa_high/chr20_p.samples -o ../analysis/uniform/ -p chr20_p -r 1e-8 -a 1e-7 -s 1e-7 -t 16
# Genetic map
time python src/run_tsinfer.py -i ../analysis/sisu42_ensembl_aa_high/chr20_p.samples -o ../analysis/map/ -p chr20_p -g assets/recomb-hg38/genetic_map_GRCh38_chr20_p_arm.txt -a 1e-7 -s 1e-7 -t 16
# Genetic map; ancestors truncated
time python src/run_tsinfer.py -i ../analysis/sisu42_ensembl_aa_high/chr20_p.samples -o ../analysis/map_truncated/ -p chr20_p -g assets/recomb-hg38/genetic_map_GRCh38_chr20_p_arm.txt --truncate_ancestors -a 1e-7 -s 1e-7 -t 16
