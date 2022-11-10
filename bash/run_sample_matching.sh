$!/usr/bin/bash


in_chip_file="../data/fimm/chip_positions_chr20.txt"
in_map_file="./assets/recomb-hg38/genetic_map_GRCh38_chr20_p.txt"
in_dir="../analysis/sisu/"
out_dir="../analysis/sisu/"


time python src/perform_imputation_by_sample_matching.py \
    -i1 ${in_dir}"sisu42_chr20_p.samples" \
    -i2 ${in_dir}"sisu3_affylike_chr20_p.samples" \
    -c ${in_chip_file} \
    -o ${out_dir} \
    -p chr20_p.sample_matched.precision10 \
    --precision 10

time python src/perform_imputation_by_sample_matching.py \
    -i1 ${in_dir}"sisu42_chr20_p.samples" \
    -i2 ${in_dir}"sisu3_affylike_chr20_p.samples" \
    -c ${in_chip_file} \
    -o ${out_dir} \
    -p chr20_p.sample_matched.genetic_map.precision10 \
    -g ${in_map_file} \
    --precision 10
