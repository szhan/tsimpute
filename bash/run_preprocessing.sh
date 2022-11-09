#!/usr/bin/bash

out_dir="../analysis/sisu/"

ancestral_states_file="../analysis/external/chr20_ancestral_states.fa"
end="25700000"
seq_len="25700000"
num_threads=16


# Prepare samples files
python src/convert.py generic \
    ../data/fimm/sisu_v4_2/v4.2.chr20_phased_SNPID.vcf.gz \
    ${ancestral_states_file} \
    ${out_dir}"sisu42_chr20.samples" \
    --reference-name GRCh38 \
    --num-threads ${num_threads} \
    > \
    ${out_dir}"sisu42_chr20.samples.report"

python src/convert.py generic \
    ../data/fimm/sisu_v3/sisu3_affylike_chr20.vcf.gz \
    ${ancestral_states_file} \
    ${out_dir}"sisu3_affylike_chr20.samples" \
    --reference-name GRCh38 \
    --num-threads ${num_threads} \
    --exclude-indels True \
    > \
    ${out_dir}"sisu3_affylike_chr20.samples.report"

python src/convert.py generic \
    ../data/fimm/sisu_v3/sisu3_merged_nochip_chr20.vcf.gz \
    ${ancestral_states_file} \
    ${out_dir}"sisu3_merged_nochip_chr20.samples" \
    --reference-name GRCh38 \
    --num-threads ${num_threads} \
    > \
    ${out_dir}"sisu3_merged_nochip_chr20.samples.report"

python src/convert.py generic \
    ../data/fimm/sisu_v3/sisu3_imputed_merged_nochip_info_chr20.vcf.gz \
    ${ancestral_states_file} \
    ${out_dir}"sisu3_imputed_merged_nochip_info_chr20.samples" \
    --reference-name GRCh38 \
    --num-threads ${num_threads} \
    > \
    ${out_dir}"sisu3_imputed_merged_nochip_info_chr20.samples.report"


# Subset the samples files by coordinate
python src/extract_samples_by_coordinates.py \
    -i ${out_dir}"sisu42_chr20.samples" \
    -o ${out_dir}"sisu42_chr20_p.samples" \
    --end ${end} \
    --seq_len ${seq_len}

python src/extract_samples_by_coordinates.py \
    -i ${out_dir}"sisu3_affylike_chr20.samples" \
    -o ${out_dir}"sisu3_affylike_chr20_p.samples" \
    --end ${end} \
    --seq_len ${seq_len}

python src/extract_samples_by_coordinates.py \
    -i ${out_dir}"sisu3_merged_nochip_chr20.samples" \
    -o ${out_dir}"sisu3_merged_nochip_chr20_p.samples" \
    --end ${end} \
    --seq_len ${seq_len}

python src/extract_samples_by_coordinates.py \
    -i ${out_dir}"sisu3_imputed_merged_nochip_info_chr20.samples" \
    -o ${out_dir}"sisu3_imputed_merged_nochip_info_chr20_p.samples" \
    --end ${end} \
    --seq_len ${seq_len}
