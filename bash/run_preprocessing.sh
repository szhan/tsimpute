#!/usr/bin/bash

out_dir="../analysis/sisu/"

ancestral_states_file="../analysis/external/chr20_ancestral_states.fa"
end="25700000"
seq_len="25700000"
num_threads="16"


# Prepare samples files
dataset="sisu42_chr20"
out_file=${out_dir}${dataset}".samples"
python src/convert.py \
    --source generic \
    --data_file ../data/fimm/sisu_v4_2/v4.2.chr20_phased_SNPID.vcf.gz \
    --ancestral_states_file ${ancestral_states_file} \
    --output_file ${out_file} \
    --reference_name GRCh38 \
    --num_threads ${num_threads} \
    --exclude_indels \
    --progress \
    > ${out_file}".report"

dataset="sisu3_affylike_chr20"
out_file=${out_dir}${dataset}".samples"
python src/convert.py \
    --source generic \
    --data_file ../data/fimm/sisu_v3/sisu3_affylike_chr20.vcf.gz \
    --ancestral_states_file ${ancestral_states_file} \
    --output_file ${out_file} \
    --reference_name GRCh38 \
    --num_threads ${num_threads} \
    --exclude_indels \
    --progress \
    > ${out_file}".report"

dataset="sisu3_merged_nochip_chr20"
out_file=${out_dir}${dataset}".samples"
python src/convert.py \
    --source generic \
    --data_file ../data/fimm/sisu_v3/sisu3_merged_nochip_chr20.vcf.gz \
    --ancestral_states_file ${ancestral_states_file} \
    --output_file ${out_file} \
    --reference_name GRCh38 \
    --num_threads ${num_threads} \
    --exclude_indels \
    --progress \
    > ${out_file}".report"

dataset="sisu3_imputed_merged_nochip_info_chr20"
out_file=${out_dir}${dataset}".samples"
python src/convert.py \
    --source generic \
    --data_file ../data/fimm/sisu_v3/sisu3_imputed_merged_nochip_info_chr20.vcf.gz \
    --ancestral_states_file ${ancestral_states_file} \
    --output_file ${out_file} \
    --reference_name GRCh38 \
    --num_threads ${num_threads} \
    --exclude_indels \
    --progress \
    > ${out_file}".report"


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
