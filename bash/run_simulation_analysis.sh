pop_ref=$1
pop_query=$2
perc_mask_sites=$3

script_file="./src/run_simulations.py"

sampling_time="0"
prefix="i"

base_dir="./analysis/genealogy_only"
out_dir=$base_dir"/ten_${pop_ref}_${pop_query}_t${sampling_time}_p${perc_mask_sites}"

mkdir -p $base_dir

for i in {1..100}; do
    echo "python ${script_file} -i ${i} -s ${sampling_time} -p 0.${perc_mask_sites} -m ten_pop --pop_ref ${pop_ref} --pop_query ${pop_query} -o ${out_dir}"
done
