base_dir="analysis/genealogy_only/ten_yri_ceu_t0_p10/"

for i in {1..100}
do
    csv_file=${base_dir}"/i"${i}".csv"
    echo "python python/main.py -i "${i}" -t 0 -p 0.10 -m ten_pop --pop_ref YRI --pop_query CEU --out_prefix "${csv_file}
done
