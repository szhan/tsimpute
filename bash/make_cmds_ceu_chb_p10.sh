base_dir="analysis/genealogy_only/ten_ceu_chb_t0_p10/"

for i in {1..100}
do
    csv_file=${base_dir}"/i"${i}".csv"
    echo "python python/main.py -i "${i}" -t 0 -p 0.10 -m ten_pop --pop_ref CEU --pop_query CHB --out_prefix "${csv_file}
done
