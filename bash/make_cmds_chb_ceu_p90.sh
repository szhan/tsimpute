base_dir="analysis/genealogy_only/ten_chb_ceu_t0_p90/"

for i in {1..100}
do
    csv_file=${base_dir}"/i"${i}".csv"
    echo "python python/main.py -i "${i}" -t 0 -p 0.90 -m ten_pop --pop_ref CHB --pop_query CEU --out_prefix "${csv_file}
done
