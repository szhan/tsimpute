base_dir="analysis/genealogy_only/ten_ceu_yri_t0_p10/"

for i in {1..100}
do
    prefix=${base_dir}"/i"
    echo "python python/main.py -i "${i}" -t 0 -p 0.10 -m ten_pop --pop_ref CEU --pop_query YRI --out_prefix "${prefix}
done
