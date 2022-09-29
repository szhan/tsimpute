in_dir="../analysis/tuning/"
out_dir="../analysis/tuning/imputed_leaves_kept/"
script_file="./src/perform_imputation.py"

in_target_file="../analysis/tuning/out.region.query.samples"
in_chip_file="../data/fimm/chip_positions_chr20.txt"

cpus="1"

exponents=(-7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7)

for i in ${exponents[@]}; do
    for j in ${exponents[@]}; do
        prefix="a"$i"s"$j".1Mb"
        in_trees_file=${in_dir}${prefix}".inferred.trees"
        echo "python $script_file -r 1e-8 -s 1e$j -i1 $in_trees_file -i2 $in_target_file -c $in_chip_file -o $out_dir -p $prefix --remove_leaves False -t $cpus"
    done
done
