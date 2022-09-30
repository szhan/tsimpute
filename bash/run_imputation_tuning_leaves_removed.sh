in_dir="../analysis/tuning/"
out_dir="../analysis/tuning/imputed_leaves_removed/"
remove_leaves="True"

cpus="1"

impute_script_file="./src/perform_imputation.py"
evaluate_script_file="./src/evaluate_imputation.py"

in_query_file="../analysis/tuning/out.region.query.samples"
in_chip_file="../data/fimm/chip_positions_chr20.txt"

exponents=(-7 -6 -5 -4 -3 -2 -1 0 1 2 3 4)

for i in ${exponents[@]}; do
    for j in ${exponents[@]}; do
        prefix="a"$i"s"$j".1Mb"
        in_trees_file=${in_dir}${prefix}".inferred.trees"
        out_trees_file=${out_dir}${prefix}".imputed.trees"
        out_csv_file=${out_dir}${prefix}".imputation.csv"
        if [ ! -f $out_csv_file ]; then
            echo "python $impute_script_file -i1 $in_trees_file -i2 $in_query_file -c $in_chip_file -o $out_dir -p $prefix -r 1e-8 -s 1e$j --remove_leaves $remove_leaves -t $cpus"
            echo "python $evaluate_script_file -i1 $out_trees_file -i2 $in_query_file -r $in_trees_file --remove_leaves $remove_leaves -c $in_chip_file -o $out_csv_file"
        fi
    done
done
