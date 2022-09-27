in_dir="../analysis/tuning/"
out_dir="../analysis/tuning/imputed/"
script_file="src/perform_imputation.py"

exponents=(-5 -4 -3 -2 -1 0 1 2 3 4)

for i in ${exponents[@]}; do
    for j in ${exponents[@]}; do
        prefix="a"$i"s"$j".1Mb"
        in_file=${in_dir}${prefix}".inferred.trees"
        echo "python $script_file -r 1e-8 -s 1e$j -i $in_file -o $out_dir -p $prefix -t 16"
    done
done
