in_file="../analysis/tuning/out.region.ref.samples"
out_dir="../analysis/tuning/"
script_file="src/run_tsinfer_standard.py"

exponents=(-5 -4 -3 -2 -1 0 1 2 3 4)

for i in ${exponents[@]}; do
    for j in ${exponents[@]}; do
        prefix="a"$i"s"$j".1Mb"
        echo "python $script_file -r 1e-8 -a 1e$i -s 1e$j -i $in_file -o $out_dir -p $prefix -t 16"
    done
done
