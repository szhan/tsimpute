base_dir=$1

ref_dir=${base_dir}"ref/"
miss_dir=${base_dir}"miss/"
true_dir=${base_dir}"true/"
ts_anc_ref_dir=${base_dir}"ts_anc_ref/"
imputed_tsinfer_dir=${base_dir}"imputed_tsinfer/"
imputed_tsonly_dir=${base_dir}"imputed_tsonly/"
imputed_beagle_dir=${base_dir}"imputed_beagle/"

mkdir -p ${base_dir}
mkdir -p ${ref_dir}
mkdir -p ${miss_dir}
mkdir -p ${true_dir}
mkdir -p ${ts_anc_ref_dir}
mkdir -p ${imputed_tsinfer_dir}
mkdir -p ${imputed_tsonly_dir}
mkdir -p ${imputed_beagle_dir}

