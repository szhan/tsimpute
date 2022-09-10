base_dir="./assets"
dest_dir=${base_dir}"/ensembl"

mkdir -p ${base_dir}
mkdir -p ${dest_dir}

base_url="http://ftp.ensembl.org/pub/release-107/variation/vcf/homo_sapiens/"
vcf_file="homo_sapiens-chr20.vcf.gz"

cd ${dest_dir}
curl ${base_url}"README" -O
curl ${base_url}"CHECKSUMS" -O
curl ${base_url}"/"${vcf_file} -O

vcf_file_pattern=${vcf_file}"$"
sum_src=`grep ${vcf_file_pattern} CHECKSUMS`
sum_dest=$(sum ${vcf_file})

echo "SUM SRC="${sum_src}
echo "SUM DEST="${sum_dest}

cd ..
