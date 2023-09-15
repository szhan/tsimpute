base_dir="./assets"
dest_dir=${base_dir}"/ensembl"

mkdir -p ${base_dir}
mkdir -p ${dest_dir}

base_url="http://ftp.ensembl.org/pub/release-107/variation/vcf/homo_sapiens/"
vcf_file="homo_sapiens-chr20.vcf.gz"
csi_file=${vcf_file}".csi"

cd ${dest_dir}
curl ${base_url}"README" -O
curl ${base_url}"CHECKSUMS" -O
curl ${base_url}"/"${vcf_file} -O
curl ${base_url}"/"${csi_file} -O

vcf_file_pattern=${vcf_file}"$"
csi_file_pattern=${csi_file}"$"

sum_vcf_src=`grep ${vcf_file_pattern} CHECKSUMS`
sum_csi_src=`grep ${csi_file_pattern} CHECKSUMS`
sum_vcf_dest=$(sum ${vcf_file})
sum_csi_dest=$(sum ${csi_file})

echo "SUM VCF SRC="${sum_vcf_src}
echo "SUM VCF DEST="${sum_vcf_dest}
echo "SUM CSI SRC="${sum_csi_src}
echo "SUM CSI DEST="${sum_csi_dest}

cd ..
