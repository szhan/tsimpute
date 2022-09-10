base_dir="./assets"
dest_dir=${base_dir}"/ensembl"

mkdir -p ${base_dir}
mkdir -p ${dest_dir}

base_url="http://ftp.ensembl.org/pub/release-107/variation/vcf/homo_sapiens/"
readme_url=${base_url}"README"
chk_url=${base_url}"CHECKSUMS"
gvf_url=${base_url}"homo_sapiens-chr20.vcf.gz"

cd ${dest_dir}
curl ${readme_url} -O
curl ${chk_url} -O
curl ${gvf_url} -O

sum_src=`grep 'vcf.gz' CHECKSUMS`
sum_dest=`sum *vcf.gz`

echo "SUM SRC="${sum_src}
echo "SUM DEST="${sum_dest}

cd ..
