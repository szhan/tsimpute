base_dir="./assets"
dest_dir=${base_dir}"/ensembl"

mkdir -p ${base_dir}
mkdir -p ${dest_dir}

readme_url="http://ftp.ensembl.org/pub/release-107/variation/gvf/homo_sapiens/README"
chk_url="http://ftp.ensembl.org/pub/release-107/variation/gvf/homo_sapiens/CHECKSUMS"
gvf_url="http://ftp.ensembl.org/pub/release-107/variation/gvf/homo_sapiens/homo_sapiens-chr20.gvf.gz"

cd ${dest_dir}
curl ${readme_url} -O
curl ${chk_url} -O
curl ${gvf_url} -O

sum_src=`grep 'README' CHECKSUMS`
sum_dest=`sum README`

echo "SUM SRC="${sum_src}
echo "SUM DEST="${sum_dest}

cd ..
