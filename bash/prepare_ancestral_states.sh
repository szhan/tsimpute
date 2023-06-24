# Original source code from here:
# https://github.com/awohns/unified_genealogy_paper/blob/51dc9130275ec93dfeba83bb3d31cf27d95e2dbb/all-data/Makefile

####################################################
# File and directory paths
####################################################
CHR="20"
SEQ_NAME="chr"${CHR}
REFERENCE_NAME=GRCh38

ANCESTRAL_STATES_FASTA=${SEQ_NAME}"_ancestral_states.fa"
ANCESTRAL_STATES_FASTA_FAI=${ANCESTRAL_STATES_FASTA}".fai"
ANCESTORS_STATS_FILE=${OUT_DIR}${ANCESTORS_TREES_FILE}".stats"

#############################################
# Ancestral states from Ensembl
#############################################
ANCESTRAL_STATES_PREFIX=homo_sapiens_ancestor_${REFERENCE_NAME}
ANCESTRAL_STATES_TARBALL=${ANCESTRAL_STATES_PREFIX}.tar.gz
ANCESTRAL_STATES_URL=ftp://ftp.ensembl.org/pub/release-100/fasta/ancestral_alleles/${ANCESTRAL_STATES_TARBALL}

curl ${ANCESTRAL_STATES_URL} -o ${ANCESTRAL_STATES_TARBALL}
ln -sf ${ANCESTRAL_STATES_PREFIX}/homo_sapiens_ancestor_${CHR}.fa ${ANCESTRAL_STATES_FASTA}
samtools faidx ${ANCESTRAL_STATES_FASTA}
