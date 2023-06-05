# Original source code from here:
# https://github.com/awohns/unified_genealogy_paper/blob/51dc9130275ec93dfeba83bb3d31cf27d95e2dbb/all-data/Makefile

####################################################
# File and directory paths
####################################################
SEQ_NAME="chr20"
ANCESTRAL_STATES_FASTA=${SEQ_NAME}"_ancestral_states.fa"
ANCESTRAL_STATES_FASTA_FAI=${ANCESTRAL_STATES_FASTA}".fai"
ANCESTORS_STATS_FILE=${OUT_DIR}${ANCESTORS_TREES_FILE}".stats"

#############################################
# Ancestral states from Ensembl
#############################################
REFERENCE_NAME=GRCh38

ANCESTRAL_STATES_PREFIX=homo_sapiens_ancestor_GRCh38
ANCESTRAL_STATES_TARBALL=${ANCESTRAL_STATES_PREFIX}.tar.gz
ANCESTRAL_STATES_URL=ftp://ftp.ensembl.org/pub/release-100/fasta/ancestral_alleles/${ANCESTRAL_STATES_TARBALL}

${ANCESTRAL_STATES_TARBALL}:
		curl ${ANCESTRAL_STATES_URL} -o ${ANCESTRAL_STATES_TARBALL}

${ANCESTRAL_STATES_PREFIX}/README: ${ANCESTRAL_STATES_TARBALL}
		rm -fR ${ANCESTRAL_STATES_PREFIX}
		tar -xvzf ${ANCESTRAL_STATES_TARBALL}
		# Update access times or we'll keep rebuilding this rule. Have to make sure 
		# that the README we touch is older than the actual fa files.
		touch $@
		touch ${ANCESTRAL_STATES_PREFIX}/*.fa

chr%_ancestral_states.fa: ${ANCESTRAL_STATES_PREFIX}/README
		ln -sf ${ANCESTRAL_STATES_PREFIX}/homo_sapiens_ancestor_$*.fa $@

chr%_ancestral_states.fa.fai: chr%_ancestral_states.fa
		samtools faidx $^
