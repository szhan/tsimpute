NUM_THREADS ?= 0

# Requirements: bcftools, tabix, convertf, plink, samtools, python3
# See requirements.txt for Python package requirements.
# Install required software using tools/Makefile
#
# Original source code from here:
# https://github.com/awohns/unified_genealogy_paper/blob/51dc9130275ec93dfeba83bb3d31cf27d95e2dbb/all-data/Makefile
#
help:
	@echo Makefile to create trees from FinnGen/FIMM genomic data

all: finns_chr20.trees

# Save all intermediate files
.SECONDARY:

# Allow filtering in prerequisites
.SECONDEXPANSION:

####################################################
# File and directory paths
####################################################
SEQ_NAME="chr20"
PREFIX="v4.2."${SEQ_NAME} # SiSu v4.2
IN_DIR="../data/fimm/"
OUT_DIR="../analysis/finns/"${PREFIX}"/"
VCF_FILE=${IN_DIR}${PREFIX}"_phased_SNPID.vcf.gz"
ANCESTRAL_STATES_FASTA=${SEQ_NAME}"_ancestral_states.fa"
ANCESTRAL_STATES_FASTA_FAI=${ANCESTRAL_STATES_FASTA}".fai"
SAMPLES_FILE=${OUT_DIR}${PREFIX}".samples"
ANCESTORS_TREES_FILE=${OUT_DIR}${PREFIX}".ancestors.trees"
SAMPLES_TREES_FILE=${OUT_DIR}${PREFIX}".samples.trees"
ANCESTORS_STATS_FILE=${OUT_DIR}${ANCESTORS_TREES_FILE}".stats"
SAMPLES_STATS_FILE=${OUT_DIR}${SAMPLES_TREES_FILE}".stats"

#############################################
# Ancestral states from Ensembl
#############################################

# Recorded in the sample file provenance.
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

####################################################
# Standard pipeline from VCF to .trees
####################################################

${SAMPLES_FILE}: ${VCF_FILE} ${ANCESTRAL_STATES_FASTA_FAI}
		tabix -f -p vcf $<
		python python/convert.py generic -p \
				${VCF_FILE} \
				${ANCESTRAL_STATES_FASTA} \
				-m None \
				--ancestral-states-url=${ANCESTRAL_STATES_URL} \
				--reference-name=${REFERENCE_NAME} \
				--num-threads=${NUM_THREADS} \
				$@ > $@.report
		echo "Preparing samples file from VCF file"

${SAMPLES_TREES_FILE}: ${SAMPLES_FILE}
		python pipelines/run_tsinfer_standard.py \
			-i ${SAMPLES_FILE} \
			-o ${OUT_DIR} \
			-p ${PREFIX} \
			-t ${NUM_THREADS}
		echo "Inferring trees"

${ANCESTORS_STATS_FILE}: ${ANCESTORS_TREES_FILE}
		python pipelines/inspect_trees.py \
			-i ${ANCESTORS_TREES_FILE}
			-o $@

${SAMPLES_STATS_FILE}: ${SAMPLES_TREES_FILE}
		python pipelines/inspect_trees.py \
			-i ${SAMPLES_TREES_FILE}
			-o $@