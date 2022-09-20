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
# Standard pipeline from .samples to .trees
####################################################

finns_chr20.trees: finns_chr20.samples
		# TODO: Infer tree sequence
		echo "Preparing samples file to analyse Finns genomic data"

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

#############################################
# Finns genomics data.
#############################################

finns_%.samples: ../data/fimm/v4.2.chr20_phased_SNPID.vcf.gz %_ancestral_states.fa.fai
		#tabix -f -p vcf $<
		python python/convert.py generic -p \
				../data/fimm/v4.2.chr20_phased_SNPID.vcf.gz \
				$*_ancestral_states.fa \
				-m None \
				--ancestral-states-url=${ANCESTRAL_STATES_URL} \
				--reference-name=${REFERENCE_NAME} \
				--num-threads=${NUM_THREADS} \
				$@ > $@.report
		echo "Preparing samples file"

clean:
		rm -f *.samples *.trees
