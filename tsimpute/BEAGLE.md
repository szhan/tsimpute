Notes about this re-implementation of the BEAGLE 4.1 algorithm.

BEAGLE imputes alleles by linear interpolation of the hidden state probabilities
at ungenotyped site positions.

This was implemented while closely consulting the BEAGLE 4.1 paper and source code:
* Browning & Browning (2016). Am J Hum Genet 98:116-126. doi:10.1016/j.ajhg.2015.11.020
* Source code: https://faculty.washington.edu/browning/beagle/b4_1.html

These notations are used throughout:
h = number of reference haplotypes.
m = number of genotyped positions.
x = number of ungenotyped positions.

This implementation takes the following inputs:
* Reference haplotypes in a matrix of size (m + x, h).
* Query haplotype in an array of size (m + x).
* Physical positions of all the markers in an array of size (m + x).
* Genetic map.

In the query haplotype:
* Genotyped positions take values of 0, 1, 2, or 3 (i.e. ACGT encoding).
* Ungenotyped positions take -1.

The following objects are computed:
* Forward and backward probability matrices of size (m, h).
* Hidden state probability matrix of size (m, h).
* Interpolated state probability matrix of size (x, h).
* Imputed allele probability matrix of size (x, 4),
* Imputed alleles as the maximum a posteriori alleles.

The following evaluation metrics are produced in VCF format:
* Estimated allelic R-squared (AR2).
* Dosage R-squared (DR2).
* Estimated allele frequency (AF).
* Genotype probabilities of 00, 01/10, and 11 (GP).
* Estimated dosage (DS).

To improve computational efficiency, BEAGLE uses aggregated markers, which are clusters
of markers within a 0.005 cM interval (default). Because the genotypes are phased,
the alleles in the aggregated markers form distinct "allele sequences". Below, we do not
use aggregated markers or allele sequences, which would complicate the implementation.

Rather than exactly replicate the original BEAGLE algorithm, this implementation uses
Equation 1 of BB2016.