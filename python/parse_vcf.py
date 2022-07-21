import cyvcf2


def parse_vcf_file(vcf_file):
    """
    If gt_types = False, then 0=HOM_REF, 1=HET, 2=UNKNOWN, the coordinates are 0-based.
    It returns a list of dictionaries, each containing a VCF record.
    """
    parsed_vcf = []
    for variant in cyvcf2.VCF(vcf_file,
                              gts012 = False,
                              strict_gt = True):
        record = {
            'ref': variant.REF,
            'alt': variant.ALT,
            'ctg': variant.CHROM, # Contig id/name
            'pos': int(variant.start),
            'aa' : variant.INFO.get('AA'), # Ancestral allele
            'gt' : variant.genotypes
        }
        parsed_vcf.append(record)
    return(parsed_vcf)


def compare_vcf(vcf_1, vcf_2):
    assert len(vcf_1) == len(vcf_2)
    for i in range(len(vcf_1)):
        is_valid_ref = vcf_1[i].get('ref') == vcf_2[i].get('ref')
        is_valid_alt = vcf_1[i].get('alt') == vcf_2[i].get('alt')
        is_valid_ctg = vcf_1[i].get('ctg') == vcf_2[i].get('ctg')
        is_valid_pos = vcf_1[i].get('pos') == vcf_2[i].get('pos')
        is_valid_aa  = vcf_1[i].get('aa' ) == vcf_2[i].get('aa' )
        is_all_valid = np.all([is_valid_ref,
                               is_valid_alt,
                               is_valid_ctg,
                               is_valid_pos,
                               is_valid_aa])
        if not is_all_valid:
            return(False)
    return(True)


def get_common_positions_in_vcf(vcf_1, vcf_2):
    pos_1 = []
    pos_2 = []
    for i, record in enumerate(vcf_1):
        pos_1.append(record.get('pos'))
    for i, record in enumerate(vcf_2):
        pos_2.append(record.get('pos'))
    # All positions should be unique.
    assert len(pos_1) == len(set(pos_1)),\
        "The positions in vcf_1 are not all unique."
    assert len(pos_2) == len(set(pos_2)),\
        "The positions in vcf_2 are not all unique."
    common_pos = list(set.intersection(set(pos_1), set(pos_2)))
    return(common_pos)


def compare_variants(true_vcf_file,
                     miss_vcf_file,
                     imputed_vcf_file):
    true_vcf    = parse_vcf_file(true_vcf_file)
    miss_vcf    = parse_vcf_file(miss_vcf_file)
    imputed_vcf = parse_vcf_file(imputed_vcf_file)    
    assert compare_vcf(true_vcf, miss_vcf),\
        "true_vcf and miss_vcf are not comparable."
    # Imputed VCF file must have at most the number of positions as the true/miss VCF files.
    common_pos = get_common_positions_in_vcf(miss_vcf, imputed_vcf)
    # Number of genotypes imputed, correctly or not.
    nbr_gt_total = 0
    # Number of instances of genotypes correctly imputed.
    nbr_gt_correct = 0
    for i in range(len(imputed_vcf)):
        if true_vcf[i]['pos'] not in common_pos\
            or miss_vcf[i]['pos'] not in common_pos\
            or imputed_vcf[i]['pos'] not in common_pos:
            continue
        imputed_bool = [x == [-1, -1, True]
                        for x
                        in miss_vcf[i]['gt']]
        true_gt_oi = [x
                      for x, y
                      in zip(true_vcf[i]['gt'], imputed_bool) if y]
        imputed_gt_oi = [x
                         for x, y
                         in zip(imputed_vcf[i]['gt'], imputed_bool) if y]
        nbr_gt_total   += len(true_gt_oi)
        nbr_gt_correct += np.count_nonzero([x == y
                                            for x, y
                                            in zip(true_gt_oi, imputed_gt_oi)])
    concordance_rate = float(nbr_gt_correct) / float(nbr_gt_total)
    return((nbr_gt_total, nbr_gt_correct, concordance_rate))
