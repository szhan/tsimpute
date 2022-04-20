require(readxl)
require(tidyverse)
require(ggplot2)


categorize_maf <- function(csv_file) {
  dat <- read.csv(csv_file, sep = ',')
  dat <- dat[dat$maf != 0.50, ]
  
  maf_category <- c()
  
  maf_levels <- c('( 0.0000%,  0.0050%)',
                  '[ 0.0050%,  0.0100%)',
                  '[ 0.0100%,  0.1000%)',
                  '[ 0.1000%,  0.2000%)',
                  '[ 0.2000%,  0.3000%)',
                  '[ 0.3000%,  0.4000%)',
                  '[ 0.4000%,  0.5000%)',
                  '[ 0.5000%,  1.0000%)',
                  '[ 1.0000%,  2.0000%)',
                  '[ 2.0000%,  5.0000%)',
                  '[ 5.0000%, 10.0000%)',
                  '[10.0000%, 20.0000%)',
                  '[20.0000%, 30.0000%)',
                  '[30.0000%, 40.0000%)',
                  '[40.0000%, 50.0000%)')
  
  for(i in 1:nrow(dat)){
    maf <- dat$maf[i]
    
    if(maf < 0.00005){
      maf_category <- c(maf_category, '( 0.0000%,  0.0050%)')
    } else if(maf >= 0.00005 && maf < 0.0001){
      maf_category <- c(maf_category, '[ 0.0050%,  0.0100%)')
    } else if(maf >= 0.0001 && maf < 0.001){
      maf_category <- c(maf_category, '[ 0.0100%,  0.1000%)')
    } else if(maf >= 0.001 && maf < 0.002){
      maf_category <- c(maf_category, '[ 0.1000%,  0.2000%)')
    } else if(maf >= 0.002 && maf < 0.003){
      maf_category <- c(maf_category, '[ 0.2000%,  0.3000%)')
    } else if(maf >= 0.003 && maf < 0.004){
      maf_category <- c(maf_category, '[ 0.3000%,  0.4000%)')
    } else if(maf >= 0.004 && maf < 0.005){
      maf_category <- c(maf_category, '[ 0.4000%,  0.5000%)')
    } else if(maf >= 0.005 && maf < 0.01){
      maf_category <- c(maf_category, '[ 0.5000%,  1.0000%)')
    } else if(maf >= 0.01 && maf < 0.02){
      maf_category <- c(maf_category, '[ 1.0000%,  2.0000%)')
    } else if(maf >= 0.02 && maf < 0.05){
      maf_category <- c(maf_category, '[ 2.0000%,  5.0000%)')
    } else if(maf >= 0.05 && maf < 0.10){
      maf_category <- c(maf_category, '[ 5.0000%, 10.0000%)')
    } else if(maf >= 0.10 && maf < 0.20){
      maf_category <- c(maf_category, '[10.0000%, 20.0000%)')
    } else if(maf >= 0.20 && maf < 0.30){
      maf_category <- c(maf_category, '[20.0000%, 30.0000%)')
    } else if(maf >= 0.30 && maf < 0.40){
      maf_category <- c(maf_category, '[30.0000%, 40.0000%)')
    } else if(maf >= 0.40 && maf < 0.50){
      maf_category <- c(maf_category, '[40.0000%, 50.0000%)')
    } else {
      print(maf)
      print("MAF value is invalid!")
    }
  }
  
  dat$maf_category <- factor(maf_category, levels = maf_levels, ordered = TRUE)
  
  return(dat)
}


setwd("tsimpute/")


csv_file_tsonly <- "data/modern_ooa_unequal_900505_haploid_miss10/results/tsonly.csv"
csv_file_tsinfer <- "data/modern_ooa_unequal_900505_haploid_miss10/results/tsinfer.csv"
csv_file_beagle <- "data/modern_ooa_unequal_900505_haploid_miss10/results/beagle.csv"

pdf_file <- "figure_ooa_unequal_rare.pdf"

dat_tsonly <- categorize_maf(csv_file_tsonly)
dat_tsinfer <- categorize_maf(csv_file_tsinfer)
dat_beagle <- categorize_maf(csv_file_beagle)

dat_tsonly <- dat_tsonly[dat_tsonly$position %in% dat_tsinfer$position, ]
dat_tsinfer <- dat_tsinfer[dat_tsinfer$position %in% dat_tsonly$position, ]
dat_beagle <- dat_beagle[dat_beagle$position %in% dat_tsonly$position, ]

dim(dat_tsonly)
dim(dat_tsinfer)
dim(dat_beagle)

dat <- rbind(dat_beagle, dat_tsinfer, dat_tsonly)
dat <- dat[dat$maf <= 0.002, ]

summary(dat_tsonly[dat_tsonly$maf_category == '[ 0.0100%,  0.1000%)', ]$iqs)
summary(dat_tsinfer[dat_tsinfer$maf_category == '[ 0.0100%,  0.1000%)', ]$iqs)
summary(dat_beagle[dat_beagle$maf_category == '[ 0.0100%,  0.1000%)', ]$iqs)
length(dat_beagle[dat_beagle$maf_category == '[ 0.0100%,  0.1000%)', ]$iqs)

g <- dat %>%
  ggplot(aes(y = iqs,
             x = maf_category,
             fill = method)) +
  #stat_summary(fun.data = "mean_sdl",
  #             fun.args = list(mult = 1), 
  #             geom     = "crossbar",
  #             width    = 0.5,
  #             col      = "orange") +
  geom_boxplot() +
  #geom_jitter(shape = 4,
  #            size  = 6,
  #            position = position_dodge()) +
  #ggtitle("Single, panmictic population") +
  #ggtitle("Out-of-Africa (YRI: 33%; CHB: 33%; CEU: 33%)") +
  #ggtitle("Out-of-Africa (CEU, 90%; YRI, 5%; CHB, 5%)") +
  ggtitle("Out-of-Africa") +
  ylab("IQS") +
  xlab("MAF category") +
  theme(panel.background = element_blank(),
        panel.grid  = element_line(size     = 0.5,
                                   linetype = "dashed",
                                   colour   = "lightgrey"),
        axis.line   = element_line(size = 1),
        axis.text.y = element_text(size = 16),
        axis.text.x = element_text(size = 14,
                                   angle = 30,
                                   vjust = 0.5),
        axis.title  = element_text(size = 18),
        title       = element_text(size = 22)) +
  scale_fill_manual(values = c("yellowgreen",
                               "#ff7518",
                               "#002147")) +
  scale_alpha_manual(values = c(0.90,
                                0.90,
                                0.65))

pdf(pdf_file, useDingbats = FALSE)
g
dev.off()
