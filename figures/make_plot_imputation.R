require(readxl)
require(tidyverse)
require(ggplot2)


csv_file <- "tsimpute/results/modern_ooa_unequal_900505_haploid_miss05_yri_demes_yri//results.csv"
pdf_file <- "figure_ooa_unequal.pdf"

dat <- read.csv(csv_file, sep = ';')
dat <- dat[dat$maf_category != '( 0.0000%,  0.0100%)', ]
dat$perc_genotypes_correct_asin <- asin(dat$perc_genotypes_correct)


g <- dat %>%
  ggplot(aes(y = perc_genotypes_correct_asin,
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
  ggtitle("Out-of-Africa (YRI: 2.5%; CHB: 2.5%; CEU: 95%)") +
  ylab("Concordance rate") +
  xlab("MAF category") +
  theme(panel.background = element_blank(),
        panel.grid  = element_line(size     = 0.5,
                                   linetype = "dashed",
                                   colour   = "lightgrey"),
        axis.line   = element_line(size = 1),
        axis.text.y = element_text(size = 16),
        axis.text.x = element_text(size = 16,
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
