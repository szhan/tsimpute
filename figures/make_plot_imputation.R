require(readxl)
require(tidyverse)
require(ggplot2)


csv_file <- "../Projects/tsimpute/results/modern_ooa_equal/results.csv"
pdf_file <- "figure_ooa_equal.pdf"

dat <- read.csv(csv_file, sep = ';')


g <- dat %>%
  ggplot(aes(y = concordance_rate,
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
  ggtitle("Out-of-Africa (YRI: 1/3; CHB: 1/3; CEU: 1/3)") +
  #ggtitle("Out-of-Africa (YRI: 0.05; CHB: 0.05; CEU: 0.95)") +
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

#pdf(pdf_file, useDingbats = FALSE)
g
#dev.off()
