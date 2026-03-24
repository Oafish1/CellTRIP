library(Matrix)

# Load ExpVal
setwd("~/repos/celltrip/data/ExpVal")
# data <- readRDS("monkey_acute.RDs")
# data <- readRDS("monkey_cultured.RDS")
# data <- readRDS("integratedRNA_with3BatchCorrections.RDS")
data <- readRDS("human_cultured.RDS")
counts <- data@assays$RNA@counts
meta <- data@meta.data

# Load NHP
setwd("~/repos/celltrip/data/NHP")
load("NHP_top5kPeaks.RData")
counts <- peaks
meta <- meta

# Convert factors
i <- sapply(meta, is.factor)
meta[i] <- lapply(meta[i], as.character)

# Write
# writeMM(counts, "monkey_acute_counts.mtx")
# write(counts@Dimnames[[1]], "monkey_acute_genes.txt")
# write.csv(meta, "monkey_acute_meta.csv")
# writeMM(counts, "monkey_cultured_counts.mtx")
# write(counts@Dimnames[[1]], "monkey_cultured_genes.txt")
# write.csv(meta, "monkey_cultured_meta.csv")
# writeMM(counts, "integratedRNA_with3BatchCorrections_counts.mtx")
# write(counts@Dimnames[[1]], "integratedRNA_with3BatchCorrections_genes.txt")
# write.csv(meta, "integratedRNA_with3BatchCorrections_meta.csv")
# writeMM(counts, "human_cultured_counts.mtx")
# write(counts@Dimnames[[1]], "human_cultured_genes.txt")
# write.csv(meta, "human_cultured_meta.csv")
writeMM(counts, "top5k_counts.mtx")
write(counts@Dimnames[[1]], "top5k_genes.txt")
write(counts@Dimnames[[2]], "top5k_bars.txt")
write.csv(meta, "top5k_meta.csv")
