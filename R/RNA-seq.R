#!/usr/bin/env R
setwd("/Users/sm5911/OneDrive - Imperial College London/Graduate_School/Computational_Biology_Week_2019/Code/")

# Packages & Dependencies
library(RColorBrewer)
library(ggplot2)
library(KEGG.db)
library(goseq)
# Workflow: Raw data -> quality control -> reads -> alignment -> SAM/BAM ->
#           feature counting -> counts -> visualisation -> PCA analysis ->
#           differential expression analysis -> CSV -> Functional analysis

# Load sequence data
targets <- read.table("../Data/LMSBioinformatics-LMS_RNAseq_short-a7be0ab/course/targets.txt",
                      sep="\t", header=TRUE)
head(targets)

AllCounts <- read.csv("../Data/LMSBioinformatics-LMS_RNAseq_short-a7be0ab/course/AllCounts.csv",
                      header=TRUE, row.names=1)
head(AllCounts)

# Prepare Deseq dataset
cData <- data.frame(name=targets$Sample,
                    Group=targets$Group,
                    Batch=targets$Batch)
rownames(cData) <- cData[,1]

# Construct deseqdataset object
dds <- DESeqDataSetFromMatrix(countData=AllCounts,
                              colData=cData,
                              design=~Group)

# Differential expression analysis - (1, 2, & 3) OVERALL
dds <- DESeq(dds) # See ?DESeq


# Differential expression analysis - (1) Estimation of size factors
sizeFactors(dds)

# Divide by size factor to yield normalized counts - Size factor normalization

# Estimate dispersions
head(dispersions(dds))
plotDispEsts(dds)

# Hypothesis test for differential expression significance testing
# Wald test tests whether each model coefficient differs significantly from zero,
# using previously calculated sizeFactors and dispersion estimates.
# P values are adjusted for multiple testing using the procedure of Benjamini and Hochberg
nbinomWaldTest(dds)

# Getting results
res <- results(dds)

# Rank by p-value
resOrdered <- res[order(res$padj), ]
head(resOrdered)

# Gene annotation -- Add gene symbols
mart <- useMart('ENSEMBL_MART_ENSEMBL', dataset='mmusculus_gene_ensembl',
                host="may2012.archive.ensembl.org") # Must provide host info if you want old genome verion (eg. mm9 not mm10)

# Retrieve sample annotations
bm <- getBM(attributes=c('ensembl_gene_id','mgi_symbol'),
            filters ='ensembl_gene_id',
            values=rownames(resOrdered), mart=mart)

head(bm)

# Merge the gene symbols with our DE dataset
resAnnotated <- merge(as.data.frame(resOrdered), bm, by.x=0, by.y=1)
colnames(resAnnotated)[1] <- "ensembl_gene_id" # Update EnsID column name

# Order results by adjusted p-value
resAnnotated <- resAnnotated[order(resAnnotated$pvalue, decreasing=FALSE), ]
head(resAnnotated) # show result with gene symbols

# Save to file (CSV format - recommended) 
write.csv(resAnnotated, file="../Results/DESeq_result.csv", row.names=FALSE)

# Exploring results
summary(res, alpha=0.05)

# Adjusted p-values < 0.05
sum(res$padj < 0.05, na.rm=TRUE) # Cook's Distance used to filter row samples

# MA Plots
plotMA(res, main="DESeq2: MA plot", ylim=c(-4,4))

# Identify groups -- Click on MAplot in plot window
# identity and rownames are shown (user input required)
idx <- identify(res$baseMean, res$log2FoldChange); rownames(res)[idx]

# Plot of normalized counts for a single gene on log scale
plotCounts(dds,gene=which.min(res$padj),intgroup="Group")

# Transforming count data -- n0 -- a positive constant
# NB: For visualisation ONLY -- raw counts are used for downstream analyses
# 1. The regularized logarithm (rlog)
rld <- rlog(dds)

# 2. Variance stabilizing transformation (vst)
vsd <- varianceStabilizingTransformation(dds)

# We can now use this tranformed data for visualisation
# Sort genes by expression levels and plot heatmap
selected <- order(rowMeans(counts(dds, normalized=TRUE)), decreasing=TRUE)[1:20]
pheatmap(assay(rld)[selected, ])
pheatmap(assay(vsd)[selected, ])

# Heatmap of sample-sample distances
sampleDists <- dist(t(assay(rld)))

sampleDistMatrix <- as.matrix(sampleDists)

# Save plot as .png
png(file="../Results/sample_dis_map.png")
rownames(sampleDistMatrix) <- rld$Group
colnames(sampleDistMatrix) <- NULL
colors <- colorRampPalette(rev(RColorBrewer::brewer.pal(9, "Blues")))(255)
pheatmap(sampleDistMatrix,
         clustering_distance_rows=sampleDists,
         clustering_distance_cols=sampleDists,
         col=colors)
dev.off()

# Principal Components Analysis (PCA)
plotPCA(rld, intgroup="Group")
ggsave(file="../Results/PCA_plot_version1.png", width=6, height=6, units="in", dpi=300) # Save plot

# Explicitly setting factor levels (default = alphabetical)
cData$Group <- relevel(cData$Group, ref="Viv")

# Use 'Contrasts' to generate results for all possible comparisons
# (eg. compare B vs A, of C vs A, and C vs B)
res_contrast <- results(dds, contrast=c("Group","Hfd","Viv"))
summary(res_contrast)

# Gene Ontology and Pathway Enrichment Analysis
# NB: GO analysis using 'goseq' package, which requires: 
# (1) measured genes - all genes in RNA-seq data
# (2) differentially expressed genes - only genes that are statistically significant

# Remove NA values
resdat <- res[complete.cases(res$padj), ]

# Boolean vector (0 = below pvalue, 1 = above pvalue)
diff_expr_genes <- as.integer(resdat$padj < 0.05)
names(diff_expr_genes) <- rownames(resdat)

# Remove duplicate gene names
diff_expr_genes <- diff_expr_genes[match(unique(names(diff_expr_genes)), names(diff_expr_genes))]
table(diff_expr_genes)

# Fit a Probability Weighting Function (PWF)
pwf=nullp(diff_expr_genes, genome="mm9", id='ensGene', plot.fit=FALSE)
head(pwf)

# Plot 
graphics.off(); plotPWF(pwf)

# Change the Keggpath id to name in the goseq output
xx <- as.list(KEGGPATHID2NAME)
temp <- cbind(names(xx),unlist(xx))

addKeggTogoseq <- function(JX,temp){
  for(l in 1:nrow(JX)){
    if(JX[l,1] %in% temp[,1]){
      JX[l,"term"] <- temp[temp[,1] %in% JX[l,1],2]
      JX[l,"ontology"] <- "KEGG"
    }
  }
  return(JX)
}

# Goseq

BiocManager::install("org.Mm.eg.db")
go <- goseq(pwf, genome="mm9", id='ensGene', test.cats=c("GO:BP","GO:MF","KEGG"))
head(go)

restemp <- addKeggTogoseq(go, temp)
head(restemp)

# Save to CSV file
write.csv(restemp, file="../Results/GO_Kegg_Wallenius.csv", row.names=F)















