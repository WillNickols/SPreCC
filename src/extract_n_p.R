#! usr/bin/env Rscript

library(docopt)
library(dplyr)
library(data.table)
library(stringr)
rm(list = ls())

'Usage:
   threshold.R [-i final_pass.log -o out.tsv]

Options:
   -i final_pass.log
   -o out.tsv

' -> doc 

opts <- docopt(doc)

in_file <- read.table(opts$i, header = T, sep = "\t", check.names = F)

split1 <- strsplit(as.character(in_file[,"method"]), "\\)")
n_p <- mapply("[[", split1, sapply(split1, length)) %>% gsub(pattern = ".*, ", replacement = "") %>% as.numeric()

df = data.frame("ID" = in_file[,dim(in_file)[2]], "n_p" = n_p)

write.table(df, opts$o, quote=FALSE, sep="\t", row.names=FALSE)

#