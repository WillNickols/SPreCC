library(dplyr)
library(stringr)
library(data.table)
library(ggplot2)

# Read metadata
metadata <- read.csv("metadata/metadata_parsed_100.tsv", sep = "\t", check.names = F, header = T)

# Read encodings
encodings <- read.csv("metadata/encodings_100.tsv", sep = "\t", header = T)

figure <- ggplot(data = metadata, aes(x=(eval(parse(text = "date")) %>% gsub(pattern = "-.*", replacement = "") %>% as.numeric()))) +
  geom_histogram(binwidth = 1) + 
  theme_bw() + 
  xlab("Date")

dir.create(file.path("Figures/metadata"), showWarnings = FALSE, recursive = T)
ggsave(plot = figure, filename = "Figures/metadata/date.png", width=18, height=12, units = 'cm', dpi=300)

for (column in encodings$name[encodings$encoding=="cont" & !is.na(encodings$encoding)]) {
  options(scipen=3)
  figure <- ggplot(data = metadata, aes(x=(eval(parse(text = column))))) +
    geom_histogram(bins = 50) + 
    theme_bw() + 
    xlab(paste0(column, 
                ", Proportion NA: ", 
                format(round(mean(is.na(eval(parse(text = paste0("metadata$",column)))), na.rm = T), 3), nsmall = 3)
))
  
  dir.create(file.path("Figures/metadata"), showWarnings = FALSE, recursive = T)
  ggsave(plot = figure, filename = paste0("Figures/metadata/", column, ".png"), width=18, height=12, units = 'cm', dpi=300)
}

for (column in encodings$name[encodings$encoding=="contbin" & !is.na(encodings$encoding)]) {
  figure <- ggplot(data = metadata[(eval(parse(text = paste0("metadata$`",column, "`"))))!=0,], aes(x=(eval(parse(text = paste0("metadata$`",column, "`"))))[(eval(parse(text = paste0("metadata$`",column, "`"))))!=0])) +
    geom_histogram(bins = 50) + 
    theme_bw() + 
    xlab(paste0(column, 
                ", Proportion Zero: ", 
                format(round(mean(eval(parse(text = paste0("metadata$`",column, "`")))==0, na.rm = T), 3), nsmall = 3),
                ", Proportion NA: ", 
                format(round(mean(is.na(eval(parse(text = paste0("metadata$`",column, "`")))), na.rm = T), 3), nsmall = 3)
    ))
  
  dir.create(file.path("Figures/metadata"), showWarnings = FALSE, recursive = T)
  ggsave(plot = figure, filename = paste0("Figures/metadata/", gsub(" \\(.*", "", column), ".png"), width=18, height=12, units = 'cm', dpi=300)
}

for (column in encodings$name[encodings$encoding=="bicontbin" & !is.na(encodings$encoding)]) {
  xy <- str_split(metadata[,column][metadata[,column]!="0" & !is.na(metadata[,column])], " , ", simplify = TRUE)
  xy <- data.frame(matrix(as.numeric(xy), ncol = ncol(xy)))
  colnames(xy) <- c("Concentration", "Value")
  
  figure <- ggplot(data = xy, aes(x=Concentration)) +
    geom_histogram(bins = 50) + 
    theme_bw() + 
    xlab(paste0(column, 
                " concentration, Proportion Zero: ", 
                format(round(mean(eval(parse(text = paste0("metadata$`",column, "`")))==0, na.rm = T), 3), nsmall = 3),
                ", Proportion NA: ", 
                format(round(mean(is.na(eval(parse(text = paste0("metadata$`",column, "`")))), na.rm = T), 3), nsmall = 3)
    ))
  
  dir.create(file.path("Figures/metadata"), showWarnings = FALSE, recursive = T)
  ggsave(plot = figure, filename = paste0("Figures/metadata/", gsub(" \\(.*", "", column), "_concentration.png"), width=18, height=12, units = 'cm', dpi=300)
  
  figure <- ggplot(data = xy, aes(x=Value)) +
    geom_histogram(bins = 50) + 
    theme_bw() + 
    xlab(paste0(column, 
                " value, Proportion Zero: ", 
                format(round(mean(eval(parse(text = paste0("metadata$`",column, "`")))==0, na.rm = T), 3), nsmall = 3),
                ", Proportion NA: ", 
                format(round(mean(is.na(eval(parse(text = paste0("metadata$`",column, "`")))), na.rm = T), 3), nsmall = 3)
    ))
  
  dir.create(file.path("Figures/metadata"), showWarnings = FALSE, recursive = T)
  ggsave(plot = figure, filename = paste0("Figures/metadata/", gsub(" \\(.*", "", column), "_value.png"), width=18, height=12, units = 'cm', dpi=300)
}





