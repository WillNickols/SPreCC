#! usr/bin/env Rscript

library(docopt)
library(dplyr)
library(data.table)
library(stringr)
library(ROCR)
rm(list = ls())

'Usage:
   threshold.R [-i final_pass.log --encodings encodings_100.tsv --thresholds thresholds.tsv --make_thresholds y --only_with_sim y -o out.tsv]

Options:
   -i final_pass.log
   --encodings encodings_100.tsv
   --thresholds thresholds.tsv
   --make_thresholds y/n
   --only_with_sim y/n
   -o out.tsv

' -> doc 

opts <- docopt(doc)

if (opts$make_thresholds!="y" & opts$make_thresholds!="n") {
  stop("--make_thresholds must be y or n")
}

in_file <- read.table(opts$i, header = T, sep = "\t", check.names = F)
in_file <- in_file[,-dim(in_file)[2]]

split1 <- strsplit(as.character(in_file[,"method"]), "\\)")
n_p <- mapply("[[", split1, sapply(split1, length)) %>% gsub(pattern = ".*, ", replacement = "") %>% as.numeric()
print(mean(n_p>0))

if (opts$only_with_sim == "y") {
  in_file <- in_file[n_p>0,]
}

encodings <- read.table(opts$encodings, header = T, sep = "\t", check.names = F)
encodings <- encodings[!is.na(encodings$encoding),]
encodings_vec <- encodings$encoding
names(encodings_vec) <- encodings$name

if (opts$make_thresholds == "n") {
  thresholds_df = read.table(opts$thresholds, header = T, sep = "\t", check.names = F)
}

precision = function(y, yhat) {
  return (mean(y[yhat==1], na.rm=T))
}
recall = function(y, yhat) {
  return (mean(yhat[y==1], na.rm=T))
}
f1 = function(y, yhat) {
  precision_val = precision(y, yhat)
  recall_val = recall(y, yhat)
  f1 = 2 * precision_val * recall_val / (precision_val + recall_val)
  f1 = ifelse(is.na(f1) & !is.na(precision_val) &!is.na(recall_val), 0, f1)
  return(f1)
}
acc = function(y, yhat) {
  return(mean(y==yhat, na.rm=T))
}
auc = function(y, prob) {
  pred = tryCatch(prediction(prob, y), error=function(err) NA)
  if (is.na(pred)) {
    return (NA)
  }
  auc_ROCR <- performance(pred, measure="auc")@y.values[[1]]
  return(auc_ROCR)
}
lm_coef = function(y, yhat) {
  return(tryCatch(summary(lm(y ~ yhat))$coef[2,1], error = function(e) NA))
}
lm_signif = function(y, yhat) {
  return(tryCatch(summary(lm(y ~ yhat))$coef[2,4], error = function(e) NA))
}
is_between = function(y, lb, ub) {
  return(y >= lb & y <= ub)
}

# Returns (binary accuracy, binary precision, binary recall, binary f1, auc, mean cor 1, mean cor 2, 
# mode cor 1, mode cor 2, median cor 1, median cor 2, mean CI width 1, mean CI width 2, 
# mean capture proportion, threshold)
get_accuracy <- function(encoding_type, condition_vec, threshold) {
  if (encoding_type == "cat") {
    split1 <- strsplit(as.character(condition_vec), "\\(")
    
    # Remove NA
    split1 <- split1[sapply(split1, length)==3]
    
    # Get correct value
    correct <- sapply(split1, "[[", 2) %>% strsplit(split=",") %>% sapply(FUN = "[[", 1) %>% as.numeric()
    
    # Get prediction
    pred <- sapply(split1, "[[", 3) %>% 
      strsplit(split="\\)") %>% 
      sapply(FUN = "[[", 1) %>% 
      sapply(FUN = function(x){strsplit(x, split = ", ") %>% unlist() %>% as.numeric() %>% which.max()}) %>%
      unname()
    
    # Fix indexing
    pred <- pred - 1
    
    return(c(rep(acc(correct, pred), 4), rep(NA, 23)))
  }
  if (encoding_type == "cont") {
    split1 <- gsub("\\(|\\)", "", as.character(condition_vec)) %>% strsplit(split = ",")
    split1 <- lapply(split1, as.numeric)
    y <- sapply(split1, FUN = "[[", 1)
    split1 <- split1[!is.na(y)]
    y <- y[!is.na(y)]
    mean <- sapply(split1, FUN = "[[", 2)
    mode <- sapply(split1, FUN = "[[", 3)
    median <- sapply(split1, FUN = "[[", 4)
    lb <- sapply(split1, FUN = "[[", 5)
    ub <- sapply(split1, FUN = "[[", 6)
    
    return(c(rep(NA, 5), lm_coef(mean, y), lm_signif(mean, y), mean(abs(mean - y), na.rm=T), NA, NA, NA,
             lm_coef(mode, y), lm_signif(mode, y), mean(abs(mode - y), na.rm=T), NA, NA, NA,
             lm_coef(median, y), lm_signif(median, y), mean(abs(median - y), na.rm=T), NA, NA, NA,
             mean(ub-lb, na.rm=T), NA, mean(is_between(y, lb, ub), na.rm=T), NA))
  }
  if (encoding_type == "contbin") {
    split1 <- gsub("\\(|\\)", "", as.character(condition_vec)) %>% strsplit(split = ",")
    split1 <- lapply(split1, as.numeric)
    y1 <- sapply(split1, FUN = "[[", 1)
    split1 <- split1[!is.na(y1)]
    y1 <- y1[!is.na(y1)]
    y2 <- sapply(split1, FUN = "[[", 2)
    mean <- sapply(split1, FUN = "[[", 3)
    mode <- sapply(split1, FUN = "[[", 4)
    median <- sapply(split1, FUN = "[[", 5)
    lb <- sapply(split1, FUN = "[[", 6)
    ub <- sapply(split1, FUN = "[[", 7)
    prob <- sapply(split1, FUN = "[[", 8)
    
    if (opts$make_thresholds=="y") {
      pred = tryCatch(prediction(prob, y2), error=function(err) NA)
      if (is.na(pred)) {
        threshold = 0.5
      } else {
        threshold = unlist(pred@cutoffs)[which.max(unlist(pred@tp)/(unlist(pred@tp) + 1/2 * (unlist(pred@fp) + unlist(pred@fn))))]
      }
    } else if (is.na(threshold)) {
      threshold = 0.5
    }
    
    yhat <- sapply(split1, FUN = "[[", 8) >= threshold
    
    return(c(acc(y2, yhat), precision(y2, yhat), recall(y2, yhat), f1(y2, yhat), auc(y2, prob),
             lm_coef(mean, y1), lm_signif(mean, y1), mean(abs(mean - y1), na.rm=T), NA, NA, NA,
             lm_coef(mode, y1), lm_signif(mode, y1), mean(abs(mode - y1), na.rm=T), NA, NA, NA,
             lm_coef(median, y1), lm_signif(median, y1), mean(abs(median - y1), na.rm=T), NA, NA, NA,
             mean(ub-lb, na.rm=T), NA,
             mean(is_between(y1, lb, ub), na.rm=T), threshold))
  }
  if (encoding_type == "bicontbin") {
    split1 <- gsub("\\(|\\)", "", as.character(condition_vec)) %>% strsplit(split = ",")
    split1 <- lapply(split1, as.numeric)
    y1 <- sapply(split1, FUN = "[[", 1)
    split1 <- split1[!is.na(y1)]
    y1 <- y1[!is.na(y1)]
    y2 <- sapply(split1, FUN = "[[", 2)
    y3 <- sapply(split1, FUN = "[[", 3)
    mean1 <- sapply(split1, FUN = "[[", 4)
    mean2 <- sapply(split1, FUN = "[[", 5)
    mode1 <- sapply(split1, FUN = "[[", 6)
    mode2 <- sapply(split1, FUN = "[[", 7)
    median1 <- sapply(split1, FUN = "[[", 8)
    median2 <- sapply(split1, FUN = "[[", 9)
    lb1 <- sapply(split1, FUN = "[[", 10)
    lb2 <- sapply(split1, FUN = "[[", 11)
    ub1 <- sapply(split1, FUN = "[[", 12)
    ub2 <- sapply(split1, FUN = "[[", 13)
    prob <- sapply(split1, FUN = "[[", 14)
    
    if (opts$make_thresholds=="y") {
      pred = tryCatch(prediction(prob, y3), error=function(err) NA)
      if (is.na(pred)) {
        threshold = 0.5
      } else {
        threshold = unlist(pred@cutoffs)[which.max(unlist(pred@tp)/(unlist(pred@tp) + 1/2 * (unlist(pred@fp) + unlist(pred@fn))))]
      }
    } else if (is.na(threshold)) {
      threshold = 0.5
    }
    
    yhat <- sapply(split1, FUN = "[[", 14) >= threshold

    return(c(acc(y3, yhat), precision(y3, yhat), recall(y3, yhat), f1(y3, yhat), auc(y3, prob),
             lm_coef(mean1, y1), lm_signif(mean1, y1), mean(abs(mean1 - y1), na.rm=T), lm_coef(mean2, y2), lm_signif(mean2, y2), mean(abs(mean2 - y2), na.rm=T),
             lm_coef(mode1, y1), lm_signif(mode1, y1), mean(abs(mode1 - y1), na.rm=T), lm_coef(mode2, y2), lm_signif(mode2, y2), mean(abs(mode2 - y2), na.rm=T),
             lm_coef(median1, y1), lm_signif(median1, y1), mean(abs(median1 - y1), na.rm=T), lm_coef(median2, y2), lm_signif(median2, y2), mean(abs(median2 - y2), na.rm=T),
             mean(ub1-lb1, na.rm=T), mean(ub2-lb2, na.rm=T), 
             mean(is_between(y1, lb1, ub1) & is_between(y2, lb2, ub2), na.rm=T), threshold))
  }
}

# Using >50% as the threshold - can come back to this and use a particular FDR etc. if necessary
out_mat <- matrix(nrow = length(colnames(in_file)), ncol = 27)
rownames(out_mat) <- colnames(in_file)
colnames(out_mat) <- c("binary_accuracy", "precision", "recall", "f1", "auc", 
                       "mean coef 1", "mean signif 1", "mean abs err 1",
                       "mean coef 2", "mean signif 2", "mean abs err 2",
                       "mode coef 1", "mode signif 1", "mode abs err 1",
                       "mode coef 2", "mode signif 2", "mode abs err 2",
                       "median coef 1", "median signif 1", "median abs err 1",
                       "median coef 2", "median signif 2", "median abs err 2",
                       "mean CI width 1", "mean CI width 2", 
                       "mean capture proportion", "threshold")
for (colname in colnames(in_file)) {
  
  print(paste0("Processing ", colname, "..."))
  if (opts$make_thresholds == "n") {
    threshold = thresholds_df$thresholds[thresholds_df$condition==colname]
  } else {
    threshold = NA
  }
  out_mat[colname,] <- get_accuracy(unname(encodings_vec[colname]), in_file[,colname], threshold)
}

out_df <- as.data.frame(out_mat)
out_df$condition <- rownames(out_mat)

if (opts$make_thresholds == "y") { 
  write.table(data.frame(thresholds=out_df$threshold, condition=out_df$condition), opts$thresholds, quote=FALSE, sep="\t", row.names=FALSE)
}

out_df$threshold <- NULL
write.table(out_df, opts$o, quote=FALSE, sep="\t", row.names=FALSE)

#