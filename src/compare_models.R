library(docopt)
library(dplyr)
library(ggplot2)
library(data.table)
library(stringr)
library(ROCR)
library(gridExtra)
library(reshape2)
rm(list = ls())

# Show models created from weights
models_from_weights <- function() {
  sigmoid = function(x, w_0, w_1) {1/(1+exp(-(w_1 * x + w_0))) * 1/((log(exp(-w_1) + exp(w_0)) - log(1 + exp(w_0)))/w_1 + 1)}
  
  file_list <- list.files("train_outputs/model_weights/")
  model_list = list(length(file_list))
  for (i in 1:length(file_list)) {
    new_weights <- read.table(paste0("train_outputs/model_weights/", file_list[i]), 
                              header = T, sep = "\t", check.names = F)
    new_weights$condition[!is.na(new_weights$w_20_cont)] <- paste0(new_weights$condition[!is.na(new_weights$w_20_cont)],
                                                                   " concentration")
    new_weights[(nrow(new_weights) + 1):(nrow(new_weights) + sum(!is.na(new_weights$w_20_cont))),] <-
      data.frame("condition"=paste0(new_weights$condition[!is.na(new_weights$w_20_cont)],
                               " length"), "w_0_bin" = NA, "w_1_bin" = NA, 
            "w_10_cont" = new_weights$w_20_cont[!is.na(new_weights$w_20_cont)],
            "w_11_cont" = new_weights$w_21_cont[!is.na(new_weights$w_21_cont)],
            "w_20_cont" = NA,
            "w_21_cont" = NA)
    new_weights <- new_weights[,-c(6,7)]
    new_weights$model <- gsub("_.*", "", file_list[i])
    model_list[i] <- list(new_weights)
  }
  
  p1 <-
    ggplot() +
    xlim(0, 1) +
    ylim(0,8)
  
  for(i in 1:nrow(model_list[[1]])) {
    p1 <- p1 + geom_function(fun = sigmoid, args = list(w_0 = model_list[[1]]$w_0_bin[i], 
                                                            w_1 = model_list[[1]]$w_1_bin[i]),
                             alpha=ifelse(model_list[[1]]$condition[i] %in% 
                                            c("method", "sodium tartrate (m)", "sodium chloride(m)"),
                                          1, 0.2), 
                             color=ifelse(model_list[[1]]$condition[i] %in% 
                                            c("method", "sodium tartrate (m)", "sodium chloride(m)"),
                                          "red", "black"))
  }
  p1 <- p1 + annotate("text", x=c(0.91, 0.85, 0.85), y=c(7, 2, 0.85), label= c("Method", "Sodium tartrate (m)", "Sodium chloride(m)"),
                      color="red", size=8) + 
    theme_bw() + ylab("o(x)") + 
    labs(title="Model 1 binary weights (B=0)") + 
    theme(text = element_text(size=25))

  p2 <-
    ggplot() +
    xlim(0, 1) +
    ylim(0,6)
  
  for(i in 1:nrow(model_list[[1]])) {
    p2 <- p2 + geom_function(fun = sigmoid, args = list(w_0 = model_list[[1]]$w_10_cont[i], 
                                                        w_1 = model_list[[1]]$w_11_cont[i]),
                             alpha=ifelse(model_list[[1]]$condition[i] %in% 
                                            c("sodium acetate (m)", "temp", "bis tris methane (m)", "p_h"),
                                          1, 0.2), 
                             color=ifelse(model_list[[1]]$condition[i] %in% 
                                            c("sodium acetate (m)", "temp", "bis tris methane (m)", "p_h"),
                                                     "red", "black"))

  }
  p2 <- p2 + annotate("text", x=c(0.15,0.1,0.2,0.02), y=c(5.8,4.8, 2.2, 3.9), label= c("Sodium acetate (m)", "Temperature", "Bis tris methane (m)", "pH"),
                      color="red", size=8) + 
    theme_bw() + ylab("o(x)") + theme(legend.position = "none",
                                      text = element_text(size=25)) + 
    labs(title="Model 1 continuous weights (B=0.001)")

  p3 <-
    ggplot() +
    xlim(0, 1) +
    ylim(0,6)
  
  for(i in 1:nrow(model_list[[2]])) {
    p3 <- p3 + geom_function(fun = sigmoid, args = list(w_0 = model_list[[2]]$w_10_cont[i], 
                                                        w_1 = model_list[[2]]$w_11_cont[i]),
                             alpha=ifelse(model_list[[2]]$condition[i] %in% 
                                            c("sodium acetate (m)", "temp", "bis tris methane (m)", "p_h"),
                                          1, 0.2), 
                             color=ifelse(model_list[[2]]$condition[i] %in% 
                                            c("sodium acetate (m)", "temp", "bis tris methane (m)", "p_h"),
                                          "red", "black"))
    
  }
  p3 <- p3 + annotate("text", x=c(0.85, 0.1, 0.2, 0.04), y=c(1.2, 3.4, 2.15, 1.6), label= c("Sodium acetate (m)", "Temperature", "Bis tris methane (m)", "pH"),
                      color="red", size=8) + 
    theme_bw() + ylab("o(x)") + theme(legend.position = "none",
                                      text = element_text(size=25)) + 
    labs(title="Model 2 continuous weights (B=0.01)")
  
  p4 <-
    ggplot() +
    xlim(0, 1) +
    ylim(0,6)
  
  for(i in 1:nrow(model_list[[3]])) {
    p4 <- p4 + geom_function(fun = sigmoid, args = list(w_0 = model_list[[3]]$w_10_cont[i], 
                                                        w_1 = model_list[[3]]$w_11_cont[i]),
                             alpha=ifelse(model_list[[3]]$condition[i] %in% 
                                            c("sodium acetate (m)", "temp", "bis tris methane (m)", "p_h"),
                                          1, 0.2), 
                             color=ifelse(model_list[[3]]$condition[i] %in% 
                                            c("sodium acetate (m)", "temp", "bis tris methane (m)", "p_h"),
                                          "red", "black"))
    
  }
  p4 <- p4 + annotate("text", x=c(0.85, 0.1, 0.2, 0.3), y=c(1.2, 0.9, 1.7, 0.9), label= c("Sodium acetate (m)", "Temperature", "Bis tris methane (m)", "pH"),
                      color="red", size=8) + 
    theme_bw() + ylab("o(x)") + theme(legend.position = "none",
                                      text = element_text(size=25)) + 
    labs(title="Model 3 continuous weights (B=0.1)")
  
  dir.create(file.path("Figures/weights/"), showWarnings = FALSE)
  png(file=paste0("Figures/weights/weights_by_model.png"), width = 2400, height = 800)
  grid.arrange(p1, p2, p3, p4, nrow=1, ncol=4, widths=c(2,2,2,2))
  dev.off()
}

n_p_histogram <- function(){
  test_n_p <- read.table("train_outputs/non_model/test_n_p.tsv", 
             header = T, sep = "\t", check.names = F)
  test_n_p$partition <- paste0("Test\n(", round(100 * mean(test_n_p$n_p==0), 2), "% 0)")
  train_n_p <- read.table("train_outputs/non_model/train_n_p.tsv", 
                         header = T, sep = "\t", check.names = F)
  train_n_p$partition <- paste0("Train\n(", round(100*mean(train_n_p$n_p==0),2), "% 0)")
  merged <- rbind(test_n_p, train_n_p)
  
  dir.create(file.path("Figures/n_p/"), showWarnings = FALSE)
  png(file=paste0("Figures/n_p/n_p_by_partition.png"), width = 2400, height = 700)
  ggplot(merged, aes(x=partition, y=n_p)) +
    geom_violin(fill="grey") +
    theme_bw() + 
    ylab(expression(n[p])) + 
    xlab("") +
    coord_flip() + 
    theme(text = element_text(size=70))
  dev.off()
}

density_interval_vs_nominal <- function(){
  file_list <- list.files("train_outputs/analysis/")
  density_interval_df <- data.frame(matrix(nrow = 0, ncol = 5))
  for (i in 1:length(file_list)) {
    train_out <- read.table(paste0("train_outputs/analysis/", file_list[i]), 
                              header = T, sep = "\t", check.names = F)
    df_addition <- data.frame("capture_prop"=train_out$`mean capture proportion`,
          "nominal" = gsub(".*ci_", "", file_list[i]) %>% gsub(pattern="[^0-9].*", replacement="") %>% as.numeric(),
          "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
          "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
          "model" = gsub("_.*", "", file_list[i]))
    density_interval_df <- rbind(density_interval_df, df_addition)
    
  }
  colnames(density_interval_df) <- c("capture_prop", "nominal", "partition", "n_p", "model")
  density_interval_df$nominal <- factor(density_interval_df$nominal)
  density_interval_df$partition <- factor(density_interval_df$partition, levels=c("Train", "Test"))
  density_interval_df$n_p <- factor(density_interval_df$n_p)
  density_interval_df$model <- case_when(density_interval_df$model == "model1" ~ "Model 1",
                                         density_interval_df$model == "model2" ~ "Model 2",
                                         density_interval_df$model == "model3" ~ "Model 3",
                                         density_interval_df$model == "null" ~ "Null Model",
                                         density_interval_df$model == "untrained" ~ "Untrained Model")
  
  dir.create(file.path("Figures/density_interval/"), showWarnings = FALSE)
  png(file=paste0("Figures/density_interval/density_interval_vs_nominal.png"), width = 3200, height = 1600)
  ggplot(density_interval_df, aes(x=nominal, y=capture_prop, fill=n_p)) + 
    geom_boxplot(lwd=2, outlier.size=5) +
    scale_y_continuous(breaks = seq(0,1,0.1)) +
    theme_bw() + 
    xlab("Nominal density interval (%)") +
    ylab("Capture proportion") +
    facet_grid(partition ~ model) + 
    geom_hline(yintercept=0.75, linetype="dashed", color = "red") +
    geom_hline(yintercept=0.9, linetype="dashed", color = "red") +
    scale_fill_discrete(breaks=levels(density_interval_df$n_p),
                            labels=c(expression("Without " ~ n[p] ~ "= 0"), expression("With " ~ n[p] ~ "= 0"))) + 
    theme(legend.title=element_blank(),
          text = element_text(size=50))
  dev.off()
}

volcano_plot_mean <- function(){
  file_list <- list.files("train_outputs/analysis/")
  volcano_df <- data.frame(matrix(nrow = 0, ncol = 5))
  
  for (i in 1:length(file_list)) {
    train_out <- read.table(paste0("train_outputs/analysis/", file_list[i]), 
                            header = T, sep = "\t", check.names = F)
    df_addition <- data.frame("coef"=train_out$`mean coef 1`,
                              "signif" = train_out$`mean signif 1`,
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    volcano_df <- rbind(volcano_df, df_addition)
    df_addition <- data.frame("coef"=train_out$`mean coef 2`,
                              "signif" = train_out$`mean signif 2`,
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    volcano_df <- rbind(volcano_df, df_addition)
  }
  colnames(volcano_df) <- c("coef", "signif", "partition", "n_p", "model")
  volcano_df$partition <- factor(volcano_df$partition, levels=c("Train", "Test"))
  volcano_df$n_p <- factor(volcano_df$n_p)
  volcano_df$model <- case_when(volcano_df$model == "model1" ~ "Model 1",
                                volcano_df$model == "model2" ~ "Model 2",
                                volcano_df$model == "model3" ~ "Model 3",
                                volcano_df$model == "null" ~ "Null Model",
                                volcano_df$model == "untrained" ~ "Untrained Model")
  volcano_df$signif[volcano_df$signif==0] <- .Machine$double.xmin
  volcano_df$signif <- -log(volcano_df$signif, base = 10)
  
  dir.create(file.path("Figures/volcano/"), showWarnings = FALSE)
  png(file=paste0("Figures/volcano/mean.png"), width = 3200, height = 1600)
  ggplot(volcano_df, aes(x=coef, y=signif, color=n_p)) + 
    geom_point(size=5) +
    theme_bw() + 
    xlab("Coefficient") +
    geom_hline(yintercept=-log(0.05, 10), linetype="dashed", color = "red") +
    geom_vline(xintercept=0, linetype="dashed", color = "red") +
    ylab(expression("-" ~ log[10] ~ "(p-value)")) +
    facet_grid(partition ~ model) + 
    scale_color_discrete(breaks=levels(volcano_df$n_p),
                        labels=c(expression("Without " ~ n[p] ~ "= 0"), expression("With " ~ n[p] ~ "= 0"))) + 
    theme(legend.title=element_blank(),
          text = element_text(size=50))
  dev.off()
}

volcano_plot_mode <- function(){
  file_list <- list.files("train_outputs/analysis/")
  volcano_df <- data.frame(matrix(nrow = 0, ncol = 5))
  
  for (i in 1:length(file_list)) {
    train_out <- read.table(paste0("train_outputs/analysis/", file_list[i]), 
                            header = T, sep = "\t", check.names = F)
    df_addition <- data.frame("coef"=train_out$`mode coef 1`,
                              "signif" = train_out$`mode signif 1`,
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    volcano_df <- rbind(volcano_df, df_addition)
    df_addition <- data.frame("coef"=train_out$`mode coef 2`,
                              "signif" = train_out$`mode signif 2`,
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    volcano_df <- rbind(volcano_df, df_addition)
  }
  colnames(volcano_df) <- c("coef", "signif", "partition", "n_p", "model")
  volcano_df$partition <- factor(volcano_df$partition, levels=c("Train", "Test"))
  volcano_df$n_p <- factor(volcano_df$n_p)
  volcano_df$model <- case_when(volcano_df$model == "model1" ~ "Model 1",
                                volcano_df$model == "model2" ~ "Model 2",
                                volcano_df$model == "model3" ~ "Model 3",
                                volcano_df$model == "null" ~ "Null Model",
                                volcano_df$model == "untrained" ~ "Untrained Model")
  volcano_df$signif[volcano_df$signif==0] <- .Machine$double.xmin
  volcano_df$signif <- -log(volcano_df$signif, base = 10)
  
  dir.create(file.path("Figures/volcano/"), showWarnings = FALSE)
  png(file=paste0("Figures/volcano/mode.png"), width = 3200, height = 1600)
  ggplot(volcano_df, aes(x=coef, y=signif, color=n_p)) + 
    geom_point(size=5) +
    theme_bw() + 
    xlab("Coefficient") +
    ylab(expression("-" ~ log[10] ~ "(p-value)")) +
    facet_grid(partition ~ model) + 
    geom_hline(yintercept=-log(0.05, 10), linetype="dashed", color = "red") +
    geom_vline(xintercept=0, linetype="dashed", color = "red") +
    scale_color_discrete(breaks=levels(volcano_df$n_p),
                        labels=c(expression("Without " ~ n[p] ~ "= 0"), expression("With " ~ n[p] ~ "= 0"))) + 
    theme(legend.title=element_blank(),
          text = element_text(size=50))
  dev.off()
}

volcano_plot_median <- function(){
  file_list <- list.files("train_outputs/analysis/")
  volcano_df <- data.frame(matrix(nrow = 0, ncol = 5))
  
  for (i in 1:length(file_list)) {
    train_out <- read.table(paste0("train_outputs/analysis/", file_list[i]), 
                            header = T, sep = "\t", check.names = F)
    df_addition <- data.frame("coef"=train_out$`median coef 1`,
                              "signif" = train_out$`median signif 1`,
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    volcano_df <- rbind(volcano_df, df_addition)
    df_addition <- data.frame("coef"=train_out$`median coef 2`,
                              "signif" = train_out$`median signif 2`,
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    volcano_df <- rbind(volcano_df, df_addition)
  }
  colnames(volcano_df) <- c("coef", "signif", "partition", "n_p", "model")
  volcano_df$partition <- factor(volcano_df$partition, levels=c("Train", "Test"))
  volcano_df$n_p <- factor(volcano_df$n_p)
  volcano_df$model <- case_when(volcano_df$model == "model1" ~ "Model 1",
                                volcano_df$model == "model2" ~ "Model 2",
                                volcano_df$model == "model3" ~ "Model 3",
                                volcano_df$model == "null" ~ "Null Model",
                                volcano_df$model == "untrained" ~ "Untrained Model")
  volcano_df$signif[volcano_df$signif==0] <- .Machine$double.xmin
  volcano_df$signif <- -log(volcano_df$signif, base = 10)
  
  dir.create(file.path("Figures/volcano/"), showWarnings = FALSE)
  png(file=paste0("Figures/volcano/median.png"), width = 3200, height = 1600)
  ggplot(volcano_df, aes(x=coef, y=signif, color=n_p)) + 
    geom_point(size=5) +
    theme_bw() + 
    xlab("Coefficient") +
    ylab(expression("-" ~ log[10] ~ "(p-value)")) +
    facet_grid(partition ~ model) + 
    geom_hline(yintercept=-log(0.05, 10), linetype="dashed", color = "red") +
    geom_vline(xintercept=0, linetype="dashed", color = "red") +
    scale_color_discrete(breaks=levels(volcano_df$n_p),
                         labels=c(expression("Without " ~ n[p] ~ "= 0"), expression("With " ~ n[p] ~ "= 0"))) + 
    theme(legend.title=element_blank(),
          text = element_text(size=50))
  dev.off()
}

# All models are almost identical for the next two functions, so only display one
auc_eval <- function(){
  file_list <- list.files("train_outputs/analysis/")
  auc_df <- data.frame(matrix(nrow = 0, ncol = 5))
  for (i in 1:length(file_list)) {
    train_out <- read.table(paste0("train_outputs/analysis/", file_list[i]), 
                            header = T, sep = "\t", check.names = F)
    df_addition <- data.frame("auc"=train_out$auc,
                              "f1" = train_out$f1,
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    auc_df <- rbind(auc_df, df_addition)
    
  }
  colnames(auc_df) <- c("auc", "f1", "partition", "n_p", "model")
  auc_df$partition <- factor(auc_df$partition, levels=c("Train", "Test"))
  auc_df$n_p <- factor(auc_df$n_p)
  auc_df$model <- case_when(auc_df$model == "model1" ~ "Model 1",
                            auc_df$model == "model2" ~ "Model 2",
                            auc_df$model == "model3" ~ "Model 3",
                            auc_df$model == "null" ~ "Null Model",
                            auc_df$model == "untrained" ~ "Untrained Model")
  
  auc_df <- auc_df[auc_df$model=="Model 1",]
  
  auc_df <- melt(auc_df, id.vars=c("partition", "n_p", "model"))
  colnames(auc_df) <- c("partition", "n_p", "model", "measure", "value")
  auc_df$measure <- ifelse(auc_df$measure == "auc", "AUC", "F1")
  
  dir.create(file.path("Figures/auc/"), showWarnings = FALSE)
  png(file=paste0("Figures/auc/auc.png"), width = 3200, height = 1600)
  ggplot(auc_df, aes(x=partition, y=value, fill=n_p)) + 
    geom_boxplot(lwd=2, outlier.size=5) +
    theme_bw() + 
    scale_y_continuous(
      breaks = seq(0,1,0.1),
    ) + 
    xlab("") +
    ylab("") +
    facet_grid(~measure) +
    scale_fill_discrete(breaks=levels(auc_df$n_p),
                        labels=c(expression("Without " ~ n[p] ~ "= 0"), expression("With " ~ n[p] ~ "= 0"))) + 
    theme(legend.title=element_blank(),
          text = element_text(size=80))
  dev.off()
}

weighed_metrics <- function(){
  partitions <- read.table("train_outputs/non_model/partitions.tsv", 
                         header = T, sep = "\t", check.names = F)
  partitions <- partitions[partitions$partition!="test",]
  metadata <- read.csv("metadata/metadata_parsed_100.tsv", sep = "\t", check.names = F, header = T)
  metadata <- metadata[metadata$ID %in% partitions$ID,-c(1:3)]
  
  props <- unlist(lapply(metadata, function(x){sum(!is.na(x) & x!="0" & x!=0)})) / nrow(metadata)
  
  file_list <- list.files("train_outputs/analysis/")
  weighted_df <- data.frame(matrix(nrow = 0, ncol = 6))
  for (i in 1:length(file_list)) {
    train_out <- read.table(paste0("train_outputs/analysis/", file_list[i]), 
                            header = T, sep = "\t", check.names = F)
    df_addition <- data.frame("weighted_f1"=sum(train_out$f1 * props, na.rm = T) / sum(props, na.rm = T),
                              "weighted_precision"=sum(train_out$precision * props, na.rm = T) / sum(props, na.rm = T),
                              "weighted_recall"=sum(train_out$recall * props, na.rm = T) / sum(props, na.rm = T),
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    weighted_df <- rbind(weighted_df, df_addition)
    
  }
  colnames(weighted_df) <- c("f1", "precision", "recall", "partition", "n_p", "model")
  weighted_df$partition <- factor(weighted_df$partition, levels=c("Train", "Test"))
  weighted_df$n_p <- factor(weighted_df$n_p)
  weighted_df$model <- case_when(weighted_df$model == "model1" ~ "Model 1",
                            weighted_df$model == "model2" ~ "Model 2",
                            weighted_df$model == "model3" ~ "Model 3",
                            weighted_df$model == "null" ~ "Null Model",
                            weighted_df$model == "untrained" ~ "Untrained Model")
  weighted_df <- weighted_df[weighted_df$model=="Model 1",]
  
  weighted_df <- melt(weighted_df, id.vars=c("partition", "n_p", "model"))
  colnames(weighted_df) <- c("partition", "n_p", "model", "measure", "value")
  weighted_df$measure <- case_when(weighted_df$measure == "f1" ~ "F1", 
                                   weighted_df$measure == "precision" ~ "Precision",
                                   weighted_df$measure == "recall" ~ "Recall")
  
  dir.create(file.path("Figures/weighted_metrics/"), showWarnings = FALSE)
  png(file=paste0("Figures/weighted_metrics/weighted_metrics.png"), width = 3200, height = 1600)
  ggplot(weighted_df, aes(x=partition, y=value, fill=n_p)) + 
    geom_bar(position="dodge", stat="identity")+
    theme_bw() + 
    scale_y_continuous(
      breaks = seq(0,0.5,0.1),
      limits = c(0,0.5)
    ) + 
    xlab("") +
    ylab("") +
    facet_grid(~measure) +
    scale_fill_discrete(breaks=levels(weighted_df$n_p),
                        labels=c(expression("Without " ~ n[p] ~ "= 0"), expression("With " ~ n[p] ~ "= 0"))) + 
    theme(legend.title=element_blank(),
          text = element_text(size=80))
  dev.off()
}

mean_abs_err <- function(){
  partitions <- read.table("train_outputs/non_model/partitions.tsv", 
                           header = T, sep = "\t", check.names = F)
  partitions <- partitions[partitions$partition!="test",]
  metadata <- read.csv("metadata/metadata_parsed_100.tsv", sep = "\t", check.names = F, header = T)
  metadata <- metadata[metadata$ID %in% partitions$ID,-c(1:3)]
  
  sds <- unlist(lapply(metadata, function(x){sd(x[!is.na(x) & x!="0" & x!=0])}))
  sds[(length(sds) - 2):(length(sds) + 3)] <- c(unlist(lapply(metadata[(ncol(metadata) - 2):ncol(metadata)], function(x){sd(strsplit(x[!is.na(x) & x!="0" & x!=0], " , ") %>%
                                                                                         sapply(FUN = "[[", 1) %>% as.numeric())})),
                                                unlist(lapply(metadata[(ncol(metadata) - 2):ncol(metadata)], function(x){sd(strsplit(x[!is.na(x) & x!="0" & x!=0], " , ") %>%
                                                                                                                              sapply(FUN = "[[", 2) %>% as.numeric())})))
  
  file_list <- list.files("train_outputs/analysis/")
  weighted_df <- data.frame(matrix(nrow = 0, ncol = 4))
  for (i in 1:length(file_list)) {
    train_out <- read.table(paste0("train_outputs/analysis/", file_list[i]), 
                            header = T, sep = "\t", check.names = F)
    df_addition <- data.frame("mean_abs_err"=c(train_out$`mean abs err 1`, train_out$`mean abs err 2`[!is.na(train_out$`mean abs err 2`)]) / sds,
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    weighted_df <- rbind(weighted_df, df_addition)
    
  }
  colnames(weighted_df) <- c("mean_abs_err", "partition", "n_p", "model")
  weighted_df$partition <- factor(weighted_df$partition, levels=c("Train", "Test"))
  weighted_df$n_p <- factor(weighted_df$n_p)
  weighted_df$model <- case_when(weighted_df$model == "model1" ~ "Model 1",
                                 weighted_df$model == "model2" ~ "Model 2",
                                 weighted_df$model == "model3" ~ "Model 3",
                                 weighted_df$model == "null" ~ "Null Model",
                                 weighted_df$model == "untrained" ~ "Untrained Model")

  dir.create(file.path("Figures/abs_err/"), showWarnings = FALSE)
  png(file=paste0("Figures/abs_err/mean_abs_err.png"), width = 3200, height = 1600)
  ggplot(weighted_df, aes(x=partition, y=mean_abs_err, fill=n_p)) + 
    geom_boxplot(lwd=2, outlier.size=5)+
    theme_bw() + 
    xlab("") +
    ylab("Standard Deviations") +
    ggtitle("Absolute error in standard deviations") +
    facet_grid(~model) +
    scale_fill_discrete(breaks=levels(weighted_df$n_p),
                        labels=c(expression("Without " ~ n[p] ~ "= 0"), expression("With " ~ n[p] ~ "= 0"))) + 
    theme(legend.title=element_blank(),
          text = element_text(size=80))
  dev.off()
}

median_abs_err <- function(){
  partitions <- read.table("train_outputs/non_model/partitions.tsv", 
                           header = T, sep = "\t", check.names = F)
  partitions <- partitions[partitions$partition!="test",]
  metadata <- read.csv("metadata/metadata_parsed_100.tsv", sep = "\t", check.names = F, header = T)
  metadata <- metadata[metadata$ID %in% partitions$ID,-c(1:3)]
  
  sds <- unlist(lapply(metadata, function(x){sd(x[!is.na(x) & x!="0" & x!=0])}))
  sds[(length(sds) - 2):(length(sds) + 3)] <- c(unlist(lapply(metadata[(ncol(metadata) - 2):ncol(metadata)], function(x){sd(strsplit(x[!is.na(x) & x!="0" & x!=0], " , ") %>%
                                                                                                                              sapply(FUN = "[[", 1) %>% as.numeric())})),
                                                unlist(lapply(metadata[(ncol(metadata) - 2):ncol(metadata)], function(x){sd(strsplit(x[!is.na(x) & x!="0" & x!=0], " , ") %>%
                                                                                                                              sapply(FUN = "[[", 2) %>% as.numeric())})))
  
  file_list <- list.files("train_outputs/analysis/")
  weighted_df <- data.frame(matrix(nrow = 0, ncol = 4))
  for (i in 1:length(file_list)) {
    train_out <- read.table(paste0("train_outputs/analysis/", file_list[i]), 
                            header = T, sep = "\t", check.names = F)
    df_addition <- data.frame("median_abs_err"=c(train_out$`median abs err 1`, train_out$`median abs err 2`[!is.na(train_out$`median abs err 2`)]) / sds,
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    weighted_df <- rbind(weighted_df, df_addition)
    
  }
  colnames(weighted_df) <- c("median_abs_err", "partition", "n_p", "model")
  weighted_df$partition <- factor(weighted_df$partition, levels=c("Train", "Test"))
  weighted_df$n_p <- factor(weighted_df$n_p)
  weighted_df$model <- case_when(weighted_df$model == "model1" ~ "Model 1",
                                 weighted_df$model == "model2" ~ "Model 2",
                                 weighted_df$model == "model3" ~ "Model 3",
                                 weighted_df$model == "null" ~ "Null Model",
                                 weighted_df$model == "untrained" ~ "Untrained Model")
  
  dir.create(file.path("Figures/abs_err/"), showWarnings = FALSE)
  png(file=paste0("Figures/abs_err/median_abs_err.png"), width = 3200, height = 1600)
  ggplot(weighted_df, aes(x=partition, y=median_abs_err, fill=n_p)) + 
    geom_boxplot(lwd=2, outlier.size=5)+
    theme_bw() + 
    xlab("") +
    ylab("Standard Deviations") +
    ggtitle("Absolute error in standard deviations") +
    facet_grid(~model) +
    scale_fill_discrete(breaks=levels(weighted_df$n_p),
                        labels=c(expression("Without " ~ n[p] ~ "= 0"), expression("With " ~ n[p] ~ "= 0"))) + 
    theme(legend.title=element_blank(),
          text = element_text(size=80))
  dev.off()
}

mode_abs_err <- function(){
  partitions <- read.table("train_outputs/non_model/partitions.tsv", 
                           header = T, sep = "\t", check.names = F)
  partitions <- partitions[partitions$partition!="test",]
  metadata <- read.csv("metadata/metadata_parsed_100.tsv", sep = "\t", check.names = F, header = T)
  metadata <- metadata[metadata$ID %in% partitions$ID,-c(1:3)]
  
  sds <- unlist(lapply(metadata, function(x){sd(x[!is.na(x) & x!="0" & x!=0])}))
  sds[(length(sds) - 2):(length(sds) + 3)] <- c(unlist(lapply(metadata[(ncol(metadata) - 2):ncol(metadata)], function(x){sd(strsplit(x[!is.na(x) & x!="0" & x!=0], " , ") %>%
                                                                                                                              sapply(FUN = "[[", 1) %>% as.numeric())})),
                                                unlist(lapply(metadata[(ncol(metadata) - 2):ncol(metadata)], function(x){sd(strsplit(x[!is.na(x) & x!="0" & x!=0], " , ") %>%
                                                                                                                              sapply(FUN = "[[", 2) %>% as.numeric())})))
  
  file_list <- list.files("train_outputs/analysis/")
  weighted_df <- data.frame(matrix(nrow = 0, ncol = 4))
  for (i in 1:length(file_list)) {
    train_out <- read.table(paste0("train_outputs/analysis/", file_list[i]), 
                            header = T, sep = "\t", check.names = F)
    df_addition <- data.frame("mode_abs_err"=c(train_out$`mode abs err 1`, train_out$`mode abs err 2`[!is.na(train_out$`mode abs err 2`)]) / sds,
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    weighted_df <- rbind(weighted_df, df_addition)
    
  }
  colnames(weighted_df) <- c("mode_abs_err", "partition", "n_p", "model")
  weighted_df$partition <- factor(weighted_df$partition, levels=c("Train", "Test"))
  weighted_df$n_p <- factor(weighted_df$n_p)
  weighted_df$model <- case_when(weighted_df$model == "model1" ~ "Model 1",
                                 weighted_df$model == "model2" ~ "Model 2",
                                 weighted_df$model == "model3" ~ "Model 3",
                                 weighted_df$model == "null" ~ "Null Model",
                                 weighted_df$model == "untrained" ~ "Untrained Model")
  
  dir.create(file.path("Figures/abs_err/"), showWarnings = FALSE)
  png(file=paste0("Figures/abs_err/mode_abs_err.png"), width = 3200, height = 1600)
  ggplot(weighted_df, aes(x=partition, y=mode_abs_err, fill=n_p)) + 
    geom_boxplot(lwd=2, outlier.size=5)+
    theme_bw() + 
    xlab("") +
    ylab("Standard Deviations") +
    ggtitle("Absolute error in standard deviations") +
    facet_grid(~model) +
    scale_fill_discrete(breaks=levels(weighted_df$n_p),
                        labels=c(expression("Without " ~ n[p] ~ "= 0"), expression("With " ~ n[p] ~ "= 0"))) + 
    theme(legend.title=element_blank(),
          text = element_text(size=80))
  dev.off()
}

density_interval <- function(){
  partitions <- read.table("train_outputs/non_model/partitions.tsv", 
                           header = T, sep = "\t", check.names = F)
  partitions <- partitions[partitions$partition!="test",]
  metadata <- read.csv("metadata/metadata_parsed_100.tsv", sep = "\t", check.names = F, header = T)
  metadata <- metadata[metadata$ID %in% partitions$ID,-c(1:3)]
  
  sds <- unlist(lapply(metadata, function(x){sd(x[!is.na(x) & x!="0" & x!=0])}))
  sds[(length(sds) - 2):(length(sds) + 3)] <- c(unlist(lapply(metadata[(ncol(metadata) - 2):ncol(metadata)], function(x){sd(strsplit(x[!is.na(x) & x!="0" & x!=0], " , ") %>%
                                                                                                                              sapply(FUN = "[[", 1) %>% as.numeric())})),
                                                unlist(lapply(metadata[(ncol(metadata) - 2):ncol(metadata)], function(x){sd(strsplit(x[!is.na(x) & x!="0" & x!=0], " , ") %>%
                                                                                                                              sapply(FUN = "[[", 2) %>% as.numeric())})))
  
  file_list <- list.files("train_outputs/analysis/")
  weighted_df <- data.frame(matrix(nrow = 0, ncol = 4))
  for (i in 1:length(file_list)) {
    train_out <- read.table(paste0("train_outputs/analysis/", file_list[i]), 
                            header = T, sep = "\t", check.names = F)
    df_addition <- data.frame("density_interval"=c(train_out$`mean CI width 1`, train_out$`mean CI width 2`[!is.na(train_out$`mean CI width 2`)]) / sds,
                              "nominal" = gsub(".*ci_", "", file_list[i]) %>% gsub(pattern="[^0-9].*", replacement="") %>% as.numeric(),
                              "partition" = ifelse(grepl("val", file_list[i]), "Test", "Train"),
                              "n_p" = ifelse(grepl("only_sim", file_list[i]), "first", "second"),
                              "model" = gsub("_.*", "", file_list[i]))
    weighted_df <- rbind(weighted_df, df_addition)
    
  }
  colnames(weighted_df) <- c("density_interval", "nominal", "partition", "n_p", "model")
  weighted_df$partition <- factor(weighted_df$partition, levels=c("Train", "Test"))
  weighted_df$n_p <- factor(weighted_df$n_p)
  weighted_df$nominal <- paste0(weighted_df$nominal, "% Density interval")
  weighted_df$model <- case_when(weighted_df$model == "model1" ~ "Model 1",
                                 weighted_df$model == "model2" ~ "Model 2",
                                 weighted_df$model == "model3" ~ "Model 3",
                                 weighted_df$model == "null" ~ "Null Model",
                                 weighted_df$model == "untrained" ~ "Untrained Model")
  
  dir.create(file.path("Figures/density_interval/"), showWarnings = FALSE)
  png(file=paste0("Figures/density_interval/density_interval.png"), width = 3200, height = 1600)
  ggplot(weighted_df, aes(x=partition, y=density_interval, fill=n_p)) + 
    geom_boxplot(lwd=2, outlier.size=5)+
    theme_bw() + 
    xlab("") +
    ylab("Standard Deviations") +
    facet_grid(nominal~model) +
    scale_fill_discrete(breaks=levels(weighted_df$n_p),
                        labels=c(expression("Without " ~ n[p] ~ "= 0"), expression("With " ~ n[p] ~ "= 0"))) + 
    theme(legend.title=element_blank(),
          text = element_text(size=80))
  dev.off()
}





