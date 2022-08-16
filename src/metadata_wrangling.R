remove(list=ls())

start.time = Sys.time()

library(dplyr)
library(stringr)
library(data.table)

# Import metadata
metadata <- read.csv("metadata/crystal_metadata.tsv", sep = "\t", header = F)
colnames(metadata) <- c('ID', 'date', 'crystal_id', 'details', 'method', 'p_h', 'pdbx_details', 'pdbx_phrange', 'temp', 'temp_details')
metadata <- metadata[,c(1, 2, 7, 3, 4, 5, 6, 8, 9, 10)]
metadata[metadata==""] <- NA_character_

partitions <- read.table("train_outputs/non_model/partitions.tsv", 
                         header = T, sep = "\t", check.names = F)
metadata <- metadata[metadata$ID %in% partitions$ID,]

# Remove crystal_id because it is not relevant metadata
metadata$crystal_id <- NULL

# Only contains 6 items, redundant
metadata$details <- NULL

# Temp details provides no more info than temp
metadata$temp_details <- NULL

########################
# Standardize "method" #
########################

# 116713 entries with "method"
# These will be regrouped if possible and everything else will be left to an "other" category

metadata <- metadata %>% mutate(
  method = method %>% tolower()) %>% mutate(
    method = case_when(method == "vapor diffusion, hanging drop" ~ "vapor diffusion",
                       method == "vapor diffusion, sitting drop" ~ "vapor diffusion",
                       method == "vapor diffusion" ~ "vapor diffusion",
                       method == "evaporation" ~ "evaporation",
                       method == "microbatch" ~ "batch",
                       method == "lipidic cubic phase" ~ "lipidic cubic phase",
                       method == "batch mode" ~ "batch",
                       method == "hanging drop" ~ "vapor diffusion",
                       method == "microdialysis" ~ "microdialysis",
                       method == "small tubes" ~ "vapor diffusion", # Based on the structures that used this, it appears to just be vapor diffusion
                       method == "liquid diffusion" ~ "liquid diffusion",
                       method == "batch" ~ "batch",
                       method == "microbatch under oil" ~ "batch",
                       method == "vapor diffusion,sitting drop,nanodrop" ~ "vapor diffusion",
                       method == "counter-diffusion" ~ "counter diffusion",
                       method == "sitting drop" ~ "vapor diffusion",
                       method == "vapor diffusion, sitting drop, nanodrop" ~ "vapor diffusion",
                       method == "hanging drop vapor diffusion" ~ "vapor diffusion",
                       method == "batch method" ~ "batch",
                       method == "in cell" ~ "cell",
                       method == "hanging drop, vapor diffusion" ~ "vapor diffusion",
                       method == "evaporation, recrystallization" ~ "evaporation",
                       method == "sitting drop vapor diffusion" ~ "vapor diffusion",
                       method == "evaporation, hanging drop" ~ "evaporation",
                       method == "microbatch crystallization under oil" ~ "batch",
                       method == "slow cooling" ~ "cooling",
                       method == "modified microbatch" ~ "batch",
                       method == "microbach" ~ "batch",
                       method == "microbatch under paraffin oil" ~ "batch",
                       method == "sitting-drop vapor diffusion" ~ "vapor diffusion",
                       method == "microbatch, under oil" ~ "batch",
                       method == "batch crystallization" ~ "batch",
                       method == "oil microbatch" ~ "batch",
                       method == "microbatch underoil" ~ "batch",
                       method == "sitting drop vapor diffuction" ~ "vapor diffusion",
                       method == "under oil" ~ "batch",
                       method == "cubic lipid phase" ~ "lipidic cubic phase",
                       method == "oil batch" ~ "batch",
                       method == "vapour diffusion, hanging drop" ~ "vapor diffusion",
                       method == "dialysis" ~ "dialysis",
                       method == "sitting drop, vapor diffusion" ~ "vapor diffusion",
                       method == "counter diffusion" ~ "counter diffusion",
                       method == "lipidic cubic phase (lcp)" ~ "lipidic cubic phase",
                       method == "lipid cubic phase" ~ "lipidic cubic phase",
                       method == "oil-micro batch" ~ "batch",
                       method == "microseeding" ~ "seeding",
                       method == "macroseeding" ~ "seeding",
                       method == "soaking" ~ "soaking",
                       method == "microbatch crystallization" ~ "batch",
                       method == "microbatch technique under oil" ~ "batch",
                       method == "microfluidic" ~ "microfluidic",
                       method == "capillary counterdiffusion" ~ "counter diffusion",
                       grepl("vapor", method) ~ "vapor diffusion",
                       grepl("batch", method) ~ "batch",
                       grepl("evaporation", method) ~ "evaporation",
                       grepl("lipid", method) ~ "lipidic cubic phase",
                       grepl("lcp", method) ~ "lipidic cubic phase",
                       grepl("oil", method) ~ "batch",
                       grepl("dialysis", method) ~ "microdialysis",
                       grepl("liquid diffusion", method) ~ "liquid diffusion",
                       grepl("dialysis", method) ~ "microdialysis",
                       grepl("cooling", method) ~ "cooling",
                       grepl("capillary", method) ~ "counter diffusion",
                       method == "" ~ NA_character_,
                       is.na(method) ~ NA_character_,
                       TRUE ~ "other")
  )

# 184 other remaining, 116529 of 116713 (99.8%) classified

#####################
# Standardize "p_h" #
#####################

# Function to check if the reported pH is between the bounds if provided
check_between <- function(ph,bounds) {
  bounds = suppressWarnings(as.numeric(bounds))
  
  # If no p_h, put the average of the bounds
  if (is.na(ph)) {return(mean(c(bounds[1], bounds[2])))}
  
  # If there are not both upper and lower bounds, leave the pH
  if (sum(is.na(bounds) | length(bounds) != 2) > 0) {return(ph)}
  
  # If the pH is between the bounds, leave the pH
  if (bounds[1]<=ph & ph<=bounds[2]) {return(ph)}
  
  # If the pH is outside the bounds, put the average of the bounds
  return(mean(c(bounds[1], bounds[2])))
}

# Keep only numbers and separators
tmp1 <- str_extract(metadata$pdbx_phrange, "[0-9\\.\\-,; ]+")

# Split into a lower and upper bound
ph_range <- str_split(tmp1[!is.na(tmp1)], "[^0-9.]")

# Standardize the pH
replacement = mapply(check_between, metadata$p_h[!is.na(tmp1)], ph_range)
metadata$p_h[!is.na(tmp1)] = case_when(replacement < 2 ~ metadata$p_h[!is.na(tmp1)],
                                       replacement > 12 ~ metadata$p_h[!is.na(tmp1)],
                                       is.na(replacement) ~ metadata$p_h[!is.na(tmp1)],
                                       TRUE ~ replacement)

metadata$pdbx_phrange <- NULL

# 106220 had pH before, 107615 have pH after (1.3% improvement)

###############################
# Extract chemical conditions #
###############################

stored_pdbx_details = metadata$pdbx_details

# 115254 initially have conditions

# NA pdbx_details with no numerics or with conditions in a problematic order
# 109194 (94.7%) remain
metadata$pdbx_details <- ifelse(grepl("[0-9]", metadata$pdbx_details, perl = T), 
                                ifelse(grepl(pattern = "^[0-9]", metadata$pdbx_details, perl = T), metadata$pdbx_details, NA), NA)

# Functions to deal with ranges
process_dash <- function(x) {
  to_reformat = lapply(str_split(x, "(?<=[0-9])\\-(?=[0-9])"), length) >= 2
  reformatted = lapply(str_split(x, "(?<=[0-9])\\-(?=[0-9])"), mean_of_sides) %>% unlist()
  return (ifelse(to_reformat, reformatted, x))
}

mean_of_sides <- function(y) {
  replacements = vector(length = length(y))
  for (i in 1:(length(y) - 1)) {
    replacements[i] = (mean(c(str_extract(y[i], "[0-9\\.]*$") %>% as.numeric(), str_extract(y[i + 1], "^[0-9\\.]*") %>% as.numeric())))
    y[i] <- gsub("[0-9\\.]*$", "", y[i], perl = T)
    y[i+1] <- gsub("^[0-9\\.]*", "", y[i+1], perl = T)
  }
  replacements[length(y)] = ""
  return(paste(c(rbind(y, replacements)), collapse = ''))
}

process_to <- function(x) {
  to_reformat = lapply(str_split(x, "(?<=[0-9]) to (?=[0-9])"), length) >= 2
  reformatted = lapply(str_split(x, "(?<=[0-9]) to (?=[0-9])"), mean_of_sides) %>% unlist()
  return (ifelse(to_reformat, reformatted, x))
}

# Initial text cleaning
exclude_words = c(" with", "in", "per", "system", "a", "well", "fragment", "this", "compound", "saturated", "and", "hcl", "hydrochloride", "dibasic", "monobasic", "tribasic", "naoh", "koh", "of", "at ")
metadata = metadata %>%
  mutate(
    pdbx_details = pdbx_details %>% 
      tolower() %>%
      gsub(pattern="\\\n", replacement=" ") %>% 
      gsub(pattern=", ", replacement=" ") %>%
      gsub(pattern="\\: ", replacement=" ") %>% 
      gsub(pattern="_", replacement=" ") %>% 
      gsub(pattern="\\;", replacement=" ") %>% 
      gsub(pattern="\\(", replacement="") %>%
      gsub(pattern="\\)", replacement="") %>%
      gsub(pattern=" - ", replacement="-") %>%
      gsub(pattern="\\&", replacement=" and ") %>% 
      gsub(pattern="\\+", replacement=" and ") %>%
      gsub(pattern=" -- ", replacement=" ") %>% 
      gsub(pattern="\\*", replacement=" ") %>% 
      gsub(pattern=" -", replacement="-") %>% 
      gsub(pattern="- ", replacement="-") %>% 
      gsub(pattern="\\.$", replacement="") %>% 
      gsub(pattern="\\[", replacement=" ") %>% 
      gsub(pattern="\\]", replacement=" ") %>% 
      gsub(pattern="   ", replacement = " ") %>%
      gsub(pattern="  ", replacement = " ") %>%
      gsub(pattern="\\/sodium hydroxide", replacement = "") %>%
      gsub(pattern="\\/hydrochloric acid", replacement = "") %>%
      gsub(pattern="[0-9][ \\-]h2o", replacement = " ") %>%
      gsub(pattern="[0-9]*\\(h2o\\)[0-9]*", replacement = " ") %>%
      gsub(pattern="[ ]*percent", replacement = "\\%", perl = T) %>%
      gsub(pattern="\\%(?=[a-z])", replacement = "% ", perl = TRUE) %>%
      gsub(pattern="\\-hcl|\\:hcl|\\/hcl", replacement = "", perl = TRUE) %>%
      gsub(pattern="\\-naoh|\\:naoh|\\/naoh", replacement = "", perl = TRUE) %>%
      gsub(pattern="\\-koh|\\:koh|\\/koh", replacement = "", perl = TRUE) %>%
      gsub(pattern="microliter|microliters|microlitre|microlitres|mikroliter|micro\\-liter|micro\\-liters|micoliter|micro\\-l", replacement = "ul", perl = T) %>%
      gsub(pattern="milliliter|milliliters", replacement = "ml", perl = T) %>%
      gsub(pattern="nanoliter|nanoliters|nanolitre", replacement = "nl", perl = T) %>%
      gsub(pattern="liter|liters|litre|litres", replacement = "l", perl = T) %>%
      gsub(pattern="millimolar|mmol|milli\\-molar|millim|milli\\-m", replacement = "mm", perl = T) %>%
      gsub(pattern="micromolar|umol|micro\\-m|microm", replacement = "um", perl = T) %>%
      gsub(pattern=" molar | molar$", replacement = "m", perl = T) %>%
      gsub(pattern="milligram|milligrams", replacement = "mg", perl = T) %>%
      gsub(pattern="microgram|micrograms", replacement = "ug", perl = T) %>%
      gsub(pattern="gram|grams", replacement = "mg", perl = T) %>%
      gsub(pattern="microl", replacement = "ul", perl = T) %>%
      gsub(pattern="sulphate", replacement = "sulfate", perl = T) %>%
      gsub(pattern="(?<=[0-9]),(?=[0-9]{3})", replacement="", perl = TRUE) %>%
      gsub(pattern="(?<=[0-9][0-9]),(?=[0-9]{3})", replacement="", perl = TRUE) %>%
      gsub(pattern="(?<=[0-9]) *(?=%)", replacement = "", perl = T) %>%
      gsub(pattern = paste0(exclude_words, collapse = " | "), replacement = " ", perl = T) %>%
      gsub(pattern="4 po\\/oh", replacement="4-pooh", perl = T) %>%
      gsub(pattern="4 eo\\/oh", replacement="4-eooh", perl = T) %>%
      gsub(pattern="polyethylene glycol", replacement = "peg", perl = TRUE) %>%
      gsub(pattern="poly ethylene glycol", replacement = "peg", perl = TRUE) %>%
      gsub(pattern="monomethyl[- ]peg", replacement = "peg mme", perl = TRUE) %>%
      gsub(pattern="monomethyl ether", replacement = "mme", perl = TRUE) %>%
      gsub(pattern="(?<=peg) *peg", replacement = "", perl = TRUE) %>%
      gsub(pattern="(?<=mme) *mme", replacement = "", perl = TRUE) %>%
      gsub(pattern="peg ([0-9][0-9\\%]*) mme", replacement = "peg mme\\1 ", perl = TRUE) %>%
      gsub(pattern="peg ([0-9][0-9\\%]*)mme", replacement = "peg mme\\1 ", perl = TRUE) %>%
      gsub(pattern="peg([0-9][0-9\\%]*) mme", replacement = "peg mme\\1 ", perl = TRUE) %>%
      gsub(pattern="peg([0-9][0-9\\%]*)mme", replacement = "peg mme\\1 ", perl = TRUE) %>%
      gsub(pattern="(?<=peg)[ \\-](?=[0-9][0-9]*[^\\%])", replacement = "", perl = TRUE) %>%
      gsub(pattern="(?<=peg)([0-9]*?)k", replacement = "\\1000", perl = TRUE) %>%
      gsub(pattern="(?<=peg mme)[ \\-](?=[0-9])", replacement = "", perl = TRUE) %>%
      gsub(pattern="(?<=mme)([0-9]*?)k", replacement = "\\1000", perl = TRUE) %>%
      gsub(pattern="(?<=glycol)[ \\-](?=[0-9]{3})", replacement = "", perl = TRUE) %>%
      gsub(pattern="(?<=ether)[ \\-](?=[0-9])", replacement = "", perl = TRUE) %>%
      gsub(pattern="(?<=peg me)[ \\-](?=[0-9])", replacement = "", perl = TRUE) %>%
      gsub(pattern="(?<= ph) (?=[0-9])", replacement = "", perl = TRUE) %>%
      gsub(pattern="(?<= ph)=(?=[0-9])", replacement = "", perl = TRUE) %>%
      gsub(pattern="^ph (?=[0-9])", replacement = "ph", perl = TRUE) %>%
      gsub(pattern="temperature [0-9\\.]{3,}k", replacement = "", perl = TRUE) %>%
      gsub(pattern="[0-9\\.]{3,}k", replacement = "", perl = TRUE) %>%
      gsub(pattern = "vapor diffusion hanging drop", replacement = "") %>%
      gsub(pattern = "vapor diffusion sitting drop", replacement = "") %>%
      gsub(pattern = "vapor diffusion", replacement = "") %>%
      gsub(pattern = "hanging drop", replacement = "") %>%
      gsub(pattern = "sitting drop", replacement = "") %>%
      gsub(pattern = "crystals obtained by streak-seeding", replacement = "") %>%
      gsub(pattern=", ", replacement=" ") %>% # Second pass to replace any new quirks that came up
      gsub(pattern="(?<=[a-z]),(?=[0-9])", replacement=" ", perl = T) %>%
      gsub(pattern="(?<=[0-9]),(?=[a-z])", replacement=" ", perl = T) %>%
      gsub(pattern="(?<=[a-z]),(?=[a-z])", replacement=" ", perl = T) %>%
      gsub(pattern="(?<=[a-z])\\-(?=[a-z])", replacement=" ", perl = T) %>%
      gsub(pattern="\\: ", replacement=" ") %>% 
      gsub(pattern="\\;", replacement=" ") %>% 
      gsub(pattern="\\(", replacement="") %>%
      gsub(pattern="\\)", replacement="") %>%
      gsub(pattern=" - ", replacement="-") %>%
      gsub(pattern="\\&", replacement="and") %>% 
      gsub(pattern="\\+", replacement="and") %>%
      gsub(pattern=" -- ", replacement=" ") %>% 
      gsub(pattern=" -", replacement="-") %>% 
      gsub(pattern="- ", replacement="-") %>% 
      gsub(pattern="\\.$", replacement="", perl = T) %>% 
      gsub(pattern="([^0-9])\\.([^0-9])", replacement="\\1 \\2", perl = T) %>% 
      gsub(pattern="\\%([^ ])", replacement="\\% \\1", perl = T) %>% 
      gsub(pattern="peg[ \\-]([0-9][0-9]*)[ \\-]mme", replacement = "peg mme\\1 ", perl = TRUE) %>%
      gsub(pattern="peg[ \\-]([0-9][0-9]*)mme", replacement = "peg mme\\1 ", perl = TRUE) %>%
      gsub(pattern="peg([0-9][0-9]*)[ \\-]mme", replacement = "peg mme\\1 ", perl = TRUE) %>%
      gsub(pattern="peg([0-9][0-9]*)mme", replacement = "peg mme\\1 ", perl = TRUE) %>%
      gsub(pattern="pegmme", replacement = "peg mme", perl = TRUE) %>%
      gsub(pattern="mme peg", replacement = "peg mme", perl = TRUE) %>%
      gsub(pattern="peg000", replacement = "", perl = TRUE) %>%
      gsub(pattern=" [^\\s]*hydrate[^\\s]* | [^\\s]*hydrate$", replacement = " ", perl = TRUE) %>%
      gsub(pattern="([0-9])\\. |([0-9])\\.$", replacement = "\\1 ", perl = TRUE) %>%
      gsub(pattern = "([0-9\\.]*[0-9\\.])([a-z%])", replacement = "\\1 \\2", perl = T) %>%
      gsub(pattern = "(peg[0-9]+) ([0-9\\.]+[ ]*\\%$)", replacement = "\\2 \\1", perl = T) %>%
      gsub(pattern = paste0(exclude_words, collapse = " | "), replacement = " ", perl = T) %>%
      gsub(pattern="   ", replacement = " ") %>%
      gsub(pattern="  ", replacement = " ") %>%
      gsub(pattern=" $", replacement = ""),
  ) %>%
  mutate(
    pdbx_details = case_when(pdbx_details == "" ~ NA_character_,
                             pdbx_details == " " ~ NA_character_,
                             pdbx_details == "null" ~ NA_character_,
                             TRUE ~ pdbx_details)
  ) %>%
  mutate(
    pdbx_details = process_dash(pdbx_details) %>%
      gsub(pattern = "([0-9])NA", replacement = "\\1")
  ) %>%
  mutate(
    pdbx_details = process_to(pdbx_details) %>%
      gsub(pattern = "([0-9])NA", replacement = "\\1")
  )

# If p_h is NA and the details have a pH, use that (salvages 5911 (5.5%) pH values)
ph_replacement = ifelse(grepl(".*ph([0-9\\.]+).*", metadata$pdbx_details), gsub(".*ph([0-9\\.]+).*", "\\1", metadata$pdbx_details), NA) %>% as.numeric()
metadata$p_h <- case_when(ph_replacement > 14 | ph_replacement < -2 ~ metadata$p_h,
                          is.na(metadata$p_h) ~ ph_replacement,
                          TRUE ~ metadata$p_h)
metadata$pdbx_details = metadata$pdbx_details %>%
  gsub(pattern=" ph[0-9][0-9\\.\\-]*[,]*", replacement = " ", perl = TRUE) %>%
  gsub(pattern="^ph[0-9][0-9\\.\\-]*[,]*", replacement = " ", perl = TRUE)

# 109193 (94.7% of the original) are non-NA after the initial text cleaning

# Begin processing each condition

# NA pdbx_details with no numerics
# 103781 (90.0%) remain
metadata$pdbx_details <- ifelse(grepl("\\s+(?=[0-9\\.]* )", metadata$pdbx_details, perl = T), metadata$pdbx_details, NA)

# 363185 conditions present
conditions_list = metadata$pdbx_details %>% str_split("\\s+(?=[0-9\\.]* )")

# Remove conditions with no number (can't determine concentration)
remove_non_numeric <- function(x) {
  return (x[grepl("[0-9]", x)])
}

conditions_list <- lapply(conditions_list, remove_non_numeric)

# 333287 (91.8%) conditions remain after removing non-numeric

# Remove conditions with no units (or only units but no condition)
units_vec <- c(" % w\\/w", "% w\\/v", "% v\\/v", "w\\/w", "w\\/v", "v\\/v", "\\/v", "% \\/v", "%", "m", "mm", "mg", "mg\\/ml", "ul", "ml", "um", "nl ")

remove_no_units <- function(x) {
  return (x[grepl(paste0(units_vec, collapse = " | "), x)])
}

conditions_list <- lapply(conditions_list, remove_no_units)

# For determining how many proteins keep all their conditions
num_conditions_each = lapply(conditions_list, (function(x) length(x[!is.na(x)]))) %>% unlist() 

metadata$pdbx_details <- ifelse(unlist(lapply(conditions_list, length))>0, metadata$pdbx_details, NA)
# 103762 (90.0%) remain non-NA

# 327376 (90.1%) conditions remain after removing no units

# Function to extract conditions
get_conditions <- function(x) {
  tmp <- str_split(string = x, paste0(units_vec, collapse = " | "), n = 2)
  sub_function <- function(y) {
    if (length(y) > 1) {
      return(y[2])
    } else {
      return ("")
    }
  }
  lapply(tmp, sub_function) %>% unlist()
}

# Common words to remove
end_line1 <- c("^ solution", "buffer", "the", "to", "containing", "along", "against", "screen", "condition", "concentration", "was", "were", "mixed", "crystals", "for", "a", "soaked", "by", "well", "added", "ph", "stock", "nanodrop", "crystal$ ")
end_line1 <- paste0(c(paste0(gsub("\\$| ", "", end_line1, perl = T), collapse = " .*|^"), paste0(gsub("\\$|\\^", "", end_line1, perl = T), collapse = " .*| "), paste0(gsub("\\^| ", "", end_line1, perl = T), collapse = "$| ")), collapse = ".*| ")
end_line2 <- c("^ reservoir", "microbatch", "ratio", "as", "1:1", "base", "crystallization", "drop", "di", "tri", "precipitant", "under", "equal", "temperature", "or", "from", "an", "each", "batch", "contained", "after", "volume", "stored", "diffusion$ ")
end_line2 <- paste0(c(paste0(gsub("\\$| ", "", end_line2, perl = T), collapse = " .*|^"), paste0(gsub("\\$|\\^", "", end_line2, perl = T), collapse = " .*| "), paste0(gsub("\\^| ", "", end_line2, perl = T), collapse = "$| ")), collapse = ".*| ")
end_line3 <- c("^ using", "supplemented", "ratio", "evaporation", "over", "method", "then", "soaking", "plus", "final", "equilibrated", "tray", "=", "drops", "complex", "mix", "mixture", "additive", "incubated", "room", "all", "grown", "before$ ")
end_line3 <- paste0(c(paste0(gsub("\\$| ", "", end_line3, perl = T), collapse = " .*|^"), paste0(gsub("\\$|\\^", "", end_line3, perl = T), collapse = " .*| "), paste0(gsub("\\^| ", "", end_line3, perl = T), collapse = "$| ")), collapse = ".*| ")

process_conditions <- function(x) {
  x %>% 
    gsub(pattern = paste0(end_line1, collapse = " | "), replacement = "", perl = T) %>%
    gsub(pattern = paste0(end_line2, collapse = " | "), replacement = "", perl = T) %>%
    gsub(pattern = paste0(end_line3, collapse = " | "), replacement = "", perl = T) %>%
    gsub(pattern = "w/v|reservoir|solution|obtainedstreak-seeding|buffer|plus|precipitant|tray|fragment|nanodrop|protein|temperature", replacement = "", perl = T) %>%
    gsub(pattern=",,", replacement=" ") %>%
    gsub(pattern="\\'", replacement=" ") %>%
    gsub(pattern=", ", replacement=" ") %>%
    gsub(pattern = "  ", replacement = " ", perl = T) %>%
    gsub(pattern = "\\.$", replacement = "", perl = T) %>%
    gsub(pattern = " $", replacement = "", perl = T) %>%
    ifelse(.=="", NA_character_, .) %>%
    return()
}

# ~1000 items per second
conditions <- lapply(conditions_list, get_conditions) %>% lapply(FUN = process_conditions)

# List of conditions to possibly add
pos_add <- names(conditions %>% unlist() %>% table() %>% sort(decreasing = T))
names(pos_add) <- pos_add
# 12754 terms to possibly add

# Read renamed conditions from tsv and replace
replacement_tmp <- read.table("metadata/renamed.tsv", sep = "\t", header = T)
colnames(replacement_tmp) <- c("original", "replacement")
replacement_vec <- replacement_tmp$replacement
names(replacement_vec) <- replacement_tmp$original

replacement_vec <- c(replacement_vec, pos_add[!names(pos_add) %chin% names(replacement_vec)])

replace_conditions <- function(x) {
  replacement_vec[x] %>% unname() %>% return()
}

# ~2000 items per second
conditions <- lapply(conditions, replace_conditions)

# Replaces original with renamed values
units_sub <- units_vec
units_sub[1] <- paste0("(?<=", units_sub[1], collapse = "")
units_sub[length(units_sub)] <- paste0(units_sub[length(units_sub)], ")", collapse = "")
merge_and_check_conditions <- function(original, updated) {
  tmp <- str_split(string = original, paste0(units_sub, collapse = " )|(?<= "), n = 2)
  sub_function <- function(tmp, updated) {
    if (length(tmp) > 1 && !is.na(updated) && updated!= "") {
      return(paste0(tmp[1], updated))
    } else {
      return ("")
    }
  }
  tmp2 <- mapply(sub_function, tmp, updated) %>% unlist()
  return(tmp2[tmp2!=""])
}

# Renamed conditions
conditions_list_parsed <- mapply(merge_and_check_conditions, conditions_list, conditions)

# Convert units to common set
format_units <- function(x) {
  options(scipen=999)
  tmp <- str_split(string = x, paste0(units_sub, collapse = " )|(?<= "), n = 2) %>%
    lapply(gsub, pattern = " $", replacement = "")
  sub_function <- function(x) {
    if (length(x) < 2) {
      return(NULL)
    }
    c(str_split(x[1], " ") %>% unlist(), x[2])
  }
  tmp2 <- lapply(tmp, sub_function)
  sub_function2 <- function(x) {
    if (length(x)<2) {
      return ("")
    }
    if (x[2]=="mm") {
      x[2] = "m"
      x[1] = 0.001 * as.numeric(x[1])
    } else if (x[2]=="um") {
      x[2] = "m"
      x[1] = 0.000001 * as.numeric(x[1])
    } else if (x[2]=="mg") {
      x[2] = "g"
      x[1] = 0.001 * as.numeric(x[1])
    } else if (x[2]=="ul") {
      x[2] = "ml"
      x[1] = 0.001 * as.numeric(x[1])
    } else if (x[2]=="nl") {
      x[2] = "ml"
      x[1] = 0.000001 * as.numeric(x[1])
    } else if (x[2]=="% w\\/w") { # Assuming the solvent is around 1 kg/L
      x[2] = "%"
    } else if (x[2]=="% w\\/v") {
      x[2] = "%"
    } else if (x[2]=="w\\/w") {
      x[2] = "%"
    } else if (x[2]=="w\\/v") {
      x[2] = "%"
    } else if (x[2]=="v\\/v") {
      x[2] = "%"
    } else if (x[2]=="\\/v") {
      x[2] = "%"
    } else if (x[2]=="% \\/v") {
      x[2] = "%"
    } else if (x[2]=="mg\\/ml") {
      x[2] = "%"
    }
    return(x)
  }
  tmp3 <- lapply(tmp2, sub_function2)
  lapply(tmp3, paste0, collapse = " ") %>% unlist() %>% return()
}

conditions_list_parsed <- lapply(conditions_list_parsed, format_units)

restricted_units = c(" %", "m", "g", "ml ")

reformat_col_heads <- function(x) {
  tmp <- str_split(string = x, paste0("(?<=", paste0(restricted_units, collapse = " )|(?<= "), ")"), n = 2) %>%
    lapply(gsub, pattern = " $", replacement = "")
  sub_function <- function(x) {
    if(length(x) < 2) {
      return ("")
    }
    c(str_split(x[1], " ") %>% unlist(), x[2])
  }
  tmp2 <- lapply(tmp, sub_function)
  sub_function2 <- function(x) {
    if (length(x) < 3) {
      return (NULL)
    }
    return (paste0(c(x[3], " (", x[2], ")"), collapse = ""))
  }
  lapply(tmp2, sub_function2) %>% return()
}

# Takes a few seconds to run 
new_cols <- conditions_list_parsed %>% 
  unlist() %>% 
  reformat_col_heads() %>%
  unlist() %>%
  table() %>%
  sort(decreasing = T) %>%
  names()

# Prevent different units from adding columns of the same condition
new_cols <- new_cols[!duplicated(gsub(" \\(.*", "", new_cols))]

# Add more than eventually  keeping since some will be merged
# Get extra bicontinuous variables
new_cols <- sort(c(new_cols[!grepl("^peg.* \\(%\\)|^peg mme[0-9][0-9]* \\(%\\)|^mpeg[0-9][0-9]* \\(%\\)", new_cols)][1:97], new_cols[grepl("^peg[0-9][0-9]* \\(%\\)|^peg mme[0-9][0-9]* \\(%\\)|^mpeg[0-9][0-9]* \\(%\\)", new_cols)]))

metadata[,new_cols] <- 0

# Get units from each column
tmp1 <- strsplit(colnames(metadata)[-(1:6)], " (?!.* )", perl = TRUE)
colname_units <- sapply(tmp1,`[`,2) %>% 
  strsplit(split = "\\(") %>% 
  sapply(`[`,2) %>% 
  strsplit(split = "\\)") %>% 
  sapply(`[`,1)
names(colname_units) <- colnames(metadata)[-(1:6)]

# Add a single condition from a single protein, return 1 if 1 condition was added, 0 otherwise
get_single_condition <- function(x, i) {
  tmp <- str_split(string = x, paste0("(?<=", paste0(restricted_units, collapse = " )|(?<= "), ")"), n = 2) %>%
    lapply(gsub, pattern = " $", replacement = "")
  sub_function <- function(x) {
    if (length(x) < 2) {
      return (0)
    }
    c(str_split(x[1], " ") %>% unlist(), x[2])
  }
  tmp2 <- lapply(tmp, sub_function) %>% unlist()
  if (length(tmp2) < 3) {return (0)}
  sub_function2 <- function(x) {
    if (length(x) < 3) {
      return (0)
    }
    return (paste0(c(x[3], " (", x[2], ")"), collapse = ""))
  }
  tmp2[3] <- sub_function2(tmp2)
  if (tmp2[3] %in% colnames(metadata) && tmp2[2] == colname_units[tmp2[3]]) {
    metadata[i, tmp2[3]] <<- as.numeric(tmp2[1])
    return (1)
  }
  return(0)
}

add_conditions <- function(x, i) {
  num_full <- sum(mapply(get_single_condition, x, rep(i, length(x))), na.rm = T)
}

# About 1000 updates per second
num_full = rep(NA, length(conditions_list_parsed))
pb = txtProgressBar(min = 0, max = length(num_full), initial = 0)
for (i in 1:length(conditions_list_parsed)) {
  if (!is.null(conditions_list_parsed[[i]])) {
    num_full[i] <- add_conditions(conditions_list_parsed[[i]], i)
  }
  setTxtProgressBar(pb,i)
}
close(pb)

# Set entries with no crystallization details to NA, not 0
metadata[is.na(metadata$pdbx_details),new_cols] <- NA_real_

# Get encodings for each column; if the string ends with a number it is bicontinuous + binary
bicont_cols <- unique(gsub("[0-9].*", "", colnames(metadata)[-(1:6)])[grepl("^peg[0-9][0-9]* \\(%\\)|^peg mme[0-9][0-9]* \\(%\\)|^mpeg[0-9][0-9]* \\(%\\)", colnames(metadata)[-(1:6)])])

# Assumes only one PEG condition per sample; this is true in about 97% of cases with any PEG
combine_row <- function(x, col_names) {
  if (sum(is.na(x)) > 0) {
    return(NA_real_)
  }
  if (sum(x!=0) > 0) {
    index = which.max(x)
    return(paste0(x[index], " , ", col_names[index]))
  } else {
    return(0)
  }
}

for (column in bicont_cols) {
  col_names = colnames(metadata)[grepl(paste0("^", column, "[0-9]* \\(%\\)"), colnames(metadata))]
  col_values = gsub(column, "", col_names) %>% gsub(pattern = " \\(.*", replacement = "")
  metadata[,paste(column, gsub(".* ", "", col_names[1]))] <- apply(metadata[,col_names], 1, combine_row, col_values)
  metadata[,col_names[col_names != paste(column, gsub(".* ", "", col_names[1]))]] <- NULL
}

bound <- function(x) {
  tmp <- x[abs(x) - mean(x[x!=0], na.rm = T) < 3 * sd(x[x!=0], na.rm = T)]
  return(abs(x) - mean(tmp[tmp!=0], na.rm = T) < 20 * sd(tmp[tmp!=0], na.rm = T))
}

bound_bicont <- function(x) {
  tmp <- str_split(x, " , ", simplify = T)
  tmp <- matrix(as.numeric(tmp), ncol = ncol(tmp))
  return(bound(tmp[,1]) & bound(tmp[,2]))
}

# Threshold to avoid clearly incorrect values
threshold <- function(x) {
  return(ifelse(rep(is.numeric(x), length(x)), # Check for within 5 SDs if numeric
                ifelse(x==0, 0, 
                       ifelse(bound(x), 
                              x, 
                              NA_real_)), 
                ifelse(grepl(" , ", x),
                       ifelse(unlist(lapply(str_split(x, " , "), function(x) {return(sum(x=="0"))})) > 0 | # Check that both the condition and value are non-zero if bicontinuous
                                !bound_bicont(x),
                              NA_character_,
                              x),
                       x)))
}
metadata[,5:106] <- data.frame(lapply(metadata[,5:106], threshold), check.names = F)

metadata$pdbx_details <- stored_pdbx_details

# Write metadata to file
write.table(metadata, "metadata/metadata_parsed_100.tsv", sep = "\t", row.names = F, quote=T)

mean((num_full==num_conditions_each)[num_conditions_each!=0], na.rm = T)
# 69823 (67.3% of possible) proteins with all their numeric crystallization conditions in the final table

# Write encodings to a new file
encodings <- data.frame(name=colnames(metadata), encoding=c(NA, NA, NA, "cat", "cont", "cont", 
                                                            ifelse(grepl("^peg \\(%\\)|^peg mme \\(%\\)|^mpeg \\(%\\)", 
                                                                         colnames(metadata[7:length(colnames(metadata))])), 
                                                                   "bicontbin", 
                                                                   "contbin")))

write.table(encodings, "metadata/encodings_100.tsv", sep = "\t", row.names = F, quote=T)

end.time = Sys.time()
end.time-start.time

# ~8  minutes

