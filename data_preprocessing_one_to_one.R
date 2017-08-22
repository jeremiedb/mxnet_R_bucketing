require("readr")
require("dplyr")
require("stringr")
require("stringi")

 # download the IMDB dataset
if (!file.exists("data/char_lstm.zip")) {
  download.file("http://data.mxnet.io/data/char_lstm.zip", "data/char_lstm.zip")
  unzip("data/char_lstm.zip", files = "obama.txt", exdir = "data")
}

# install required packages
list.of.packages <- c("readr", "dplyr", "stringr", "stringi")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[, "Package"])]
if (length(new.packages)) install.packages(new.packages)

make_data <- function(path, seq.len=32, dic=NULL) {
  
  text_vec <- read_file(file = path)
  text_vec <- stri_enc_toascii(str = text_vec)
  text_vec <- str_replace_all(string = text_vec, pattern = "[^[:print:]]", replacement = "")
  text_vec <- strsplit(text_vec, '') %>% unlist
  
  if (is.null(dic)) {
    char_keep <- sort(unique(text_vec))
  } else char_keep <- names(dic)[!dic == 0]
  
  ### Remove terms not part of dictionnary
  text_vec <- text_vec[text_vec %in% char_keep]
  
  ### Build dictionnary
  dic <- 1:length(char_keep)
  names(dic) <- char_keep
  dic <- c(`¤` = 0, dic)
  
  ### reverse dictionnary
  rev_dic <- names(dic)
  names(rev_dic) <- dic

  ### Adjuste by -1 because need a 1-lag for labels
  num.seq <- as.integer(floor((length(text_vec)-1)/seq.len))
  
  features <- dic[text_vec[1:(seq.len*num.seq)]] 
  labels <- dic[text_vec[1:(seq.len*num.seq)+1]]
  
  features_array <- array(features, dim=c(seq.len, num.seq))
  labels_array <- array(labels, dim=c(seq.len, num.seq))
  
  return (list(features_array=features_array, labels_array=labels_array, dic=dic, rev_dic=rev_dic))
}


seq.len <- 100
data_prep <- make_data(path = "data/obama.txt", seq.len=seq.len, dic=NULL)

X <- data_prep$features_array
Y <- data_prep$labels_array
dic <- data_prep$dic
rev_dic <- data_prep$rev_dic
vocab <- length(dic)

shape <- dim(X)
train.val.fraction <- 0.9
size <- shape[2]

X.train.data <- X[, 1:as.integer(size * train.val.fraction)]
X.val.data <- X[, -(1:as.integer(size * train.val.fraction))]

X.train.label <- Y[, 1:as.integer(size * train.val.fraction)]
X.val.label <- Y[, -(1:as.integer(size * train.val.fraction))]

train_buckets <- list("100"=list(data=X.train.data, label=X.train.label))
eval_buckets <- list("100"=list(data=X.val.data, label=X.val.label))

train_buckets <- list(buckets = train_buckets, dic = dic, rev_dic = rev_dic)
eval_buckets <- list(buckets = eval_buckets, dic = dic, rev_dic = rev_dic)

saveRDS(train_buckets, file = "data/train_buckets_one_to_one.rds")
saveRDS(eval_buckets, file = "data/eval_buckets_one_to_one.rds")
