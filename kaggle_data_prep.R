require("readr")
require("data.table")
require("dplyr")
require("plotly")
require("stringr")
require("stringi")
require("mxnet")

#options(repos=c("CRAN"="https://cloud.r-project.org/"))
#install.packages("https://github.com/jeremiedb/mxnet_winbin/raw/master/mxnet.zip", repos=NULL)

train_raw <- fread(file = "data/kaggle/data/labeledTrainData.tsv", sep = "\t", data.table = F)
test_raw <- fread(file = "data/kaggle/data/testData.tsv", sep = "\t", data.table = F)


################################################################
### Data structure
###   Each element of the character vector text sample
###   Each element of the character vector is associated with a label

################################################################
##### Pre-process a corpus composed of a vector of sequences
##### Build a dictionnary removing too rare words
text_pre_process <- function(corpus, count_threshold=10, dic=NULL){
  
  raw_vec <- corpus
  raw_vec <- stri_enc_toascii(str = raw_vec)
  
  ### remove non-printable characters
  raw_vec <- str_replace_all(string = raw_vec, pattern = "[^[:print:]]", replacement = "")
  raw_vec <- str_to_lower(string = raw_vec)
  raw_vec <- str_replace_all(string = raw_vec, pattern = "_", replacement = " ")
  raw_vec <- str_replace_all(string = raw_vec, pattern = "\\bbr\\b", replacement = "")
  raw_vec <- str_replace_all(string = raw_vec, pattern = "\\s+", replacement = " ")
  raw_vec <- str_trim(string = raw_vec)
  
  ### Split raw sequence vectors into lists of word vectors (one list element per sequence)
  word_vec_list<- stri_split_boundaries(raw_vec, type="word", skip_word_none=T, skip_word_number=F, simplify = F)
  
  ### Build vocabulary
  if (is.null(dic)){
    word_vec_unlist<- unlist(word_vec_list)
    word_vec_table<- sort(table(word_vec_unlist), decreasing = T)
    word_cutoff<- which.max(word_vec_table<count_threshold)
    word_keep<- names(word_vec_table)[1:(word_cutoff-1)]
    stopwords<- c(letters, "an", "the", "br")
    word_keep<- setdiff(word_keep, stopwords)
  } else word_keep<-names(dic)[!dic==0]
  
  ### Clean the sentences to keep only the curated list of words
  word_vec_list<- lapply(word_vec_list, function(x) x[x %in% word_keep])
  
  #sentence_vec<- stri_split_boundaries(raw_vec, type="sentence", simplify = T)
  word_vec_length<- lapply(word_vec_list, length) %>% unlist()
  
  plot_seq_length<- plot_ly(x=word_vec_length, type="histogram")
  
  ### Build dictionnary
  dic <- 1:length(word_keep)
  names(dic)<- word_keep
  dic<- c("¤"=0, dic)
  
  ### reverse dictionnary
  rev_dic<- names(dic)
  names(rev_dic)<- dic
  
  return(list(word_vec_list=word_vec_list, dic=dic, rev_dic=rev_dic, plot_seq_length=plot_seq_length))
}


################################################################
################################################################
make_bucket_data <- function(word_vec_list, labels, ID=NULL, dic, seq_len=c(100), right_pad=T) {

  # get default ID if none is provided
  if (is.null(ID)) ID <- 1:length(word_vec_list)
  
  ### Trunc sequence to max bucket length
  word_vec_list<- lapply(word_vec_list, head, n=max(seq_len))
  
  word_vec_length <- lapply(word_vec_list, length) %>% unlist()
  bucketID <- cut(word_vec_length, breaks=c(0,seq_len, Inf), include.lowest = T, labels = F)
  #table(bucketID)
  
  ###  Right or Left side Padding
  ### Pad sequences to their bucket length with dictionnary 0-label
  word_vec_list_pad<- lapply(1:length(word_vec_list), function(x){
    length(word_vec_list[[x]])<- seq_len[bucketID[x]]
    word_vec_list[[x]][is.na(word_vec_list[[x]])]<- names(dic[1])
    if (right_pad==F) word_vec_list[[x]] <- rev(word_vec_list[[x]])
    return(word_vec_list[[x]])
  })
  
  ### Assign sequences to buckets and unroll them in order to be reshaped into arrays
  unrolled_arrays<- lapply(1:length(seq_len), function(x) unlist(word_vec_list_pad[bucketID==x]))
  
  ### Assign labels to their buckets
  bucketed_labels <- lapply(1:length(seq_len), function(x) labels[bucketID==x])
  names(bucketed_labels) <- as.character(seq_len)
  
  ### Assign IDs to their buckets
  bucketed_ID <- lapply(1:length(seq_len), function(x) ID[bucketID==x])
  names(bucketed_ID) <- as.character(seq_len)
  
  ### Assign the dictionnary to each bucket terms
  unrolled_arrays_dic <- lapply(1:length(seq_len), function(x) dic[unrolled_arrays[[x]]])
  
  #length(splitted_arrays_dic[[1]])
  ### Reshape into arrays having each sequence into a column
  features_arrays<- lapply(1:length(seq_len), function(x) array(unrolled_arrays_dic[[x]], dim=c(seq_len[x], length(unrolled_arrays_dic[[x]])/seq_len[x])))
  
  features<- lapply(1:length(seq_len), function(x) features_arrays[[x]][1:seq_len[x], ])
  names(features)<- as.character(seq_len)
  
  ### Combine data and labels into buckets
  buckets <- lapply(1:length(seq_len), function(x) c(list(data=features[[x]]), list(label=bucketed_labels[[x]]), list(ID=bucketed_ID[[x]])))
  names(buckets) <- as.character(seq_len)
  
  ### reverse dictionnary
  rev_dic <- names(dic)
  names(rev_dic) <- dic
  
  ### ID sequence
  ID = unlist(bucketed_ID)
  
  return (list(buckets=buckets, dic=dic, rev_dic=rev_dic, ID=ID))
}


corpus_preprocessed_train <- text_pre_process(corpus = train_raw$review, count_threshold = 10, dic=NULL)
corpus_preprocessed_train$plot_seq_length
length(corpus_preprocessed_train$dic)


corpus_preprocessed_test <- text_pre_process(corpus = test_raw$review, dic=corpus_preprocessed_train$dic)
corpus_preprocessed_test$plot_seq_length



### train tot
corpus_bucketed_train_tot <- make_bucket_data(word_vec_list = corpus_preprocessed_train$word_vec_list, 
                                              labels = train_raw$sentiment, 
                                              ID = train_raw$id,
                                              dic = corpus_preprocessed_train$dic,
                                              seq_len = c(60, 100, 160, 240, 400, 600), 
                                              right_pad = F)

(unlist(lapply(corpus_bucketed_train_tot$buckets, function(x) length(x[[2]]))))


### train
set.seed(44)
train_id <- sample(length(corpus_preprocessed_train$word_vec_list), size = 20000)

corpus_bucketed_train <- make_bucket_data(word_vec_list = corpus_preprocessed_train$word_vec_list[train_id], 
                                          labels = train_raw$sentiment[train_id], 
                                          ID = train_raw$id[train_id],
                                          dic = corpus_preprocessed_train$dic,
                                          seq_len = c(60, 100, 160, 240, 400, 600), 
                                          right_pad = F)

sum(unlist(lapply(corpus_bucketed_train$buckets, function(x) length(x[[2]]))))

### train
corpus_bucketed_eval <- make_bucket_data(word_vec_list = corpus_preprocessed_train$word_vec_list[-train_id], 
                                          labels = train_raw$sentiment[-train_id], 
                                          ID = train_raw$id[-train_id],
                                          dic = corpus_preprocessed_train$dic,
                                          seq_len = c(60, 100, 160, 240, 400, 600), 
                                          right_pad = F)

sum(unlist(lapply(corpus_bucketed_eval$buckets, function(x) length(x[[2]]))))


corpus_bucketed_test <- make_bucket_data(word_vec_list = corpus_preprocessed_test$word_vec_list, 
                                         labels = rep(0, nrow(test_raw)), 
                                         ID = test_raw$id,
                                         dic = corpus_preprocessed_test$dic,
                                         seq_len = c(60, 100, 160, 240, 400, 600), 
                                         right_pad = F)

sum(unlist(lapply(corpus_bucketed_test$buckets, function(x) length(x[[2]]))))


saveRDS(corpus_bucketed_train_tot, file = "data/kaggle_corpus_bucketed_train_tot_left.rds")
saveRDS(corpus_bucketed_train, file = "data/kaggle_corpus_bucketed_train_left.rds")
saveRDS(corpus_bucketed_eval, file = "data/kaggle_corpus_bucketed_eval_left.rds")
saveRDS(corpus_bucketed_test, file = "data/kaggle_corpus_bucketed_test_left.rds")
