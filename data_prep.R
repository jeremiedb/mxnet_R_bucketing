require("readr")
require("dplyr")
require("plotly")
require("stringr")
require("stringi")
require("mxnet")

#options(repos=c("CRAN"="https://cloud.r-project.org/"))
#install.packages("lubridate")
#install.packages("https://github.com/jeremiedb/mxnet_winbin/raw/master/mxnet.zip", repos=NULL)

train_raw<- readRDS(file = "data/train_raw.rds")
test_raw<- readRDS(file = "data/test_raw.rds")

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
make_bucket_data <- function(word_vec_list, labels, dic, seq_len=c(225), right_pad=T) {
  
  ### Trunc sequence to max bucket length
  word_vec_list<- lapply(word_vec_list, head, n=max(seq_len))
  
  word_vec_length<- lapply(word_vec_list, length) %>% unlist()
  bucketID<- cut(word_vec_length, breaks=c(0,seq_len, Inf), include.lowest = T, labels = F)
  #table(bucketID)
  
  
  ### Right Padding
  ### Pad sequences to their bucket length with dictionnary 0-label
  # word_vec_list_pad<- lapply(1:length(word_vec_list), function(x){
  #   length(word_vec_list[[x]])<- seq_len[bucketID[x]]
  #   word_vec_list[[x]][is.na(word_vec_list[[x]])]<- names(dic[1])
  #   return(word_vec_list[[x]])
  # })
  
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
  bucketed_labels<- lapply(1:length(seq_len), function(x) labels[bucketID==x])
  names(bucketed_labels)<- as.character(seq_len)
  
  ### Assign the dictionnary to each bucket terms
  unrolled_arrays_dic<- lapply(1:length(seq_len), function(x) dic[unrolled_arrays[[x]]])
  
  #length(splitted_arrays_dic[[1]])
  ### Reshape into arrays having each sequence into a column
  features_arrays<- lapply(1:length(seq_len), function(x) array(unrolled_arrays_dic[[x]], dim=c(seq_len[x], length(unrolled_arrays_dic[[x]])/seq_len[x])))
  
  features<- lapply(1:length(seq_len), function(x) features_arrays[[x]][1:seq_len[x], ])
  names(features)<- as.character(seq_len)
  
  ### Combine data and labels into buckets
  buckets<- lapply(1:length(seq_len), function(x) c(list(data=features[[x]]), list(label=bucketed_labels[[x]])))
  names(buckets)<- as.character(seq_len)
  
  ### reverse dictionnary
  rev_dic<- names(dic)
  names(rev_dic)<- dic
  
  return (list(buckets=buckets, dic=dic, rev_dic=rev_dic))
}


corpus_preprocessed_train<- text_pre_process(corpus = train_raw, count_threshold = 10, dic=NULL)
corpus_preprocessed_train$plot_seq_length
length(corpus_preprocessed_train$dic)

corpus_preprocessed_test<- text_pre_process(corpus = test_raw, dic=corpus_preprocessed_train$dic)
corpus_preprocessed_test$plot_seq_length

saveRDS(corpus_preprocessed_train, file = "data/corpus_preprocessed_train_10.rds")
saveRDS(corpus_preprocessed_test, file = "data/corpus_preprocessed_test_10.rds")

corpus_preprocessed_train<- readRDS(file = "data/corpus_preprocessed_train_10.rds")
corpus_preprocessed_test<- readRDS(file = "data/corpus_preprocessed_test_10.rds")


corpus_bucketed_train<- make_bucket_data(word_vec_list = corpus_preprocessed_train$word_vec_list, 
                                         labels = rep(0:1, each=12500), 
                                         dic = corpus_preprocessed_train$dic,
                                         seq_len = c(100, 200, 300, 500, 800), 
                                         right_pad = F)

lapply(corpus_bucketed_train$buckets, function(x) length(x[[2]]))


corpus_bucketed_test<- make_bucket_data(word_vec_list = corpus_preprocessed_test$word_vec_list, 
                                        labels = rep(0:1, each=12500), 
                                        dic = corpus_preprocessed_test$dic,
                                        seq_len = c(100, 200, 300, 500, 800), 
                                        right_pad = F)

lapply(corpus_bucketed_test$buckets, function(x) length(x[[2]]))


saveRDS(corpus_bucketed_train, file = "data/corpus_bucketed_train_100_200_300_500_800_left.rds")
saveRDS(corpus_bucketed_test, file = "data/corpus_bucketed_test_100_200_300_500_800_left.rds")

