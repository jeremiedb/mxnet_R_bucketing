require("readr")
require("dplyr")
require("plotly")
require("stringr")
require("stringi")
require("mxnet")

#options(repos=c("CRAN"="https://cloud.r-project.org/"))
#install.packages("lubridate")
#install.packages("https://github.com/jeremiedb/mxnet_winbin/raw/master/mxnet.zip", repos=NULL)

negative_train_list<- list.files("data/train/neg/", full.names = T)
positive_train_list<- list.files("data/train/pos/", full.names = T)

negative_test_list<- list.files("data/test/neg/", full.names = T)
positive_test_list<- list.files("data/test/pos/", full.names = T)


file_import<- function(file_list){
  import<- sapply(file_list, read_file)
  return(import)
}

negative_train_raw<- file_import(negative_train_list)
positive_train_raw<- file_import(positive_train_list)

negative_test_raw<- file_import(negative_test_list)
positive_test_raw<- file_import(positive_test_list)

train_raw<- c(negative_train_raw, positive_train_raw)
test_raw<- c(negative_test_raw, positive_test_raw)

saveRDS(train_raw, file = "data/train_raw.rds")
saveRDS(test_raw, file = "data/test_raw.rds")