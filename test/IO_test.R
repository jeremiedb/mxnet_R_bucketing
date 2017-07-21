### IO test
require("mxnet")
source("mx_io_bucket_iter.R")

data1 <- matrix(1:10, nrow=2)
data2 <- matrix(2*1:12, nrow=3)

label1 <- 1:5
label2 <- 11:14

buckets <- list("2"=list(data=data1, label=label1),
                "3"=list(data=data2, label=label2))

iter <- mx_io_bucket_iter(buckets = buckets, batch_size = 2, data_mask_element = 0, shuffle = F)

iter$reset()
iter$iter.next()
iter$bucketID()
iter$value()
