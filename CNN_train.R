require("readr")
require("dplyr")
require("plotly")
require("stringr")
require("stringi")
require("mxnet")
require("AUC")

source("cnn_bucket_sentiment.R")
source("cnn_model_bucket.R")
source("iterator_and_metrics.R")

#####################################################
### Setup the parameters for the training
# corpus_bucketed_train<- readRDS(file = "data/corpus_bucketed_train_100_200_300_500.rds")
# corpus_bucketed_test<- readRDS(file = "data/corpus_bucketed_test_100_200_300_500.rds")

corpus_bucketed_train<- readRDS(file = "data/corpus_bucketed_train_100_200_300_500_800_left.rds")
corpus_bucketed_test<- readRDS(file = "data/corpus_bucketed_test_100_200_300_500_800_left.rds")

vocab <- length(corpus_bucketed_test$dic)

ctx<- list(mx.cpu())

batch_size = 32
num.embed=32
num_filters=16
update.period = 1

num.label=2
input.size=vocab
initializer=mx.init.Xavier(rnd_type = "gaussian", factor_type = "in", magnitude = 2)
dropout=0.25
verbose=TRUE
metric<- mx.metric.accuracy
optimizer<- mx.opt.create("sgd", learning.rate=0.05, momentum=0.8, wd=0.001, clip_gradient=NULL, rescale.grad=1/batch_size)
optimizer<- mx.opt.create("adadelta", rho=0.92, epsilon=1e-6, wd=0.001, clip_gradient=NULL, rescale.grad=1/batch_size)
begin.round=1
end.round=8

### Create iterators
X_iter_train<- R_iter(buckets = corpus_bucketed_train$buckets, batch_size = batch_size, data_mask_element = 0, shuffle = T)
X_iter_test<- R_iter(buckets = corpus_bucketed_test$buckets, batch_size = batch_size, data_mask_element = 0, shuffle = T)

train.data<- X_iter_train
eval.data<- X_iter_test
kvstore<- "local"
batch.end.callback<- mx.callback.log.train.metric(period = 50)
epoch.end.callback<- mx.callback.log.train.metric(period = 1)

# X_iter_train$init()
# X_iter_train$reset()
# X_iter_train$bucket_names
# X_iter_train$iter.next()
# (label1<- as.array(X_iter_train$value()$label))
# data1<- as.array(X_iter_train$value()$data)
# paste0(rev_dic[as.character(data1[,1])], collapse = " ")

system.time(model_sentiment_cnn<- mx.cnn.buckets(train.data =  train.data,
                                                 eval.data = eval.data,
                                                 begin.round = begin.round, 
                                                 end.round = end.round, 
                                                 ctx = ctx, 
                                                 metric = metric, 
                                                 optimizer = optimizer, 
                                                 kvstore = kvstore,
                                                 num.embed=num.embed, 
                                                 num_filters = num_filters,
                                                 num.label=num.label,
                                                 input.size=input.size,
                                                 update.period=1,
                                                 initializer=initializer,
                                                 dropout=dropout,                                             
                                                 batch.end.callback=batch.end.callback,
                                                 epoch.end.callback=epoch.end.callback))

mx.model.save(model_sentiment_mask, prefix = "models/model_sentiment_cnn_mask", iteration = 50)



###############################################
### Deep CNN on words
###############################################

system.time(model_sentiment_cnn_deep<- mx.cnn.buckets.deep(train.data =  train.data,
                                                           eval.data = eval.data,
                                                           begin.round = begin.round, 
                                                           end.round = end.round, 
                                                           ctx = ctx, 
                                                           metric = metric, 
                                                           optimizer = optimizer, 
                                                           kvstore = kvstore,
                                                           num.embed=num.embed, 
                                                           num_filters = num_filters,
                                                           num.label=num.label,
                                                           input.size=input.size,
                                                           update.period=1,
                                                           initializer=initializer,
                                                           dropout=dropout,                                             
                                                           batch.end.callback=batch.end.callback,
                                                           epoch.end.callback=epoch.end.callback))

mx.model.save(model_sentiment_cnn_deep, prefix = "models/model_sentiment_cnn_deep", iteration = 8)


#####################################################
### Inference
ctx<- list(mx.cpu())
model_sentiment<- mx.model.load(prefix = "models/model_sentiment_cnn_deep", iteration = 8)

corpus_bucketed_train<- readRDS(file = "data/corpus_bucketed_train_100_200_300_500_800_left.rds")
corpus_bucketed_test<- readRDS(file = "data/corpus_bucketed_test_100_200_300_500_800_left.rds")


###############################################
### Inference on train
batch_size<- 32
X_iter_train<- R_iter(buckets = corpus_bucketed_train$buckets, batch_size = batch_size, data_mask_element = 0, shuffle = T)

infer_model_on_train <- mx.cnn.infer.buckets.sentiment(infer_iter = X_iter_train, 
                                                       model = model_sentiment,
                                                       ctx = ctx,
                                                       kvstore="local")

dim(infer_model_on_train$predict)
length(infer_model_on_train$labels)

pred_train<- apply(infer_model_on_train$predict, 1, which.max)-1
labels_train<- infer_model_on_train$labels
table(pred_train==labels_train)/length(labels_train)

roc_train<- roc(predictions = infer_model_on_train$predict[,2], labels = factor(labels_train))
auc(roc_train)


###############################################
### Inference on test
X_iter_test<- R_iter(buckets = corpus_bucketed_test$buckets, batch_size = batch_size, data_mask_element = 0, shuffle = F)

infer_model_on_test <- mx.cnn.infer.buckets.sentiment(infer_iter = X_iter_test, 
                                                      model = model_sentiment,
                                                      ctx = ctx,
                                                      kvstore="local")

pred_test<- apply(infer_model_on_test$predict, 1, which.max)-1
labels_test<- infer_model_on_test$labels
table(pred_test==labels_test)/length(labels_test)

roc_test<- roc(predictions = infer_model_on_test$predict[,2], labels = factor(labels_test))
auc(roc_test)

seq.len=12
input.size=1000
num.embed=64
num_filters=16
num.label=2
dropout=0.25

cnn_graph_test<- cnn.symbol(seq.len=seq.len,
                            input.size=input.size,
                            num.embed=num.embed,
                            num_filters = num_filters,
                            num.label=num.label,
                            dropout=dropout)
graph.viz(cnn_graph_test, shape=c(25,64))

cnn_graph_test<- cnn.symbol.deep(seq.len=seq.len,
                                 input.size=input.size,
                                 num.embed=num.embed,
                                 num_filters = num_filters,
                                 num.label=num.label,
                                 dropout=dropout)
graph.viz(cnn_graph_test, shape=c(100,128), type = "vis", direction = "UD")
