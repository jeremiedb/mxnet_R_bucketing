require("readr")
require("dplyr")
require("plotly")
require("stringr")
require("stringi")
require("mxnet")
require("AUC")

source("rnn_bucket_setup.R")
source("rnn_bucket_train.R")
source("iterator_and_metrics.R")

#####################################################
### Setup the parameters for the training
# corpus_bucketed_train<- readRDS(file = "data/corpus_bucketed_train_100_200_300_500.rds")
# corpus_bucketed_test<- readRDS(file = "data/corpus_bucketed_test_100_200_300_500.rds")

corpus_bucketed_train<- readRDS(file = "data/corpus_bucketed_train_100_200_300_500_800_left.rds")
corpus_bucketed_test<- readRDS(file = "data/corpus_bucketed_test_100_200_300_500_800_left.rds")

vocab <- length(corpus_bucketed_test$dic)

ctx<- list(mx.cpu())

batch_size = 64
num.hidden = 48
num.embed = 48
num.lstm.layer = 1
update.period = 1

num.label=2
input.size=vocab
initializer=mx.init.Xavier(rnd_type = "gaussian", factor_type = "in", magnitude = 2)
dropout=0.5
verbose=TRUE
metric<- mx.metric.accuracy
#optimizer<- mx.opt.create("sgd", learning.rate=0.05, momentum=0.8, wd=0.0002, clip_gradient=NULL, rescale.grad=1/batch_size)
optimizer<- mx.opt.create("adadelta", rho=0.92, epsilon=1e-6, wd=0.0002, clip_gradient=NULL, rescale.grad=1/batch_size)
begin.round=1
end.round=1

### Create iterators
X_iter_train<- R_iter(buckets = corpus_bucketed_train$buckets, batch_size = batch_size, data_mask_element = 0, shuffle = T)
X_iter_test<- R_iter(buckets = corpus_bucketed_test$buckets, batch_size = batch_size, data_mask_element = 0, shuffle = T)

train.data<- X_iter_train
eval.data<- X_iter_test
kvstore<- "local"
batch.end.callback<- mx.callback.log.train.metric(period = 50)
epoch.end.callback<- mx.callback.log.train.metric(period = 1)

system.time(model_lstm_sentiment<- mx.rnn.buckets(train.data =  train.data,
                                                  eval.data = eval.data,
                                                  begin.round = begin.round, 
                                                  end.round = end.round, 
                                                  ctx = ctx, 
                                                  metric = metric, 
                                                  optimizer = optimizer, 
                                                  kvstore = kvstore,
                                                  num.lstm.layer=num.lstm.layer,
                                                  num.hidden=num.hidden, 
                                                  num.embed=num.embed, 
                                                  num.label=num.label,
                                                  input.size=input.size,
                                                  update.period=1,
                                                  initializer=initializer,
                                                  dropout=dropout,                                             
                                                  batch.end.callback=batch.end.callback,
                                                  epoch.end.callback=epoch.end.callback))

mx.model.save(model_lstm_sentiment, prefix = "models/model_lstm_sentiment", iteration = 1)

#####################################################
### Inference
ctx<- list(mx.cpu())
model_sentiment<- mx.model.load(prefix = "models/model_lstm_sentiment", iteration = 1)

corpus_bucketed_train<- readRDS(file = "data/corpus_bucketed_train_100_200_300_500_800_left.rds")
corpus_bucketed_test<- readRDS(file = "data/corpus_bucketed_test_100_200_300_500_800_left.rds")


###############################################
### Inference on train
batch_size<- 64
X_iter_train<- R_iter(buckets = corpus_bucketed_train$buckets, batch_size = batch_size, data_mask_element = 0, shuffle = T)

infer_model_on_train <- mx.rnn.infer.buckets(infer_iter = X_iter_train, 
                                             model = model_sentiment,
                                             ctx = ctx,
                                             kvstore=NULL)

dim(infer_model_on_train$predict)
length(infer_model_on_train$labels)

pred_train<- apply(infer_model_on_train$predict, 1, which.max)-1
labels_train<- infer_model_on_train$labels
table(pred_train==labels_train)/length(labels_train)



###############################################
### Inference on test
X_iter_test<- R_iter(buckets = corpus_bucketed_test$buckets, batch_size = batch_size, data_mask_element = 0, shuffle = F)

infer_model_on_test <- mx.rnn.infer.buckets(infer_iter = X_iter_test, 
                                            model = model_sentiment,
                                            ctx = ctx,
                                            kvstore=NULL)

pred_test<- apply(infer_model_on_test$predict, 1, which.max)-1
labels_test<- infer_model_on_test$labels
table(pred_test==labels_test)/length(labels_test)



#########################################################
### Graph visualisation
graph_test<- rnn.unroll(num.lstm.layer=1,
                        num.hidden=20,
                        seq.len=3,
                        input.size=1000,
                        num.embed=100,
                        num.label=2,
                        dropout=0.2)
graph.viz(graph_test)
