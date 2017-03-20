require("readr")
require("dplyr")
require("plotly")
require("stringr")
require("stringi")
require("mxnet")

source("lstm_bucket.R")
source("rnn_model_bucket.R")
source("iterator_and_metrics.R")

#####################################################
### Setup the parameters for the training
# corpus_bucketed_train<- readRDS(file = "data/corpus_bucketed_train_100_200_300_500.rds")
# corpus_bucketed_test<- readRDS(file = "data/corpus_bucketed_test_100_200_300_500.rds")

corpus_bucketed_train<- readRDS(file = "data/corpus_bucketed_train_100_200_300_500_left.rds")
corpus_bucketed_test<- readRDS(file = "data/corpus_bucketed_test_100_200_300_500_left.rds")

vocab <- length(corpus_bucketed_test$dic)

ctx<- list(mx.cpu())

batch_size = 64
num.hidden = 32
num.embed = 32
num.lstm.layer = 2
update.period = 1

num.label=2
input.size=vocab
initializer=mx.init.Xavier(rnd_type = "gaussian", factor_type = "in", magnitude = 2)
dropout=0.5
verbose=TRUE
metric<- mx.metric.accuracy
optimizer<- mx.opt.create("sgd", learning.rate=0.05, momentum=0.8, wd=0.00001, clip_gradient=NULL, rescale.grad=1/batch_size)
optimizer<- mx.opt.create("adadelta", rho=0.92, epsilon=1e-6, wd=0.0001, clip_gradient=NULL, rescale.grad=1/batch_size)
begin.round=1
end.round=20

### Create iterators
X_iter_train<- R_iter(buckets = corpus_bucketed_train$buckets, batch_size = batch_size, data_mask_element = 0, shuffle = T)
X_iter_test<- R_iter(buckets = corpus_bucketed_test$buckets, batch_size = batch_size, data_mask_element = 0, shuffle = T)

train.data<- X_iter_train
eval.data<- X_iter_test
kvstore<- "local"
batch.end.callback<- mx.callback.log.train.metric(period = 50)
epoch.end.callback<- mx.callback.log.train.metric(period = 1)

system.time(model_lstm_sentiment_mask<- mx.lstm.buckets.sentiment(train.data =  train.data,
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

mx.model.save(model_sentiment_mask, prefix = "models/model_lstm_sentiment_mask", iteration = 20)


#####################################################
### Inference
ctx<- list(mx.cpu())
model_sentiment<- mx.model.load(prefix = "models/model_sentiment_mask", iteration = 20)

corpus_bucketed_train<- readRDS(file = "data/corpus_bucketed_train_100_200_300_500_left.rds")
corpus_bucketed_test<- readRDS(file = "data/corpus_bucketed_test_100_200_300_500_left.rds")


###############################################
### Inference on train
batch_size<- 128
X_iter_train<- R_iter(buckets = corpus_bucketed_train$buckets, batch_size = batch_size, data_mask_element = 0, shuffle = T)

infer_model_on_train <- mx.lstm.infer.buckets.sentiment(infer_iter = X_iter_train, 
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

infer_model_on_test <- mx.lstm.infer.buckets.sentiment(infer_iter = X_iter_test, 
                                                       model = model_sentiment,
                                                       ctx = ctx,
                                                       kvstore=NULL)

pred_test<- apply(infer_model_on_test$predict, 1, which.max)-1
labels_test<- infer_model_on_test$labels
table(pred_test==labels_test)/length(labels_test)




graph_test<- lstm.unroll.sentiment(num.lstm.layer=2,
                                   num.hidden=20,
                                   seq.len=2,
                                   input.size=1000,
                                   num.embed=100,
                                   num.label=2,
                                   dropout=0.2)
graph.viz(graph_test)
