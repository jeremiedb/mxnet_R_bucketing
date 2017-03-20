Bucketing with RNN - IMDB demo
================

> This document presents the application of a LSTM model using bucketing based on a custom iterator. More specifically, the exemple concerns a sentiment analysis on [IMDB data](http://ai.stanford.edu/~amaas/data/sentiment/)

General approach for data preperation has been to invest pre-training time in cleaning the data and limit the iterator task to data feeding.

Core training module `rnn_model_bucket.R` reuses as much the existing `model.R` in order to eventually make it a general usage training module, regardless of if the data is bucketed or not. It also provides the ability to run on multi-devices (yet to be tested).

Key approach for the training routine is to first initialise the different symbols that will be used during training (one graph is unrolled for each bucket). This it possible as the bucketing iterator provides info the the possible model configurations.

At training, the `mxnet:::mx.symbol.bind` operator is used to bind the previous model arguments to the current symbol provided by the iterator.

The lstm symbol construction has been slightly revised to correc the deprecated target\_reshape warnings as well as to structure the output in a more intuitive shape (I believe it is now as in the Python shape).

1.  Data preparation
    1.1. Read the corpus
    1.2. Apply a pre-processing that returns a list of ord vectors. The length of the list equals to the sample size. It builds a dictionnary or applies a pre-defined one.
    1.3. Applies a bucketing preperation function that convert the list of of word vectors into arrays on integers (word index) of different lengths according to the specified buckets. 0-padding is applied.

2.  Iterator
    2.1 Iterator returns bucket ID along with the associated data and label.
    2.2 BucketID is a named integer whose value is the i\_th\_ batch from the bucket identified by the name of the BucketID.

### Text pre-processing

Pre-process a corpus composed of a vector of sequences Build a dictionnary and remove too rare words

### Create the bucketed arrays

Inputs are a list of data and labels. Each element of the lists corresponds to a bucket. Within each bucket, a sequence ID in the data should matches the label with the same ID (ie. data and labels follow the same ordering). The names of the list elements indicates the sequence length of the bucket.

A bucketID is returned at initialisation - required for the initial setup the executor. This bucketID is a named integer: its value represents the i\_th\_ batch for the bucket identified by the name of the integer.

A mask element is returned but not used yet. It should be useful in the design of a seq to seq model.

### Example of resulting architecture

``` r
graph_test<- lstm.unroll.sentiment(num.lstm.layer=1,
                                   num.hidden=20,
                                   seq.len=2,
                                   input.size=1000,
                                   num.embed=100,
                                   num.label=2,
                                   dropout=0.2)

graph<- graph.viz(graph_test, shape = c(2,12), graph.width.px = 225, graph.height.px = 450)
```

![](lstm_demo.png)

### To Do

-   Fix the slicing to support multi-device training
-   Initial c and h and masking must be sliced in addition to data and labels
-   Clean out the Iterators
-   Further abstract the recurrent symbolic construction to facilitate addition of other structures (GRU, attention).
    -   Look at new module approach in Python API
-   Validate performance and memory usage (should there be a use of shared modules in binding?)
