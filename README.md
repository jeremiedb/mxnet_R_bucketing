MXNet R-package bucketing
================

> Implementation of bucketing within RNN and CNN models with MXNet R-package. Examples based on sentiment analysis on the [IMDB data](http://ai.stanford.edu/~amaas/data/sentiment/)

Look at the the [github pages](https://jeremiedb.github.io/mxnet_R_bucketing/) for detailed examples.

General approach for data preparation was to invest pre-training time in cleaning the data and limit the iterator task to data feeding.

Demo can be run by going through the following scripts:
- data\_import.R
- data\_prep.R
- LSTM\_train.R / CNN\_train.R

Core training module `rnn_bucket_train.R` reuses as much the existing `model.R` in order to ultimately make it a general usage training module, regardless if the data is bucketed or not. It should provide support for multi-devices training (some work on slicing needed before it is functional).

Key approach for the training routine is to first initialize the different symbols that will be used during training (one graph is unrolled for each bucket). This it possible as the bucketing iterator provides info the the possible model configurations.

At training, the `mxnet:::mx.symbol.bind` operator is used to bind the previous model arguments to the current symbol provided by the iterator. Shared module options seems to be missing in R's symbol\_bind operator, to be validated if poses issues on memory efficiency.

The lstm symbol construction has been slightly revised to correct the deprecated target\_reshape warnings and to structure the output in a coherent shape when inferring with batch.

1.  Data preparation
    1.  Read the corpus
    2.  Apply a pre-processing that returns a list of vectors. The length of the list equals to the sample size. It builds a dictionary or applies a pre-defined one.
    3.  Applies a bucketing preparation function that convert the list of of word vectors into arrays on integers (word index) of different lengths according to the specified buckets. 0-padding is applied.

2.  Iterator
    1.  Iterator returns bucket ID along with the associated data and label.
    2.  BucketID is a named integer whose value is the i\_th\_ batch from the bucket identified by the name of the BucketID.

### Text pre-processing

Minimal text pre-processing has been performed: lower-case, removal of sparse words. Results in a list of word vectors.

### Bucketed arrays iterator

Inputs are a list of data and labels. Each element of the lists corresponds to a bucket. Within each bucket, a sequence ID in the data should matches the label with the same ID (ie. data and labels follow the same ordering). The names of the list elements indicates the sequence length of the bucket.

A bucketID is returned at initialization - required for the initial setup the executor. This bucketID is a named integer: its value represents the i\_th\_ batch for the bucket identified by the name of the integer.

A data mask array is also returned. It's an indicator matrix (built of 0 and 1) of same dimension than the data. Used to *zeroised* the *c* and *h* output when the sequence input value is 0 (0 being the padded value).

### Example of resulting architecture

``` r
graph_test<- rnn.unroll(num.lstm.layer=1,
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
    -   Initial *c* and *h* and masking must be sliced in addition to *data* and *labels*
-   Clean out the Iterators
-   Further abstract the recurrent symbolic construction to facilitate addition of other structures (GRU, attention)
    -   Look at new module approach in Python API
-   Validate performance and memory usage (should there be a use of shared modules in binding?)
