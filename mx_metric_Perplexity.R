#############################################
### Perplexity metric for language model

mx_metric_Perplexity <- list(
  init = function() {
    c(0, 0)
  },
  update = function(label, pred, state, seq_len, batch.size) {
    m <- -sum(log(pmax(1e-15, as.array(mx.nd.choose.element.0index(pred, label)))))
    state <- c(state[[1]] + seq_len*batch.size, state[[2]] + m)
    return(state)
  },
  get = function(state) {
    list(name="Perplexity", value=exp(state[[2]]/state[[1]]))
  }
)
class(mx_metric_Perplexity) <- "mx.metric"
