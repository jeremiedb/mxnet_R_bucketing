library(mxnet)

is.MXSymbol <- function(x) {
  inherits(x, "Rcpp_MXSymbol")
}

mx.model.init.params <- function(symbol, input.shape, output.shape, initializer, ctx) {
  if (!is.MXSymbol(symbol)) stop("symbol need to be MXSymbol")
  
  arg_lst <- list(symbol = symbol)
  arg_lst <- append(arg_lst, input.shape)
  arg_lst <- append(arg_lst, output.shape)
  
  slist <- do.call(mx.symbol.infer.shape, arg_lst)
  if (is.null(slist)) stop("Not enough information to get shapes")
  arg.params <- mx.init.create(initializer, slist$arg.shapes, ctx, skip.unknown=TRUE)
  aux.params <- mx.init.create(initializer, slist$aux.shapes, ctx, skip.unknown=FALSE)
  return(list(arg.params=arg.params, aux.params=aux.params))
}


########################################### mx.rnn.buckets
mx.rnn.buckets <- function(train.data, eval.data = NULL, num.rnn.layer, num.hidden, 
                           num.embed, num.label, input.size, ctx = NULL, num.round = 1, initializer = mx.init.uniform(0.01), 
                           dropout = 0, config = "one-to-one", optimizer = "sgd", batch.end.callback = NULL, 
                           epoch.end.callback = NULL, begin.round = 1, metric = mx.metric.rmse, cell.type = "lstm", 
                           kvstore = "local", verbose = FALSE, cudnn = FALSE) {
  
  if (!train.data$iter.next()) {
    train.data$reset()
    if (!train.data$iter.next()) 
      stop("Empty train.data")
  }
  
  if (!is.null(eval.data)) {
    if (!eval.data$iter.next()) {
      eval.data$reset()
      if (!eval.data$iter.next()) 
        stop("Empty eval.data")
    }
  }
  
  if (is.null(ctx)) 
    ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) {
    ctx <- list(ctx)
  }
  if (!is.list(ctx)) 
    stop("ctx must be mx.context or list of mx.context")
  if (is.character(optimizer)) {
    if (is.numeric(input.shape)) {
      ndim <- length(input.shape)
      batchsize <- input.shape[[ndim]]
    } else {
      ndim <- length(input.shape[[1]])
      batchsize <- input.shape[[1]][[ndim]]
    }
    optimizer <- mx.opt.create(optimizer, rescale.grad = (1/batchsize), ...)
  }
  
  if (cudnn) unroll <- rnn.unroll.cudnn else 
    unroll <- rnn.unroll
  
  # get unrolled lstm symbol
  sym_list <- sapply(train.data$bucket.names, function(x) {
    unroll(num.rnn.layer = num.rnn.layer, num.hidden = num.hidden, seq.len = as.integer(x), 
           input.size = input.size, num.embed = num.embed, num.label = num.label, 
           dropout = dropout, cell.type = cell.type, config = config)
  }, simplify = F, USE.NAMES = T)
  
  # setup lstm model
  symbol <- sym_list[[names(train.data$bucketID)]]
  
  arg.names <- symbol$arguments
  input.names <- if (cudnn) c("data", "seq.mask") else c("data", "data.mask.array")
  input.shape <- sapply(input.names, function(n) {
    dim(train.data$value()[[n]])
  }, simplify = FALSE)
  output.names <- "label"
  output.shape <- sapply(output.names, function(n) {
    dim(train.data$value()[[n]])
  }, simplify = FALSE)
  
  params <- mx.model.init.params(symbol, input.shape, output.shape, initializer, 
                                 mx.cpu())
  
  kvstore <- mxnet:::mx.model.create.kvstore(kvstore, params$arg.params, length(ctx), 
                                             verbose = verbose)
  
  ### Execute training - rnn.model.R
  model <- mx.model.train.rnn.buckets(sym_list = sym_list, input.shape = input.shape, 
                                      output.shape = output.shape, arg.params = params$arg.params, aux.params = params$aux.params, 
                                      optimizer = optimizer, train.data = train.data, eval.data = eval.data, verbose = verbose, 
                                      begin.round = begin.round, end.round = num.round, metric = metric, ctx = ctx, 
                                      batch.end.callback = batch.end.callback, epoch.end.callback = epoch.end.callback, 
                                      kvstore = kvstore)
  
  return(model)
}


# get the argument name of data and label
mx.model.check.arguments <- function(symbol) {
  data <- NULL
  label <- NULL
  for (nm in arguments(symbol)) {
    if (mx.util.str.endswith(nm, "data")) {
      if (!is.null(data)) {
        stop("Multiple fields contains suffix data")
      } else {
        data <- nm
      }
    }
    if (mx.util.str.endswith(nm, "label")) {
      if (!is.null(label)) {
        stop("Multiple fields contains suffix label")
      } else {
        label <- nm
      }
    }
  }
  return(c(data, label))
}
