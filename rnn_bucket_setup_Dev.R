#############################################
#### Symbol design for single output

# lstm cell symbol
lstm.symbol <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout=0, data_masking){
  if (dropout > 0) indata <- mx.symbol.Dropout(data=indata, p=dropout)
  i2h <- mx.symbol.FullyConnected(data=indata,
                                  weight=param$i2h.weight,
                                  bias=param$i2h.bias,
                                  num.hidden=num.hidden * 4,
                                  name=paste0("t", seqidx, ".l", layeridx, ".i2h"))
  if (!is.null(prev.state)){
    h2h <- mx.symbol.FullyConnected(data=prev.state$h,
                                    weight=param$h2h.weight,
                                    bias=param$h2h.bias,
                                    num.hidden=num.hidden * 4,
                                    name=paste0("t", seqidx, ".l", layeridx, ".h2h"))
    gates <- i2h + h2h
  } else gates<- i2h
  
  split.gates <- mx.symbol.split(gates, num.outputs=4, axis=1, squeeze.axis=F,
                                 name=paste0("t", seqidx, ".l", layeridx, ".slice"))
  
  in.gate <- mx.symbol.Activation(split.gates[[1]], act.type="sigmoid")
  in.transform <- mx.symbol.Activation(split.gates[[2]], act.type="tanh")
  forget.gate <- mx.symbol.Activation(split.gates[[3]], act.type="sigmoid")
  out.gate <- mx.symbol.Activation(split.gates[[4]], act.type="sigmoid")
  
  if (!is.null(prev.state)){
    next.c <- (forget.gate * prev.state$c) + (in.gate * in.transform)
  } else next.c <- in.gate * in.transform
  
  next.h <- out.gate * mx.symbol.Activation(next.c, act.type="tanh")
  
  ### Add a mask - using the mask_array approach
  data_mask_expand<- mx.symbol.Reshape(data=data_masking, shape=c(1,-2))
  next.c<- mx.symbol.broadcast_mul(lhs = next.c, rhs=data_mask_expand)
  next.h<- mx.symbol.broadcast_mul(lhs = next.h, rhs=data_mask_expand)
  
  return (list(c=next.c, h=next.h))
}


# unrolled lstm network
rnn.unroll <- function(num.rnn.layer, 
                       seq.len, 
                       input.size,
                       num.hidden, 
                       num.embed, 
                       num.label, 
                       dropout=0.,
                       ignore_label=0,
                       config="one-to-one") {
  
  embed.weight <- mx.symbol.Variable("embed.weight")
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  
  param.cells <- lapply(1:num.rnn.layer, function(i) {
    cell <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                 i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                 h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                 h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
    return (cell)
  })
  
  
  # embeding layer
  label <- mx.symbol.Variable("label")
  data <- mx.symbol.Variable("data")
  data_mask <- mx.symbol.Variable("data_mask")
  data_mask_array <- mx.symbol.Variable("data_mask_array")
  data_mask_array<- mx.symbol.stop_gradient(data_mask_array, name="data_mask_array")
  
  embed <- mx.symbol.Embedding(data=data, input_dim=input.size,
                               weight=embed.weight, output_dim=num.embed, name="embed")
  
  wordvec <- mx.symbol.split(data=embed, axis=1, num.outputs=seq.len, squeeze_axis=T)
  data_mask_split <- mx.symbol.split(data=data_mask_array, axis=1, num.outputs=seq.len, squeeze_axis=T)
  
  last.hidden <- list()
  last.states<- list()
  decode<- list()
  softmax<- list()
  fc<- list()
  
  for (seqidx in 1:seq.len) {
    hidden <- wordvec[[seqidx]]
    
    for (i in 1:num.rnn.layer) {
      
      if (seqidx==1) prev.state<- NULL else prev.state<- last.states[[i]]
      
      next.state <- lstm.symbol(num.hidden = num.hidden, 
                                indata=hidden,
                                prev.state=prev.state,
                                param=param.cells[[i]],
                                seqidx=seqidx, 
                                layeridx=i,
                                dropout=0,
                                data_masking=data_mask_split[[seqidx]])
      hidden <- next.state$h
      if (dropout > 0) hidden <- mx.symbol.Dropout(data=hidden, p=dropout)
      last.states[[i]] <- next.state
    }
    
    # Decoding
    if (config=="one-to-one"){
      last.hidden <- c(last.hidden, hidden)
    }
  }
  
  if (config=="seq-to-one"){
    fc <- mx.symbol.FullyConnected(data=hidden,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label)
    
    loss <- mx.symbol.SoftmaxOutput(data=fc, name="sm", label=label, ignore_label=ignore_label)
    
  } else if (config=="one-to-one"){

    last.hidden_expand = lapply(last.hidden, function(i) mx.symbol.expand_dims(i, axis=1))
    concat <-mx.symbol.Concat(last.hidden_expand, num.args = seq.len, dim = 1)
    reshape = mx.symbol.Reshape(concat, shape=c(num.hidden, -1))
    
    fc <- mx.symbol.FullyConnected(data=reshape,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label)
    
    label <- mx.symbol.reshape(data=label, shape=c(-1))
    loss <- mx.symbol.SoftmaxOutput(data=fc, name="sm", label=label, ignore_label=ignore_label)
    
  }
  
  return(loss)
}

###########################################
#### 
mx.rnn.buckets <- function(train.data, 
                           eval.data=NULL,
                           num.rnn.layer,
                           num.hidden, 
                           num.embed, 
                           num.label,
                           input.size,
                           ctx=list(mx.cpu()),
                           num.round=1, 
                           update.period=1,
                           initializer=mx.init.uniform(0.01),
                           dropout=0,
                           config="one-to-one",
                           kvstore="local",
                           optimizer='sgd',
                           batch.end.callback=NULL,
                           epoch.end.callback=NULL,
                           begin.round=1,
                           end.round=1,
                           metric=mx.metric.rmse,
                           verbose=FALSE) {
  
  # check data and change data into iterator
  #train.data <- check.data(train.data, batch.size, TRUE)
  #eval.data <- check.data(eval.data, batch.size, FALSE)
  
  train.data$init()
  if (!is.null(eval.data)) eval.data$init()
  
  train.data$reset()
  if (!is.null(eval.data)) eval.data$reset()
  
  batch_size<- train.data$batch_size
  
  # get unrolled lstm symbol
  sym_list<- sapply(train.data$bucket_names, function(x) {
    rnn.unroll(num.rnn.layer=num.rnn.layer,
               num.hidden=num.hidden,
               seq.len=as.integer(x),
               input.size=input.size,
               num.embed=num.embed,
               num.label=num.label,
               dropout=dropout, 
               config = config)}, 
    simplify = F, USE.NAMES = T)
  
  init.states.name<- as.character()
  for (i in 1:num.rnn.layer){
    state.c <- paste0("l", i, ".init.c")
    state.h <- paste0("l", i, ".init.h")
    init.states.name<- c(init.states.name, state.c, state.h)
  }
  
  ##############################################################
  # setup lstm model
  symbol <- sym_list[[names(train.data$bucketID())]]
  
  arg.names <- symbol$arguments
  input.shape<- lapply(train.data$value(), dim)
  input.shape<- input.shape[names(input.shape) %in% arg.names]
  
  args<- input.shape
  args$ctx <- mx.cpu()
  args$grad.req <- "write"
  args$symbol <- symbol
  
  infer_shapes<- symbol$infer.shape(input.shape)
  arg.params <- mx.init.create(initializer, infer_shapes$arg.shapes, mx.cpu(), skip.unknown=TRUE)
  aux.params <- mx.init.create(initializer, infer_shapes$aux.shapes, mx.cpu(), skip.unknown=TRUE)

  kvstore <- mxnet:::mx.model.create.kvstore(kvstore, params$arg.params, length(ctx), verbose=verbose)
  
  #####################################################################
  ### Execute training -  rnn.model.R
  model<- mx.model.train.rnn.buckets(sym_list=sym_list,
                                     args=args, 
                                     input.shape=input.shape,
                                     arg.params=arg.params, 
                                     aux.params=aux.params,
                                     optimizer=optimizer,
                                     train.data=train.data, 
                                     batch.size=batch_size,
                                     eval.data=eval.data,
                                     kvstore=kvstore,
                                     verbose=verbose,
                                     begin.round = begin.round,
                                     end.round = end.round,
                                     metric = metric,
                                     ctx=ctx,
                                     batch.end.callback=batch.end.callback,
                                     epoch.end.callback=epoch.end.callback)
  
  return(model)
}


# slice the shape on the highest dimension
mx.model.slice.shape <- function(shape, nsplit) {
  ndim <- length(shape)
  batchsize <- shape[[ndim]]
  step <- as.integer((batchsize + nsplit - 1) / nsplit)
  lapply(0:(nsplit - 1), function(k) {
    begin = min(k * step, batchsize)
    end = min((k + 1) * step, batchsize)
    s <- shape
    s[[ndim]] = end - begin
    return(list(begin=begin, end=end, shape=s))
  })
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

# Internal function to check if name end with suffix
mx.util.str.endswith <- function(name, suffix) {
  slen <- nchar(suffix)
  nlen <- nchar(name)
  if (slen > nlen) return (FALSE)
  nsuf <- substr(name, nlen - slen + 1, nlen)
  return (nsuf == suffix)
}

mx.util.str.startswith <- function(name, prefix) {
  slen <- nchar(prefix)
  nlen <- nchar(name)
  if (slen > nlen) return (FALSE)
  npre <- substr(name, 1, slen)
  return (npre == prefix)
}

# filter out null, keep the names
mx.util.filter.null <- function(lst) {
  lst[!sapply(lst, is.null)]
}



#' Training LSTM Unrolled Model with bucketing
#'
#' @param input_seq integer
#'      The initializing sequence
#' @param input_length integer
#'      The number of initializing elements
#' @param infer_length integer
#'      The number of infered elements
#' @param model The model from which to perform inference. 
#' @param random Logical, Whether to infer based on modeled probabilities (T) or by selecting most likely outcome
#' @param ctx mx.context, optional
#'      The device used to perform training.
#' @param kvstore string, not currently supported
#'      The optimization method.
#' @return an integer vector corresponding to the encoded dictionnary
#'
#' @export
mx.rnn.infer.buckets <- function(infer_iter,
                                 model,
                                 ctx=list(mx.cpu()),
                                 kvstore=NULL){
  
  ### Infer parameters from model
  num.rnn.layer=((length(model$arg.params)-3)/6)
  num.hidden=dim(model$arg.params$l1.init.h)[1]
  input.size=dim(model$arg.params$embed.weight)[2]
  num.embed=dim(model$arg.params$embed.weight)[1]
  num.label=dim(model$arg.params$cls.bias)
  
  ### Initialise the iterator
  infer_iter$init()
  infer_iter$reset()
  batch_size<- infer_iter$batch_size
  
  # get unrolled lstm symbol
  sym_list<- sapply(infer_iter$bucket_names, function(x) {
    rnn.unroll(num.rnn.layer=num.rnn.layer,
               num.hidden=num.hidden,
               seq.len=as.integer(x),
               input.size=input.size,
               num.embed=num.embed,
               num.label=num.label,
               dropout=0)}, 
    simplify = F, USE.NAMES = T)
  
  init.states.name<- as.character()
  for (i in 1:num.rnn.layer){
    state.c <- paste0("l", i, ".init.c")
    state.h <- paste0("l", i, ".init.h")
    init.states.name<- c(init.states.name, state.c, state.h)
  }
  
  ##############################################################
  # set up lstm model
  symbol <- sym_list[[names(infer_iter$bucketID())]]
  
  arg.names <- symbol$arguments
  
  input.shapes <- list()
  infer_shapes<- function(seq.len){
    for (name in arg.names) {
      if (name %in% init.states.name) {
        input.shapes[[name]] <- c(num.hidden, batch_size)
      }
      else if (grepl('data$', name)) {
        if (seq.len == 1) {
          input.shapes[[name]] <- c(batch_size)
        } else {
          input.shapes[[name]] <- c(seq.len, batch_size)
        }
      } 
      else if (grepl('data_mask$', name)) {
        input.shapes[[name]] <- c(batch_size)
      }
      else if (grepl('data_mask_array$', name)) {
        if (seq.len == 1) {
          input.shapes[[name]] <- c(batch_size)
        } else {
          input.shapes[[name]] <- c(seq.len, batch_size)
        }
      }
      else if (grepl('label$', name)) {
        input.shapes[[name]] <- c(batch_size)
      }
    }
    return(input.shapes)
  }
  
  input.shape <- infer_shapes(seq.len = as.integer(names(infer_iter$bucketID())))
  args<- input.shape
  args$ctx <- ctx[[1]]
  args$grad.req <- "write"
  args$symbol <- symbol
  
  #####################################################################
  ### The above preperation is essentially the same as for training
  ### Should consider modulising it
  #####################################################################
  
  #####################################################################
  ### Binding seq to executor and iteratively predict
  #####################################################################
  
  ndevice <- length(ctx)
  
  # create the executors - need to adjust for the init_cand init_h
  sliceinfo <- mx.model.slice.shape(input.shape, ndevice)
  
  train.execs <- lapply(1:ndevice, function(i) {
    do.call(mx.simple.bind, args)
  })
  
  # set the parameters into executors
  for (texec in train.execs) {
    mx.exec.update.arg.arrays(texec, model$arg.params, match.name=TRUE)
    mx.exec.update.aux.arrays(texec, model$aux.params, match.name=TRUE)
  }
  
  # KVStore related stuffs
  params.index <-
    as.integer(mx.util.filter.null(
      lapply(1:length(train.execs[[1]]$ref.grad.arrays), function(k) {
        if (!is.null(train.execs[[1]]$ref.grad.arrays[[k]])) k else NULL
      })))
  if (!is.null(kvstore)) {
    kvstore$init(params.index, train.execs[[1]]$ref.arg.arrays[params.index])
  }
  # Get the input names
  input.names <- mx.model.check.arguments(args$symbol)
  input.names<- c(input.names[1], "data_mask", input.names[2])
  
  ### initialize the predict
  predict<- NULL
  labels<- NULL
  
  while (infer_iter$iter.next()){
    seq_len<- as.integer(names(infer_iter$bucketID()))
    # Get input data slice
    dlist <- infer_iter$value()
    slices <- lapply(1:ndevice, function(i) {
      s <- sliceinfo[[i]]
      ret <- list(data=mxnet:::mx.nd.slice(dlist$data, s$begin, s$end),
                  label=mxnet:::mx.nd.slice(dlist$label, s$begin, s$end))
      return(ret)
    })
    
    ### get the new symbol
    ### Bind the arguments and symbol for the BucketID
    symbol<- sym_list[[names(infer_iter$bucketID())]]
    arg.names<- setdiff(symbol$arguments, input.names)
    
    train.execs <- lapply(1:ndevice, function(i) {
      s <- slices[[i]]
      names(s) <- input.names
      #mx.exec.update.arg.arrays(train.execs[[i]], s, match.name=TRUE)
      mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s[1], train.execs[[i]]$arg.arrays[arg.names], s[2]), aux.arrays = train.execs[[i]]$aux.arrays, ctx=ctx[[i]], grad.req=c("null", rep("write", length(symbol$arguments)-2), "null"))
    })
    
    for (texec in train.execs) {
      mx.exec.forward(texec, is.train=FALSE)
    }
    
    # copy outputs to CPU
    out.preds <- lapply(train.execs, function(texec) {
      mx.nd.copyto(texec$ref.outputs[[length(symbol$outputs)]], mx.cpu())
    })
    
    predict <- rbind(predict, matrix(sapply(1:ndevice, function(i) {
      t(as.matrix(out.preds[[i]]))
    }), nrow=batch_size))
    
    labels <- c(labels, sapply(1:ndevice, function(i) {
      as.numeric(as.array(mx.nd.Reshape(slices[[i]]$label, shape=-1)))
    }))
    
  }
  
  return(list(predict=predict, labels=labels))
}



# Extract model from executors
mx.model.extract.model <- function(symbol, train.execs) {
  reduce.sum <- function(x) Reduce("+", x)
  # Get the parameters
  ndevice <- length(train.execs)
  narg <- length(train.execs[[1]]$ref.arg.arrays)
  arg.params <- lapply(1:narg, function(k) {
    if (is.null(train.execs[[1]]$ref.grad.arrays[[k]])) {
      result <- NULL
    } else {
      result <- reduce.sum(lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.arg.arrays[[k]], mx.cpu())
      })) / ndevice
    }
    return(result)
  })
  names(arg.params) <- names(train.execs[[1]]$ref.arg.arrays)
  arg.params <- mx.util.filter.null(arg.params)
  # Get the auxiliary
  naux <- length(train.execs[[1]]$ref.aux.arrays)
  if (naux != 0) {
    aux.params <- lapply(1:naux, function(k) {
      reduce.sum(lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.aux.arrays[[k]], mx.cpu())
      })) / ndevice
    })
    names(aux.params) <- names(train.execs[[1]]$ref.aux.arrays)
  } else {
    aux.params <- list()
  }
  # Get the model
  model <- list(symbol=symbol, arg.params=arg.params, aux.params=aux.params)
  return(structure(model, class="MXFeedForwardModel"))
}
