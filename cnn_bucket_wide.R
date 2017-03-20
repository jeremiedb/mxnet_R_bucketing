#############################################
#### Symbol design for single output

cnn.symbol <- function(seq.len, 
                       input.size,
                       num.embed, 
                       num_filters,
                       num.label, 
                       dropout=0.,
                       ignore_label=0) {
  
  
  conv_params <- list(embed_weight=mx.symbol.Variable("embed_weight"),
                      conv1.weight = mx.symbol.Variable("conv1_weight"),
                      conv1.bias = mx.symbol.Variable("conv1_bias"),
                      conv2.weight = mx.symbol.Variable("conv2_weight"),
                      conv2.bias = mx.symbol.Variable("conv2_bias"),
                      conv3.weight = mx.symbol.Variable("conv3_weight"),
                      conv3.bias = mx.symbol.Variable("conv3_bias"),
                      fc1.weight = mx.symbol.Variable("fc1_weight"),
                      fc1.bias = mx.symbol.Variable("fc1_bias"),
                      fc_final.weight = mx.symbol.Variable("fc_final.weight"),
                      fc_final.bias = mx.symbol.Variable("fc_final.bias"))
  
  # embeding layer
  label <- mx.symbol.Variable("label")
  data <- mx.symbol.Variable("data")
  data_mask <- mx.symbol.Variable("data_mask")
  data_mask_array <- mx.symbol.Variable("data_mask_array")
  data_mask_array<- mx.symbol.BlockGrad(data_mask_array)
  
  embed <- mx.symbol.Embedding(data=data, weight=conv_params$embed_weight, input_dim=input.size, output_dim=num.embed, name="embed")
  embed_expand <- mx.symbol.expand_dims(data=embed, axis=1, name="embed_expand")

  ### infer shapes test
  #test<- embed_expand$get.internals()
  #test$infer.shape(list(data=c(12,128)))
  
  conv1<- mx.symbol.Convolution(data=embed_expand, weight=conv_params$conv1.weight, bias=conv_params$conv1.bias, kernel=c(num.embed, 1), stride=c(1,1), pad=c(0,0), num.filter=8)
  act1<- mx.symbol.Activation(data=conv1, act.type="relu", name="act1")
  pool1<- mx.symbol.Pooling(data=act1, global.pool=T, pool.type="max", kernel=c(1,1), name="pool1")
  
  conv2<- mx.symbol.Convolution(data=embed_expand, weight=conv_params$conv2.weight, bias=conv_params$conv2.bias, kernel=c(num.embed, 2), stride=c(1,1), pad=c(0,0), num.filter=16)
  act2<- mx.symbol.Activation(data=conv2, act.type="relu", name="act2")
  pool2<- mx.symbol.Pooling(data=act2, global.pool=T, pool.type="max", kernel=c(num.embed,seq.len), name="pool2")
  
  conv3<- mx.symbol.Convolution(data=embed_expand, weight=conv_params$conv3.weight, bias=conv_params$conv3.bias, kernel=c(num.embed, 3), stride=c(1,1), pad=c(0,0), num.filter=32)
  act3<- mx.symbol.Activation(data=conv3, act.type="relu", name="act3")
  pool3<- mx.symbol.Pooling(data=act3, global.pool=T, pool.type="max", kernel=c(num.embed,seq.len), name="pool3")
  
  concat<- mx.symbol.Concat(data=c(pool1, pool2, pool3), num.args=3, name="concat")
  flatten<- mx.symbol.Flatten(data=concat, name="flatten")
  dropout<- mx.symbol.Dropout(data=flatten, p=dropout, name="drop")
  
  fc1<- mx.symbol.FullyConnected(data=dropout, weight=conv_params$fc1.weight, bias=conv_params$fc1.bias, num.hidden=16, name="fc1")
  act_fc<- mx.symbol.Activation(data=fc1, act.type="relu", name="act_fc")
  
  fc_final<- mx.symbol.FullyConnected(data=act_fc, weight=conv_params$fc_final.weight, bias=conv_params$fc_final.bias, num.hidden=2, name="fc_final")
  
  ### Removed the ignore label in softmax
  softmax <- mx.symbol.SoftmaxOutput(data=fc_final, name="sm")
  return(softmax)
  
}


###########################################
#### 
mx.cnn.buckets <- function(train.data, 
                           eval.data=NULL,
                           num.embed, 
                           num_filters,
                           num.label,
                           input.size,
                           ctx=list(mx.cpu()),
                           num.round=10, 
                           update.period=1,
                           initializer=mx.init.uniform(0.01),
                           dropout=0,
                           kvstore="local",
                           optimizer='sgd',
                           batch.end.callback,
                           epoch.end.callback,
                           begin.round=1,
                           end.round=1,
                           metric=mx.metric.rmse) {
  # check data and change data into iterator
  #train.data <- check.data(train.data, batch.size, TRUE)
  #eval.data <- check.data(eval.data, batch.size, FALSE)
  
  train.data$init()
  if (!is.null(eval.data)) eval.data$init()
  
  train.data$reset()
  if (!is.null(eval.data)) eval.data$reset()
  
  batch_size<- train.data$batch_size
  
  
  # conv_params <- list(embed_weight=mx.symbol.Variable("embed_weight"),
  #                     conv1.weight = mx.symbol.Variable("conv1_weight"),
  #                     conv1.bias = mx.symbol.Variable("conv1_bias"),
  #                     conv2.weight = mx.symbol.Variable("conv2_weight"),
  #                     conv2.bias = mx.symbol.Variable("conv2_bias"),
  #                     conv3.weight = mx.symbol.Variable("conv3_weight"),
  #                     conv3.bias = mx.symbol.Variable("conv3_bias"),
  #                     # fc1.weight = mx.symbol.Variable("fc1_weight"),
  #                     # fc1.bias = mx.symbol.Variable("fc1_bias"),
  #                     fc_final.weight = mx.symbol.Variable("fc_final.weight"),
  #                     fc_final.bias = mx.symbol.Variable("fc_final.bias"))
  
  # get unrolled lstm symbol
  sym_list<- sapply(train.data$bucket_names, function(x) {
    cnn.symbol(seq.len=as.integer(x),
               input.size=input.size,
               num.embed=num.embed,
               num_filters=num_filters,
               num.label=num.label,
               #conv_params=conv_params,
               dropout=dropout)}, 
    simplify = F, USE.NAMES = T)

  ##############################################################
  # set up lstm model
  symbol <- sym_list[[names(train.data$bucketID())]]
  
  seq.len<- as.integer(names(train.data$bucketID()))
  input.shape<- list(data=c(seq.len, batch_size))
  
  arg.names <- symbol$arguments
  args<- input.shape
  args$ctx <- ctx[[1]]
  args$grad.req <- "write"
  args$symbol <- symbol
  
  mx.model.init.params.rnn <- function(symbol, input.shape, initializer, ctx) {
    if (!is.mx.symbol(symbol)) stop("symbol need to be MXSymbol")
    slist <- symbol$infer.shape(input.shape)
    if (is.null(slist)) stop("Not enough information to get shapes")
    arg.params <- mx.init.create(initializer, slist$arg.shapes, ctx, skip.unknown=TRUE)
    aux.params <- mx.init.create(initializer, slist$aux.shapes, ctx, skip.unknown=FALSE)
    return(list(arg.params=arg.params, aux.params=aux.params))
  }
  
  params <- mx.model.init.params.rnn(symbol = symbol, input.shape = input.shape, initializer = initializer, ctx = mx.cpu())
  kvstore <- mxnet:::mx.model.create.kvstore(kvstore, params$arg.params, length(ctx), verbose=verbose)
  
  #####################################################################
  ### GO TO rnn.model.R
  #####################################################################
  model<- mx.model.train.rnn(sym_list=sym_list,
                             args=args, 
                             input.shape=input.shape,
                             arg.params=params$arg.params, 
                             aux.params=params$aux.params,
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
mx.lstm.infer.buckets.sentiment <- function(infer_iter,
                                            model,
                                            ctx=list(mx.cpu()),
                                            kvstore=NULL){
  
  ### Infer parameters from model
  num.lstm.layer=((length(model$arg.params)-3)/6)
  num.hidden=dim(model$arg.params$l1.init.h)[1]
  input.size=dim(model$arg.params$embed.weight)[2]
  num.embed=dim(model$arg.params$embed.weight)[1]
  num.label=2
  
  ### Initialise the iterator
  infer_iter$init()
  infer_iter$reset()
  batch_size<- infer_iter$batch_size
  
  
  sym_list<- sapply(train.data$bucket_names, function(x) {
    cnn.symbol(seq.len=as.integer(x),
               input.size=input.size,
               num.embed=num.embed,
               num_filters=num_filters,
               num.label=num.label,
               dropout=dropout)}, 
    simplify = F, USE.NAMES = T)
  
  
  init.states.name<- as.character()
  for (i in 1:num.lstm.layer){
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
    
    #infer_iter$iter.next()
    #infer_iter$value()$label
    #apply(predict, 1, which.max)-1
    #table(apply(predict, 1, which.max)-1 == as.numeric(as.array(infer_iter$value()$label)))
    
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
