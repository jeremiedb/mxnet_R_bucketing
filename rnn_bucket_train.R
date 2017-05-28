# Internal function to do multiple device training on RNN
mx.model.train.rnn.buckets <- function(ctx,
                                       sym_list,
                                       args,
                                       arg.params, 
                                       aux.params,
                                       input.shape,
                                       begin.round, 
                                       end.round, 
                                       optimizer,
                                       train.data, 
                                       eval.data,
                                       metric,
                                       epoch.end.callback,
                                       batch.end.callback,
                                       kvstore,
                                       verbose=TRUE,
                                       batch.size) {
  
  ndevice <- length(ctx)
  if(verbose) cat(paste0("Start training with ", ndevice, " devices\n"))
  
  # create the executors - need to adjust for the init_cand init_h
  sliceinfo <- mx.model.slice.shape(input.shape, ndevice)
  
  train.execs <- lapply(1:ndevice, function(i) {
    do.call(mx.simple.bind, args)
  })
  # set the parameters into executors
  for (texec in train.execs) {
    mx.exec.update.arg.arrays(texec, arg.params, match.name=TRUE)
    mx.exec.update.aux.arrays(texec, aux.params, match.name=TRUE)
  }

  # KVStore related stuffs
  params.index <-
    as.integer(mx.util.filter.null(
      lapply(1:length(train.execs[[1]]$ref.grad.arrays), function(k) {
        if (!is.null(train.execs[[1]]$ref.grad.arrays[[k]])) k else NULL
      })))
  update.on.kvstore <- FALSE
  if (!is.null(kvstore) && kvstore$update.on.kvstore) {
    update.on.kvstore <- TRUE
    kvstore$set.optimizer(optimizer)
  } else {
    updaters <- lapply(1:ndevice, function(i) {
      mx.opt.get.updater(optimizer, train.execs[[i]]$ref.arg.arrays)
    })
  }
  if (!is.null(kvstore)) {
    kvstore$init(params.index, train.execs[[1]]$ref.arg.arrays[params.index])
  }
  # Get the input names
  input.names <- mx.model.check.arguments(args$symbol)
  input.names<- c(input.names[1], "data_mask_array", input.names[2])
  
  # Grad request
  grad_req<- rep("write", length(args$symbol$arguments))
  grad_null_idx<- match(input.names, args$symbol$arguments)
  grad_req[grad_null_idx]<- "null"
  
  # Arg array order
  sym_arguments<- args$symbol$arguments
  arg.names<- setdiff(sym_arguments, input.names)
  update_names<- c(input.names, arg.names)
  arg_update_idx<- match(sym_arguments, update_names)
  
  for (iteration in begin.round:end.round) {
    nbatch <- 0
    if (!is.null(metric)) {
      train.metric <- metric$init()
    }
    train.data$reset()
    
    while (train.data$iter.next()) {
      seq_len<- as.integer(names(train.data$bucketID()))
      # Get input data slice
      dlist <- train.data$value()
      slices <- lapply(1:ndevice, function(i) {
        s <- sliceinfo[[i]]
        ret <- list(data=mxnet:::mx.nd.slice(dlist$data, s$begin, s$end),
                    data_mask_array=mxnet:::mx.nd.slice(dlist$data_mask_array, s$begin, s$end),
                    label=mxnet:::mx.nd.slice(dlist$label, s$begin, s$end))
        return(ret)
      })
      
      ### Get the new symbol
      ### Bind the arguments from previous executor state and symbol for the BucketID
      symbol = sym_list[[names(train.data$bucketID())]]
      
      train.execs <- lapply(1:ndevice, function(i) {
        s <- slices[[i]]
        names(s) <- input.names
        mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.names])[arg_update_idx], aux.arrays = train.execs[[i]]$aux.arrays, ctx=ctx[[i]], grad.req=grad_req)
      })
      
      for (texec in train.execs) {
        mx.exec.forward(texec, is.train=TRUE)
      }
      
      ## debug
      # dim(train.execs[[1]]$outputs$embed_output)
      # dim(train.execs[[1]]$outputs$slicechannel21_output0)
      # dim(train.execs[[1]]$outputs$slicechannel21_output198)
      # 
      # train.execs[[1]]$arg.arrays$data_mask_array
      # train.execs[[1]]$outputs$slicechannel29_output126
      # train.execs[[1]]$outputs$reshape3703_output
      # 
      # lala<- as.array(train.execs[[1]]$outputs$concat28_output)
      # lala1<- (lala[1,,])
      # summary(lala)
      # dim(lala1)
      # dlist$data_mask
      # lala1
      
      # copy outputs to CPU
      out.preds <- lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.outputs[[length(symbol$outputs)]], mx.cpu())
      })
      
      # backward pass
      for (texec in train.execs) {
        mx.exec.backward(texec)
      }
      if (!is.null(kvstore)) {
        # push the gradient
        kvstore$push(params.index, lapply(train.execs, function(texec) {
          texec$ref.grad.arrays[params.index]
        }), -params.index)
      }
      if (update.on.kvstore) {
        # pull back weight
        kvstore$pull(params.index, lapply(train.execs, function(texec) {
          texec$ref.arg.arrays[params.index]
        }), -params.index)
      } else {
        # pull back gradient sums
        if (!is.null(kvstore)) {
          kvstore$pull(params.index, lapply(train.execs, function(texec) {
            texec$ref.grad.arrays[params.index]
          }), -params.index)
        }
        arg.blocks <- lapply(1:ndevice, function(i) {
          updaters[[i]](train.execs[[i]]$ref.arg.arrays, train.execs[[i]]$ref.grad.arrays)
        })
        for (i in 1:ndevice) {
          mx.exec.update.arg.arrays(train.execs[[i]], arg.blocks[[i]], skip.null=TRUE)
        }
      }
      # Update the evaluation metrics
      if (!is.null(metric)) {
        for (i in 1 : ndevice) {
          #train.metric <- metric$update(label = mx.nd.Reshape(train.execs[[i]]$ref.arg.arrays[["label"]], shape = -1), pred = train.execs[[i]]$ref.outputs[["sm_output"]], state = train.metric)
          train.metric <- metric$update(label = mx.nd.Reshape(slices[[i]]$label, shape=-1), pred = out.preds[[i]], state = train.metric)
          #train.metric <- metric$update(label = mx.nd.Reshape(slices[[i]]$label, shape=-1), pred = out.preds[[i]], state = train.metric, seq_len = seq_len, batch.size=batch.size)
        }
      }
      nbatch <- nbatch + 1
      
      if (!is.null(batch.end.callback)) {
        batch.end.callback(iteration, nbatch, environment())
      }
    }
    
    if (!is.null(metric)) {
      result <- metric$get(train.metric)
      if(verbose) cat(paste0("[", iteration, "] Train-", result$name, "=", result$value, "\n"))
    }
    
    if (!is.null(eval.data)) {
      if (!is.null(metric)) {
        eval.metric <- metric$init()
      }
      eval.data$reset()
      while (eval.data$iter.next()) {
        
        seq_len<- as.integer(names(eval.data$bucketID()))
        # Get input data slice
        dlist <- eval.data$value()
        slices <- lapply(1:ndevice, function(i) {
          s <- sliceinfo[[i]]
          ret <- list(data=mxnet:::mx.nd.slice(dlist$data, s$begin, s$end),
                      data_mask_array=mxnet:::mx.nd.slice(dlist$data_mask_array, s$begin, s$end),
                      label=mxnet:::mx.nd.slice(dlist$label, s$begin, s$end))
          return(ret)
        })
        
        symbol = sym_list[[names(eval.data$bucketID())]]

        train.execs <- lapply(1:ndevice, function(i) {
          s <- slices[[i]]
          names(s) <- input.names
          mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.names])[arg_update_idx], aux.arrays = train.execs[[i]]$aux.arrays, ctx=ctx[[i]], grad.req=grad_req)
        })
        
        for (texec in train.execs) {
          mx.exec.forward(texec, is.train=FALSE)
        }
        out.preds <- lapply(train.execs, function(texec) {
          mx.nd.copyto(texec$ref.outputs[[length(symbol$outputs)]], mx.cpu())
        })
        if (!is.null(metric)) {
          for (i in 1 : ndevice) {
            eval.metric <- metric$update(label = mx.nd.Reshape(slices[[i]]$label, shape=-1), pred = out.preds[[i]], state = eval.metric)
          }
        }
      }
      #eval.data$reset()
      if (!is.null(metric)) {
        result <- metric$get(eval.metric)
        if(verbose) cat(paste0("[", iteration, "] Validation-", result$name, "=", result$value, "\n"))
      }
    } else {
      eval.metric <- NULL
    }
    # get the model out
    model <- mx.model.extract.model(args$symbol, train.execs)
    
    epoch_continue <- TRUE
    if (!is.null(epoch.end.callback)) {
      epoch_continue <- epoch.end.callback(iteration, 0, environment(), verbose = verbose)
    }
    
    if (!epoch_continue) {
      break
    }
  }
  return(model)
}


