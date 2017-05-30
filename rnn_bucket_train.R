# Internal function to do multiple device training on RNN
mx.model.train.rnn.buckets <- function(ctx,
                                       sym_list,
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
                                       batch_size) {
  
  ndevice <- length(ctx)
  if(verbose) cat(paste0("Start training with ", ndevice, " devices\n"))
  
  ###################################
  ## Initialisation
  # Get the input names
  symbol<- sym_list[[names(train.data$bucketID())]]
  
  input.names <- names(input.shape)
  arg.names<- names(arg.params)
  
  # Grad request
  grad_req<- rep("write", length(symbol$arguments))
  grad_null_idx<- match(input.names, symbol$arguments)
  grad_req[grad_null_idx]<- "null"
  
  # Arg array order
  update_names<- c(input.names, arg.names)
  arg_update_idx<- match(symbol$arguments, update_names)
  
  # Initial input shapes - need to be adapted for multi-devices - divide highest dimension by device nb
  s<- sapply(input.shape, function(shape){
    mx.nd.zeros(shape=shape, ctx = mx.cpu())
  })
  
  #####################################################
  ### Initial binding
  train.execs <- lapply(1:ndevice, function(i) {
    mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, arg.params)[arg_update_idx], aux.arrays = aux.params, ctx=ctx[[i]], grad.req=grad_req)
  })

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
  
  
  for (iteration in begin.round:end.round) {
    nbatch <- 0
    if (!is.null(metric)) {
      train.metric <- metric$init()
    }
    train.data$reset()
    
    while (train.data$iter.next()) {
      
      seq_len<- as.integer(names(train.data$bucketID()))
      
      # Get inputs from iterator
      dlist <- train.data$value()
      
      # Slice inputs for multi-devices
      slices <- lapply(dlist[input.names], function(input) {
        mx.nd.SliceChannel(data=input, num_outputs = ndevice, axis = 0, squeeze_axis = F)
      })
      
      ### Get the new symbol
      ### Bind the arguments from previous executor state and symbol for the BucketID
      symbol<- sym_list[[names(train.data$bucketID())]]
      
      train.execs <- lapply(1:ndevice, function(i) {
        if (ndevice>1) s <- lapply(slices, function(x) x[[i]]) else 
          s<- slices
        mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.names])[arg_update_idx], aux.arrays = train.execs[[i]]$aux.arrays, ctx=ctx[[i]], grad.req=grad_req)
      })
      
      for (texec in train.execs) {
        mx.exec.forward(texec, is.train=TRUE)
      }
      
      # copy outputs to CPU
      out.preds <- lapply(train.execs, function(texec) {
        #mx.nd.copyto(texec$ref.outputs[[length(symbol$outputs)]], mx.cpu())
        # keep all outputs
        lapply(texec$ref.outputs, function(texec_out) {
          mx.nd.copyto(texec_out, mx.cpu())
        })
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
        if (ndevice==1) {
          train.metric <- metric$update(label = mx.nd.Reshape(slices$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = train.metric)
          #train.metric <- metric$update(label = mx.nd.Reshape(slices$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = train.metric, seq_len=seq_len, batch_size=batch_size)
          #train.metric <- metric$update(label = mx.nd.Reshape(slices$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = train.metric)
        } else{
          for (i in 1:ndevice) {
            train.metric <- metric$update(label = mx.nd.Reshape(slices[[i]]$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = train.metric)
            #train.metric <- metric$update(label = mx.nd.Reshape(slices[[i]]$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = train.metric, seq_len=seq_len, batch_size=batch_size)
            #train.metric <- metric$update(label = mx.nd.Reshape(slices[[i]]$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = train.metric)
          }
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
        slices <- lapply(dlist[input.names], function(input) {
          mx.nd.SliceChannel(data=input, num_outputs = ndevice, axis = 0, squeeze_axis = F)
        })
        
        symbol = sym_list[[names(eval.data$bucketID())]]

        train.execs <- lapply(1:ndevice, function(i) {
          if (ndevice>1) s <- lapply(slices, function(x) x[[i]]) else 
            s<- slices
          mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.names])[arg_update_idx], aux.arrays = train.execs[[i]]$aux.arrays, ctx=ctx[[i]], grad.req=grad_req)
        })
        
        for (texec in train.execs) {
          mx.exec.forward(texec, is.train=FALSE)
        }
        
        # copy outputs to CPU
        out.preds <- lapply(train.execs, function(texec) {
          #mx.nd.copyto(texec$ref.outputs[[length(symbol$outputs)]], mx.cpu())
          # keep all outputs
          lapply(texec$ref.outputs, function(texec_out) {
            mx.nd.copyto(texec_out, mx.cpu())
          })
        })
        
        if (!is.null(metric)) {
          if (ndevice==1) {
            eval.metric <- metric$update(label = mx.nd.Reshape(slices$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = eval.metric)
            #eval.metric <- metric$update(label = mx.nd.Reshape(slices$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = eval.metric, seq_len=seq_len, batch_size=batch_size)
            #eval.metric <- metric$update(label = mx.nd.Reshape(slices$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = eval.metric)
          } else{
            for (i in 1:ndevice) {
              eval.metric <- metric$update(label = mx.nd.Reshape(slices[[i]]$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = eval.metric)
              #eval.metric <- metric$update(label = mx.nd.Reshape(slices[[i]]$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = eval.metric, seq_len=seq_len, batch_size=batch_size)
              #eval.metric <- metric$update(label = mx.nd.Reshape(slices[[i]]$label, shape=-1), pred = out.preds[[i]][[length(out.preds[[i]])]], state = eval.metric)
            }
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
    model <- mx.model.extract.model(symbol, train.execs)
    
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

