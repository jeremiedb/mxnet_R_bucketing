# LSTM cell symbol
lstm.cell <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0) {
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$i2h.weight, bias = param$i2h.bias, 
                                  num.hidden = num.hidden * 4, name = paste0("t", seqidx, ".l", layeridx, ".i2h"))
  
  if (dropout > 0) 
    i2h <- mx.symbol.Dropout(data = i2h, p = dropout)
  
  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$h2h.weight, 
                                    bias = param$h2h.bias, num.hidden = num.hidden * 4, 
                                    name = paste0("t", seqidx, ".l", layeridx, ".h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }
  
  split.gates <- mx.symbol.split(gates, num.outputs = 4, axis = 1, squeeze.axis = F, 
                                 name = paste0("t", seqidx, ".l", layeridx, ".slice"))
  
  in.gate <- mx.symbol.Activation(split.gates[[1]], act.type = "sigmoid")
  in.transform <- mx.symbol.Activation(split.gates[[2]], act.type = "tanh")
  forget.gate <- mx.symbol.Activation(split.gates[[3]], act.type = "sigmoid")
  out.gate <- mx.symbol.Activation(split.gates[[4]], act.type = "sigmoid")
  
  if (is.null(prev.state)) {
    next.c <- in.gate * in.transform
  } else {
    next.c <- (forget.gate * prev.state$c) + (in.gate * in.transform)
  }
  
  next.h <- out.gate * mx.symbol.Activation(next.c, act.type = "tanh")
  
  return(list(c = next.c, h = next.h))
}

# GRU cell symbol
gru.cell <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0) {
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$gates.i2h.weight, 
                                  bias = param$gates.i2h.bias, num.hidden = num.hidden * 2, 
                                  name = paste0("t", seqidx, ".l", layeridx, ".gates.i2h"))
  
  if (dropout > 0) 
    i2h <- mx.symbol.Dropout(data = i2h, p = dropout)
  
  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$gates.h2h.weight, 
                                    bias = param$gates.h2h.bias, num.hidden = num.hidden * 2, 
                                    name = paste0("t", seqidx, ".l", layeridx, ".gates.h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }
  
  split.gates <- mx.symbol.split(gates, num.outputs = 2, axis = 1, squeeze.axis = F, 
                                 name = paste0("t", seqidx, ".l", layeridx, ".split"))
  
  update.gate <- mx.symbol.Activation(split.gates[[1]], act.type = "sigmoid")
  reset.gate <- mx.symbol.Activation(split.gates[[2]], act.type = "sigmoid")
  
  htrans.i2h <- mx.symbol.FullyConnected(data = indata, weight = param$trans.i2h.weight, 
                                         bias = param$trans.i2h.bias, num.hidden = num.hidden, 
                                         name = paste0("t", seqidx, ".l", layeridx, ".trans.i2h"))
  
  if (is.null(prev.state)) {
    h.after.reset <- reset.gate * 0
  } else {
    h.after.reset <- prev.state$h * reset.gate
  }
  
  htrans.h2h <- mx.symbol.FullyConnected(data = h.after.reset, weight = param$trans.h2h.weight, 
                                         bias = param$trans.h2h.bias, num.hidden = num.hidden, 
                                         name = paste0("t", seqidx, ".l", layeridx, ".trans.h2h"))
  
  h.trans <- htrans.i2h + htrans.h2h
  h.trans.active <- mx.symbol.Activation(h.trans, act.type = "tanh")
  
  if (is.null(prev.state)) {
    next.h <- update.gate * h.trans.active
  } else {
    next.h <- prev.state$h + update.gate * (h.trans.active - prev.state$h)
  }

  return(list(h = next.h))
}

# 
#' unroll representation of RNN running on non CUDA device
#' 
#' @export
rnn.unroll <- function(num.rnn.layer, 
                       seq.len, 
                       input.size,
                       num.embed, 
                       num.hidden,
                       num.label,
                       dropout,
                       ignore_label,
                       init.state=NULL,
                       config,
                       cell.type="lstm",
                       output_last_state=F) {
  
  embed.weight <- mx.symbol.Variable("embed.weight")
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  
  param.cells <- lapply(1:num.rnn.layer, function(i) {
    
    if (cell.type=="lstm"){
      cell <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                   i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                   h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                   h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
    } else if (cell.type=="gru"){
      cell <- list(gates.i2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.i2h.weight")),
                   gates.i2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.i2h.bias")),
                   gates.h2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.h2h.weight")),
                   gates.h2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.h2h.bias")),
                   trans.i2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.i2h.weight")),
                   trans.i2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.i2h.bias")),
                   trans.h2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.h2h.weight")),
                   trans.h2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.h2h.bias")))
    }
    return (cell)
  })
  
  # embeding layer
  label <- mx.symbol.Variable("label")
  data <- mx.symbol.Variable("data")
  
  embed <- mx.symbol.Embedding(data=data, input_dim=input.size,
                               weight=embed.weight, output_dim=num.embed, name="embed")
  
  wordvec <- mx.symbol.split(data=embed, axis=1, num.outputs=seq.len, squeeze_axis=T)

  last.hidden <- list()
  last.states <- list()
  
  for (seqidx in 1:seq.len) {
    hidden <- wordvec[[seqidx]]
    
    for (i in 1:num.rnn.layer) {
      
      if (seqidx==1) prev.state<- init.state[[i]] else prev.state <- last.states[[i]]
      
      if (cell.type=="lstm") {
        cell.symbol <- lstm.cell
      } else if (cell.type=="gru"){
        cell.symbol <- gru.cell
      }
      
      next.state <- cell.symbol(num.hidden = num.hidden, 
                                indata=hidden,
                                prev.state=prev.state,
                                param=param.cells[[i]],
                                seqidx=seqidx, 
                                layeridx=i,
                                dropout=dropout)
      hidden <- next.state$h
      #if (dropout > 0) hidden <- mx.symbol.Dropout(data=hidden, p=dropout)
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
    
    loss <- mx.symbol.SoftmaxOutput(data=fc, name="sm", label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label)
    
  } else if (config=="one-to-one"){
    
    last.hidden_expand = lapply(last.hidden, function(i) mx.symbol.expand_dims(i, axis=1))
    concat <-mx.symbol.concat(last.hidden_expand, num.args = seq.len, dim = 1)
    reshape = mx.symbol.reshape(concat, shape=c(num.hidden, -1))
    
    fc <- mx.symbol.FullyConnected(data=reshape,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label)
    
    label <- mx.symbol.reshape(data=label, shape=c(-1))
    loss <- mx.symbol.SoftmaxOutput(data=fc, name="sm", label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label)
    
  }
  return(loss)
}
