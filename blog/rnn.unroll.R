library(mxnet)

source("lstm.cell.R")
source("gru.cell.R")

# unrolled RNN network
rnn.unroll <- function(num.rnn.layer, seq.len, input.size, num.embed, num.hidden, 
                       num.label, dropout = 0, ignore_label = 0, init.state = NULL, config, cell.type = "lstm", 
                       output_last_state = F) {
  embed.weight <- mx.symbol.Variable("embed.weight")
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  
  param.cells <- lapply(1:num.rnn.layer, function(i) {
    if (cell.type == "lstm") {
      cell <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")), 
                   i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")), 
                   h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")), 
                   h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
    } else if (cell.type == "gru") {
      cell <- list(gates.i2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.i2h.weight")), 
                   gates.i2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.i2h.bias")), 
                   gates.h2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.h2h.weight")), 
                   gates.h2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.h2h.bias")), 
                   trans.i2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.i2h.weight")), 
                   trans.i2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.i2h.bias")), 
                   trans.h2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.h2h.weight")), 
                   trans.h2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.h2h.bias")))
    }
    return(cell)
  })
  
  # embeding layer
  label <- mx.symbol.Variable("label")
  data <- mx.symbol.Variable("data")
  data_mask_array <- mx.symbol.Variable("data.mask.array")
  data_mask_array <- mx.symbol.stop_gradient(data_mask_array, name = "data.mask.array")
  
  embed <- mx.symbol.Embedding(data = data, input_dim = input.size, weight = embed.weight, 
                               output_dim = num.embed, name = "embed")
  
  wordvec <- mx.symbol.split(data = embed, axis = 1, num.outputs = seq.len, squeeze_axis = T)
  data_mask_split <- mx.symbol.split(data = data_mask_array, axis = 1, num.outputs = seq.len, 
                                     squeeze_axis = T)
  
  last.hidden <- list()
  last.states <- list()
  decode <- list()
  softmax <- list()
  fc <- list()
  
  for (seqidx in 1:seq.len) {
    hidden <- wordvec[[seqidx]]
    
    for (i in 1:num.rnn.layer) {
      if (seqidx == 1) {
        prev.state <- init.state[[i]]
      } else {
        prev.state <- last.states[[i]]
      }
      
      if (cell.type == "lstm") {
        cell.symbol <- lstm.cell
      } else if (cell.type == "gru") {
        cell.symbol <- gru.cell
      }
      
      next.state <- cell.symbol(num.hidden = num.hidden, indata = hidden, prev.state = prev.state, 
                                param = param.cells[[i]], seqidx = seqidx, layeridx = i, dropout = dropout, 
                                data_masking = data_mask_split[[seqidx]])
      hidden <- next.state$h
      # if (dropout > 0) hidden <- mx.symbol.Dropout(data=hidden, p=dropout)
      last.states[[i]] <- next.state
    }
    
    # Decoding
    if (config == "one-to-one") {
      last.hidden <- c(last.hidden, hidden)
    }
  }
  
  if (config == "seq-to-one") {
    fc <- mx.symbol.FullyConnected(data = hidden, weight = cls.weight, bias = cls.bias, 
                                   num.hidden = num.label)
    
    loss <- mx.symbol.SoftmaxOutput(data = fc, name = "sm", label = label, ignore_label = ignore_label)
    
  } else if (config == "one-to-one") {
    last.hidden_expand <- lapply(last.hidden, function(i) mx.symbol.expand_dims(i, 
                                                                                axis = 1))
    concat <- mx.symbol.concat(last.hidden_expand, num.args = seq.len, dim = 1)
    reshape <- mx.symbol.reshape(concat, shape = c(num.hidden, -1))
    
    fc <- mx.symbol.FullyConnected(data = reshape, weight = cls.weight, bias = cls.bias, 
                                   num.hidden = num.label)
    
    label <- mx.symbol.reshape(data = label, shape = c(-1))
    loss <- mx.symbol.SoftmaxOutput(data = fc, name = "sm", label = label, ignore_label = ignore_label)
    
  }
  
  if (output_last_state) {
    group <- mx.symbol.Group(c(unlist(last.states), loss))
    return(group)
  } else {
    return(loss)
  }
}

# rnn.unroll.cudnn
rnn.unroll.cudnn <- function(num.rnn.layer, 
                             seq.len, 
                             input.size,
                             num.embed, 
                             num.hidden,
                             num.label,
                             dropout=0,
                             ignore_label=0,
                             init.state=NULL,
                             config,
                             cell.type="lstm",
                             output_last_state=F) {
  
  embed.weight <- mx.symbol.Variable("embed.weight")
  rnn.weight <- mx.symbol.Variable("rnn.weight")
  rnn.state.weight <- mx.symbol.Variable("rnn.state.weight")
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  
  # embeding layer
  label <- mx.symbol.Variable("label")
  data <- mx.symbol.Variable("data")
  data_mask_array <- mx.symbol.Variable("data.mask.array")
  data_mask_array <- mx.symbol.stop_gradient(data_mask_array, name = "data.mask.array")
  
  embed <- mx.symbol.Embedding(data=data, input_dim=input.size,
                               weight=embed.weight, output_dim=num.embed, name="embed")
  
  wordvec <- mx.symbol.split(data=embed, axis=1, num.outputs=seq.len, squeeze_axis=F)
  data_mask_split <- mx.symbol.split(data=data_mask_array, axis=1, num.outputs=seq.len, squeeze_axis=F)
  
  last.hidden <- list()
  last.states <- list()
  decode <- list()
  softmax <- list()
  fc <- list()
  
  seqidx <- 1
  
  for (seqidx in 1:seq.len) {
    hidden <- wordvec[[seqidx]]
    if (seqidx==1) {
      next.state <- mx.symbol.RNN(data=hidden, state=rnn.state.weight, parameters=rnn.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=T, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", seqidx, sep="_"))
    } else {
      next.state <- mx.symbol.RNN(data=hidden, state=next.state[[2]], parameters=rnn.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=T, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", seqidx, sep="_"))
    }
    
    # Decoding
    if (config=="one-to-one") {
      last.hidden <- c(last.hidden, next.state[[1]])
    }
  }
  
  if (config=="seq-to-one") {
    fc <- mx.symbol.FullyConnected(data=next.state[[1]],
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label)
    
    loss <- mx.symbol.SoftmaxOutput(data=fc, name="sm", label=label, ignore_label=ignore_label)
    
  } else if (config=="one-to-one"){
    
    last.hidden_expand = lapply(last.hidden, function(i) mx.symbol.expand_dims(i, axis=1))
    concat <-mx.symbol.concat(last.hidden_expand, num.args = seq.len, dim = 1)
    reshape = mx.symbol.Reshape(concat, shape=c(num.hidden, -1))
    
    fc <- mx.symbol.FullyConnected(data=reshape,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label)
    
    label <- mx.symbol.reshape(data=label, shape=c(-1))
    loss <- mx.symbol.SoftmaxOutput(data=fc, name="sm", label=label, ignore_label=ignore_label)
    
  }
  
  if (output_last_state){
    group <- mx.symbol.Group(c(unlist(last.states), loss))
    return(group)
  } else return(loss)
}


# single shot rnn
rnn.unroll.cudnn <- function(num.rnn.layer, 
                             seq.len, 
                             input.size,
                             num.embed, 
                             num.hidden,
                             num.label,
                             dropout=0,
                             ignore_label=0,
                             init.state=NULL,
                             config,
                             cell.type="lstm",
                             output_last_state=F) {
  
  embed.weight <- mx.symbol.Variable("embed.weight")
  rnn.weight <- mx.symbol.Variable("rnn.weight")
  rnn.state.weight <- mx.symbol.Variable("rnn.state.weight")
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  
  # embeding layer
  label <- mx.symbol.Variable("label")
  data <- mx.symbol.Variable("data")
  # data_mask_array <- mx.symbol.Variable("data.mask.array")
  # data_mask_array <- mx.symbol.stop_gradient(data_mask_array, name = "data.mask.array")
  
  data <- mx.symbol.transpose(data=data)
  embed <- mx.symbol.Embedding(data=data, input_dim=input.size,
                               weight=embed.weight, output_dim=num.embed, name="embed")
  
  # wordvec <- mx.symbol.split(data=embed, axis=1, num.outputs=seq.len, squeeze_axis=F)
  # data_mask_split <- mx.symbol.split(data=data_mask_array, axis=1, num.outputs=seq.len, squeeze_axis=F)
  
  last.hidden <- list()
  last.states <- list()
  decode <- list()
  softmax <- list()
  fc <- list()
  
  rnn <- mx.symbol.RNN(data=embed, state=rnn.state.weight, parameters=rnn.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=F, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", sep="_"))
  
  # Decoding
  # if (config=="one-to-one") {
  #   last.hidden <- c(last.hidden, next.state[[1]])
  # }
  
  if (config=="seq-to-one") {
    last.seq <- mx.symbol.SequenceLast(rnn[[1]], name = "last.seq")
    fc <- mx.symbol.FullyConnected(data=last.seq,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label)
    
    loss <- mx.symbol.SoftmaxOutput(data=fc, name="sm", label=label, ignore_label=ignore_label)
    
  } else if (config=="one-to-one"){
    
    last.hidden_expand = lapply(last.hidden, function(i) mx.symbol.expand_dims(i, axis=1))
    concat <-mx.symbol.concat(last.hidden_expand, num.args = seq.len, dim = 1)
    reshape = mx.symbol.reshape(concat, shape=c(num.hidden, -1))
    
    fc <- mx.symbol.FullyConnected(data=reshape,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label)
    
    label <- mx.symbol.reshape(data=label, shape=c(-1))
    loss <- mx.symbol.SoftmaxOutput(data=fc, name="sm", label=label, ignore_label=ignore_label)
    
  }
  
  if (output_last_state){
    group <- mx.symbol.Group(c(unlist(last.states), loss))
    return(group)
  } else return(loss)
}


# RNN test
# data <- mx.symbol.Variable("data")
# embed.weight <- mx.symbol.Variable("embed.weight")
# rnn.state.weight <- mx.symbol.Variable("rnn.state.weight")
# rnn.params.weight <- mx.symbol.Variable("rnn.params.weight")
# 
# batch.size <- 32
# seq.len <- 5
# input.size <- 50
# num.embed <- 8
# num.hidden <- 10
# seqidx <- 1
# num.rnn.layer <- 2
# 
# data <- mx.symbol.transpose(data=data)
# embed <- mx.symbol.Embedding(data=data, input_dim=input.size, weight=embed.weight, output_dim=num.embed, name="embed")
# rnn <- mx.symbol.RNN(data=embed, state=rnn.state.weight, parameters=rnn.params.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=T, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", seqidx, sep="_"))
# last.state <- mx.symbol.SequenceLast(rnn[[1]])
# 
# rnn$infer.shape(list(data=c(5, 32)))
# rnn$infer.shape(list(data=c(32, 5)))
# 
# last.state$infer.shape(list(data=c(5, 32)))
# last.state$infer.shape(list(data=c(32, 5)))
# 
# embed$infer.shape(list(data=c(5, 32)))
# embed$infer.shape(list(data=c(32, 5)))
# 
# embed <- mx.symbol.Embedding(data=data, input_dim=input.size, weight=embed.weight, output_dim=num.embed, name="embed")
# wordvec <- mx.symbol.split(data=embed, axis=1, num.outputs=seq.len, squeeze_axis=F)
# rnn <- mx.symbol.RNN(data=wordvec[[1]], state=rnn.state.weight, parameters=rnn.params.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=T, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", seqidx, sep="_"))
# last.state <- mx.symbol.SequenceLast(rnn[[1]])
# rnn$infer.shape(list(data=c(5, 32)))
