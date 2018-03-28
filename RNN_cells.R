# LSTM cell symbol
lstm.cell <- function(num_hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0, prefix = "") {
  
  if (dropout > 0 && layeridx > 1) 
    indata <- mx.symbol.Dropout(data = indata, p = dropout)
  
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$i2h.weight, bias = param$i2h.bias, 
                                  num_hidden = num_hidden * 4, name = paste0(prefix, "t", seqidx, ".l", layeridx, ".i2h"))
  
  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$h2h.weight, 
                                    bias = param$h2h.bias, num_hidden = num_hidden * 4, 
                                    name = paste0(prefix, "t", seqidx, ".l", layeridx, ".h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }
  
  split.gates <- mx.symbol.split(gates, num.outputs = 4, axis = 1, squeeze.axis = F, 
                                 name = paste0(prefix, "t", seqidx, ".l", layeridx, ".slice"))
  
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
  
  return(list(h = next.h, c = next.c))
}


# SRU cell symbol
sru.cell <- function(num_hidden, indata, prev.state, param, seqidx, layeridx, dropout, prefix = "") {
  
  if (layeridx == 1) {
    proj_full <- mx.symbol.FullyConnected(data = indata, weight = param$input.weight, bias = param$input.bias,
                                          num_hidden = num_hidden*3, name = paste0(prefix, "input", "_l", layeridx, "_s", seqidx))
    
    proj_full <- mx.symbol.split(proj_full, num_outputs = 3, axis=1)
    
    # if (dropout > 0)
    #   in_proj <- mx.symbol.Dropout(data = in_proj, p = dropout)
    
  } else if (layeridx > 1) {
    # in_proj <- indata
    proj_full <- mx.symbol.FullyConnected(data = indata, weight = param$write.weight, bias = param$write.bias,
                                          num_hidden = num_hidden*3, name = paste0(prefix, "write", "_l", layeridx, "_s", seqidx))
    
    proj_full <- mx.symbol.split(proj_full, num_outputs = 3, axis=1)
    
    # if (dropout > 0)
    #   indata <- mx.symbol.Dropout(data = indata, p = dropout)
  }
  
  in_proj <- proj_full[[1]]
  forget <- mx.symbol.sigmoid(proj_full[[2]])
  reset <- mx.symbol.sigmoid(proj_full[[3]])
  
  if (is.null(prev.state)) {
    next.c <- (1 - forget) * in_proj
    next.h <- (1 - reset) * indata
  } else {
    next.c <- forget * prev.state$c + (1 - forget) * in_proj
    next.h <- reset * next.c + (1 - reset) * indata
  }
  
  return(list(h = next.h, c = next.c))
}



# Straight cell symbol
straight.cell <- function(num_hidden, indata, prev.state, param, seqidx, layeridx, dropout, prefix = "") {
  
  if (layeridx == 1) {
    # in_proj <- mx.symbol.FullyConnected(data = indata, weight = param$input.weight, bias = param$input.bias,
    #                                     num_hidden = num_hidden, name = paste0(prefix, "input", "_l", layeridx, "_s", seqidx))
    # if (dropout > 0) 
    #   in_proj <- mx.symbol.Dropout(data = in_proj, p = dropout)
    
  } else if (layeridx > 1) {
    # in_proj <- indata
    # if (dropout > 0)
    #   indata <- mx.symbol.Dropout(data = indata, p = dropout)
  }
  
  # proj_full <- mx.symbol.FullyConnected(data = indata, weight = param$write.weight, bias = param$write.bias, 
  #                                       num_hidden = num_hidden*5, name = paste0(prefix, "write", "_l", layeridx, "_s", seqidx))
  # 
  # proj_full <- mx.symbol.split(proj_full, num_outputs = 5, axis=1)
  
  # in_proj <- proj_full[[1]]
  # write <- mx.symbol.tanh(proj_full[[2]])
  # highway <- mx.symbol.sigmoid(proj_full[[3]])
  # read <- mx.symbol.sigmoid(proj_full[[4]])
  # mem <- proj_full[[5]]
  
  in_proj <- mx.symbol.FullyConnected(data = indata, weight = param$input.weight, bias = param$input.bias,
                                      num_hidden = num_hidden, name = paste0(prefix, "input", "_l", layeridx, "_s", seqidx))
  
  write <- mx.symbol.FullyConnected(data = indata, weight = param$write.weight, bias = param$write.bias,
                                    num_hidden = num_hidden, name = paste0(prefix, "write", "_l", layeridx, "_s", seqidx)) %>%
    mx.symbol.tanh()
  
  # mem <- mx.symbol.FullyConnected(data = indata, weight = param$mem.weight, bias = param$mem.bias,
  #                                 num_hidden = num_hidden, name = paste0(prefix, "mem", "_l", layeridx, "_s", seqidx))
  
  highway <- mx.symbol.FullyConnected(data = indata, weight = param$highway.weight, bias = param$highway.bias,
                                      num_hidden = num_hidden, name = paste0(prefix, "highway", "_l", layeridx, "_s", seqidx)) %>%
    mx.symbol.sigmoid()
  
  if (is.null(prev.state)) {
    next.c <- write * in_proj
    next.h <- highway * in_proj
  } else {
    read <- mx.symbol.FullyConnected(data = indata, weight = param$read.weight, bias = param$read.bias,
                                     num_hidden = num_hidden, name = paste0(prefix, "read", "_l", layeridx, "_s", seqidx)) %>%
      mx.symbol.sigmoid()
    
    next.c <- prev.state$c + write * in_proj
    next.h <- read * prev.state$c + highway * in_proj
  }
  
  return(list(h = next.h, c = next.c))
}



# Rich cell symbol
rich.cell <- function(num_hidden, indata, prev.state, param, seqidx, layeridx, dropout, prefix = "") {
  
  if (layeridx == 1) {
    in_proj <- mx.symbol.FullyConnected(data = indata, weight = param$input.weight, bias = param$input.bias,
                                        num_hidden = num_hidden, name = paste0(prefix, "input", "_l", layeridx, "_s", seqidx))
    if (dropout > 0)
      in_proj <- mx.symbol.Dropout(data = in_proj, p = dropout)
    
  } else if (layeridx > 1) {
    in_proj <- indata
    # if (dropout > 0) 
    #   in_proj <- mx.symbol.Dropout(data = in_proj, p = dropout)
  }
  
  mem <- mx.symbol.FullyConnected(data = indata, weight = param$mem.weight, bias = param$mem.bias, 
                                  num_hidden = num_hidden, name = paste0(prefix, "mem", "_l", layeridx, "_s", seqidx))
  
  write.in <- mx.symbol.FullyConnected(data = indata, weight = param$write.in.weight, bias = param$write.in.bias, 
                                       num_hidden = num_hidden, name = paste0(prefix, "write.in", "_l", layeridx, "_s", seqidx))
  
  highway <- mx.symbol.FullyConnected(data = indata, weight = param$highway.weight, bias = param$highway.bias, 
                                      num_hidden = num_hidden, name = paste0(prefix, "highway", "_l", layeridx, "_s", seqidx)) %>% 
    mx.symbol.sigmoid()
  
  if (is.null(prev.state)) {
    write <- mx.symbol.sigmoid(write.in)
    next.c <- mx.symbol.relu(write * mem)
    next.h <- highway * in_proj
  } else {
    
    read.in <- mx.symbol.FullyConnected(data = indata, weight = param$read.in.weight, bias = param$read.in.bias,
                                        num_hidden = num_hidden, name = paste0(prefix, "read.in", "_l", layeridx, "_s", seqidx))
    
    read.c <- mx.symbol.FullyConnected(data = prev.state$c, weight = param$read.c.weight, bias = param$read.c.bias,
                                       num_hidden = num_hidden, name = paste0(prefix, "read.c", "_l", layeridx, "_s", seqidx))
    
    write.c <- mx.symbol.FullyConnected(data = prev.state$c, weight = param$write.c.weight, bias = param$write.c.bias, 
                                        num_hidden = num_hidden, name = paste0(prefix, "write.c", "_l", layeridx, "_s", seqidx))
    
    read <- mx.symbol.tanh(read.in + read.c)
    write <- mx.symbol.sigmoid(write.in + write.c)
    
    next.c <- mx.symbol.relu(prev.state$c + write * mem)
    next.h <- read * prev.state$c + highway * in_proj
  }
  return(list(h = next.h, c = next.c))
}


# GRU cell symbol
gru.cell <- function(num_hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0, prefix)
{
  if (dropout > 0 && layeridx > 1) 
    indata <- mx.symbol.Dropout(data = indata, p = dropout)
  
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$gates.i2h.weight, 
                                  bias = param$gates.i2h.bias, num_hidden = num_hidden * 2, 
                                  name = paste0(prefix, "t", seqidx, ".l", layeridx, ".gates.i2h"))
  
  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$gates.h2h.weight, 
                                    bias = param$gates.h2h.bias, num_hidden = num_hidden * 2, 
                                    name = paste0(prefix, "t", seqidx, ".l", layeridx, ".gates.h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }
  
  split.gates <- mx.symbol.split(gates, num.outputs = 2, axis = 1, squeeze.axis = F, 
                                 name = paste0(prefix, "t", seqidx, ".l", layeridx, ".split"))
  
  update.gate <- mx.symbol.Activation(split.gates[[1]], act.type = "sigmoid")
  reset.gate <- mx.symbol.Activation(split.gates[[2]], act.type = "sigmoid")
  
  htrans.i2h <- mx.symbol.FullyConnected(data = indata, weight = param$trans.i2h.weight, 
                                         bias = param$trans.i2h.bias, num_hidden = num_hidden, 
                                         name = paste0(prefix, "t", seqidx, ".l", layeridx, ".trans.i2h"))
  
  if (is.null(prev.state)) {
    h.after.reset <- reset.gate * 0
  } else {
    h.after.reset <- prev.state$h * reset.gate
  }
  
  htrans.h2h <- mx.symbol.FullyConnected(data = h.after.reset, weight = param$trans.h2h.weight, 
                                         bias = param$trans.h2h.bias, num_hidden = num_hidden, 
                                         name = paste0(prefix, "t", seqidx, ".l", layeridx, ".trans.h2h"))
  
  h.trans <- htrans.i2h + htrans.h2h
  h.trans.active <- mx.symbol.Activation(h.trans, act.type = "tanh")
  
  if (is.null(prev.state)) {
    next.h <- update.gate * h.trans.active
  } else {
    next.h <- prev.state$h + update.gate * (h.trans.active - prev.state$h)
  }
  
  return(list(h = next.h))
}
