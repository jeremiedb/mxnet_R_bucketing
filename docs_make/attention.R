### Attention mechanisms

attn_key_create <- function(encode, num_proj_key = NULL) {
  
  # assign value to encode
  value = mx.symbol.identity(encode, name = "value")
  
  if (!is.null(num_proj_key)) {
    attn_key_weight = mx.symbol.Variable("attn_key_weight")
    key = mx.symbol.FullyConnected(data = encode, num_hidden = num_proj_key, weight = attn_key_weight, no_bias = T, flatten = F, name = "key")
  } else key = mx.symbol.identity(encode, name = "key")
  return(list(key = key, value = value))
}

attention_dot <- function(hidden, key, value, scaled = T, weighting = F, num_hidden, num_proj_key = NULL, query_proj_weight = NULL) {
  
  # query: either a copy of last hidden or projection of it
  if (!is.null(num_proj_key)) {
    query = mx.symbol.FullyConnected(data = hidden, num_hidden = num_proj_key, weight = query_proj_weight, no_bias = T, flatten = F)
    if (scaled) query = query / sqrt(num_proj_key)
    # score: [features x seq x batch] dot [1 x features x batch] -> [1 x seq x batch]
    query = mx.symbol.expand_dims(query, axis = 2)
    score = mx.symbol.batch_dot(lhs = key, rhs = query)
  } else if (weighting) {
    if (scaled) hidden = hidden / sqrt(num_hidden)
    query = mx.symbol.expand_dims(hidden, axis = 1)
    score = mx.symbol.broadcast_mul(query, key)
    score = mx.symbol.FullyConnected(data = score, num_hidden = 1, weight = query_proj_weight, no_bias = T, flatten = F)
  } else {
    query = mx.symbol.expand_dims(hidden, axis = 2)
    # score: [features x seq x batch] dot [1 x features x batch] -> [1 x seq x batch]
    score = mx.symbol.batch_dot(lhs = key, rhs = query)
  }

  # attention - softmax applied on seq_len axis
  attn_wgt = mx.symbol.softmax(score, axis = 1)
  # ctx vector:  [1 x seq x batch] dot [features x seq x batch] -> [features x 1 x batch]
  ctx_vector = mx.symbol.batch_dot(lhs = attn_wgt, rhs = value, transpose_a = T)
  ctx_vector = mx.symbol.reshape(ctx_vector, shape = c(0, -1), reverse=T)
  return(ctx_vector)
}


attention_ini <- function(key, value) {
  
  ini_weighting_weight = mx.symbol.Variable("ini_weighting_weight")
  score = mx.symbol.FullyConnected(data = key, num_hidden = 1, weight = ini_weighting_weight, no_bias = T, flatten = F)
  
  # attention - softmax applied on seq_len axis
  attn_wgt = mx.symbol.softmax(score, axis = 1)
  # ctx vector:  [1 x seq x batch] dot [features x seq x batch] -> [features x 1 x batch]
  ctx_vector = mx.symbol.batch_dot(lhs = attn_wgt, rhs = value, transpose_a = T)
  ctx_vector = mx.symbol.reshape(ctx_vector, shape = c(-1, 0))
  return(ctx_vector)
}
