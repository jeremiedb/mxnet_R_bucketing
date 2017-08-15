require("mxnet")

###data for masking should be in the form : features x batch.size x max.seq.length according to SequenceMask
### Input data is in form max.seq.length x batch.size
### Embed is in dim features x max.seq.length x batch.size
### Concat dim: features x max.seq.length x batch.size

(xx<- mx.nd.array(array(1:60, dim=c(2,3,4))))

#### It works if using 2 transpositions...
(xx_t<- mx.nd.transpose(xx, axes = c(2,0,1)))
dim(xx_t)
mask<- mx.nd.array(c(1,2,3,3))
(xx_t_masked<- mx.nd.SequenceMask(data = xx_t, sequence_length = mask, use_sequence_length = T))
(xx_masked<- mx.nd.transpose(xx_t_masked, axes = c(2,0,1)))
dim(xx_masked)

(xx<- mx.nd.array(array(1:60, dim=c(4,3))))
dim(xx)


#######################
### Use symbols
data<- mx.symbol.Variable("data")
mask<- mx.symbol.Variable("mask")
xx_nd<- mx.nd.array(array(1:60, dim=c(2,3,4)))
mask_nd<- mx.nd.array(c(1,2,3,3))

xx_t<- mx.symbol.transpose(data, axes = c(2,0,1))
xx_t_masked<- mx.symbol.SequenceMask(data = xx_t, sequence_length = mask, use_sequence_length = T)
xx_masked<- mx.symbol.transpose(xx_t_masked, axes = c(2,0,1))

graph.viz(xx_masked, shape = c(2,3,4))

xx_masked$arguments
exec<- mxnet:::mx.symbol.bind(symbol = xx_masked, arg.arrays = c(list(data=xx_nd), list(mask=mask_nd)), aux.arrays = NULL, grad.reqs = c("null", "write"), ctx = mx.cpu())

exec$outputs$transpose7_output
mx.exec.forward(exec, is.train = T)
exec$outputs$transpose7_output

(mask<- mx.nd.array(array(c(1,1,1,0, 1,1,1,1, 1,0,0,0), dim=c(4,3))))
(mx.nd.reverse(data = mask, axis = 0))
(mx.nd.reverse(data = mask, axis = 1))

(mask<- mx.nd.array(array(c(1,1,1,0), dim=c(4))))
(mx.nd.reverse(data = mask, axis = 0))
(mx.nd.Reshape(data = mask, shape=c(4,1)))

final<- mx.nd.SequenceMask(data = xx, sequence_length = mask, use_sequence_length = T)



########################################
### Mask Array

mask<- mx.nd.array(array(c(1,1,0,1,1,1,1,0,0,0,0,0), dim=c(3,4)))
mask_expand<- mx.nd.Reshape(mask, shape=c(1,-2))
hidden<- mx.nd.array(array(1:60, dim=c(2,3,4)))
hidden_masked<- mx.nd.broadcast.mul(lhs = hidden, rhs=mask_expand)

mask2<- mx.nd.array(array(c(1,0,0,1), dim=c(4)))
mask_split1<- mx.nd.Reshape(data=mask2, shape=c(1,-2))
mask_split2<- mx.nd.expand.dims(data=mask2, axis = 1)
hidden<- mx.nd.array(array(1:60, dim=c(2,4)))
hidden_masked<- mx.nd.broadcast.mul(lhs = hidden, rhs=mask_split1)



mask2<- mx.nd.array(array(c(1,0,0,1), dim=c(4)))
mask_split1<- mx.nd.Reshape(data=mask2, shape=c(1,-2))
mask_split2<- mx.nd.expand.dims(data=mask2, axis = 1)
hidden<- mx.nd.array(array(1:60, dim=c(5,4)))
hidden_masked<- mx.nd.broadcast.mul(lhs = hidden, rhs=mask_split1)
