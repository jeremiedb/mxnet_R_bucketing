### multinomial sample

require("mxnet")

probs <- c(0, 0.1, 0.2, 0.3, 0.4)
probs <- cbind(probs, rev(probs))
probs <- mx.nd.array(probs)

# mx.nd.sample.multinomial(data = probs, shape=1, get_prob = F)

data <- mx.symbol.Variable("data")
sample_multi <- mx.symbol.sample_multinomial(data=data)
sample_unif <- mx.symbol.sample_uniform(data=data, low=1, high=2)

exec <- mx.simple.bind(symbol = sample_multi, ctx = mx.cpu(), grad.req = T, arg.arrays=probs)
