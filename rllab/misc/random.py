from theano.tensor.shared_randomstreams import RandomStreams
_srng = RandomStreams(seed=234)

def uniform(size=(), low=0.0, high=1.0, ndim=None):
    return _srng.uniform(size=size, low=low, high=high, ndim=ndim)
