import numpy as np
import scipy
import scipy.signal
import theano.tensor.nnet
import theano.tensor as TT
import theano.tensor.extra_ops
from collections import OrderedDict


def weighted_sample(weights, objects):
    """
    Return a random item from objects, with the weighting defined by weights
    (which must sum to 1).
    """
    # An array of the weights, cumulatively summed.
    cs = np.cumsum(weights)
    # Find the index of the first weight over a random value.
    idx = sum(cs < np.random.rand())
    return objects[min(idx, len(objects) - 1)]


def weighted_sample_n(prob_matrix, items):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])
    k = (s < r.reshape((-1, 1))).sum(axis=1)
    n_items = len(items)
    return items[np.minimum(k, n_items - 1)]


# compute softmax for each row
def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    expx = np.exp(shifted)
    return expx / np.sum(expx, axis=-1, keepdims=True)


def softmax_sym(x):
    return theano.tensor.nnet.softmax(x)


# compute entropy for each row
def cat_entropy(x):
    return -np.sum(x * np.log(x), axis=-1)


# compute perplexity for each row
def cat_perplexity(x):
    return np.exp(cat_entropy(x))


def explained_variance_1d(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 0:
            return 0
        else:
            return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)


def to_onehot(ind, dim):
    ret = np.zeros(dim)
    ret[ind] = 1
    return ret


def to_onehot_n(inds, dim):
    ret = np.zeros((len(inds), dim))
    ret[np.arange(len(inds)), inds] = 1
    return ret


def to_onehot_sym(ind, dim):
    assert ind.ndim == 1
    return theano.tensor.extra_ops.to_one_hot(ind, dim)


def from_onehot(v):
    return np.nonzero(v)[0][0]


def from_onehot_n(v):
    if len(v) == 0:
        return []
    return np.nonzero(v)[1]


def normalize_updates(old_mean, old_std, new_mean, new_std, old_W, old_b):
    """
    Compute the updates for normalizing the last (linear) layer of a neural
    network
    """
    # Make necessary transformation so that
    # (W_old * h + b_old) * std_old + mean_old == \
    #   (W_new * h + b_new) * std_new + mean_new
    new_W = old_W * old_std[0] / (new_std[0] + 1e-6)
    new_b = (old_b * old_std[0] + old_mean[0] - new_mean[0]) / (new_std[0] + 1e-6)
    return OrderedDict([
        (old_W, TT.cast(new_W, old_W.dtype)),
        (old_b, TT.cast(new_b, old_b.dtype)),
        (old_mean, new_mean),
        (old_std, new_std),
    ])


def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def discount_return(x, discount):
    return np.sum(x * (discount ** np.arange(len(x))))


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    *y0*
        initial state vector

    *t*
        sample times

    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``

    *args*
        additional arguments passed to the derivative function

    *kwargs*
        additional keyword arguments passed to the derivative function

    Example 1 ::

        ## 2D system

        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2::

        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)

        y0 = 1
        yout = rk4(derivs, y0, t)


    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0
    i = 0

    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
