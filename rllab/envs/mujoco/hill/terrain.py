from scipy.stats import multivariate_normal
from scipy.signal import convolve2d
import matplotlib
try:
    matplotlib.pyplot.figure()
    matplotlib.pyplot.close()
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os

# the colormap should assign light colors to low values
TERRAIN_CMAP = 'Greens'
DEFAULT_PATH = '/tmp/mujoco_terrains'
STEP = 0.1

def generate_hills(width, height, nhills):
    '''
    @param width float, terrain width
    @param height float, terrain height
    @param nhills int, #hills to gen. #hills actually generted is sqrt(nhills)^2
    '''
    # setup coordinate grid
    xmin, xmax = -width/2.0, width/2.0
    ymin, ymax = -height/2.0, height/2.0
    x, y = np.mgrid[xmin:xmax:STEP, ymin:ymax:STEP]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    
    # generate hilltops
    xm, ym = np.mgrid[xmin:xmax:width/np.sqrt(nhills), ymin:ymax:height/np.sqrt(nhills)]
    mu = np.c_[xm.flat, ym.flat]
    sigma = float(width*height)/(nhills*8)
    for i in range(mu.shape[0]):
        mu[i] = multivariate_normal.rvs(mean=mu[i], cov=sigma)
    
    # generate hills
    sigma = sigma + sigma*np.random.rand(mu.shape[0])
    rvs = [ multivariate_normal(mu[i,:], cov=sigma[i]) for i in range(mu.shape[0]) ]
    hfield = np.max([ rv.pdf(pos) for rv in rvs ], axis=0)
    return x, y, hfield

def clear_patch(hfield, box):
    ''' Clears a patch shaped like box, assuming robot is placed in center of hfield
    @param box: rllab.spaces.Box-like
    '''
    if box.flat_dim > 2:
        raise ValueError("Provide 2dim box")
    
    # clear patch
    h_center = int(0.5 * hfield.shape[0])
    w_center = int(0.5 * hfield.shape[1])
    fromrow, torow = w_center + int(box.low[0]/STEP), w_center + int(box.high[0] / STEP)
    fromcol, tocol = h_center + int(box.low[1]/STEP), h_center + int(box.high[1] / STEP)
    hfield[fromrow:torow, fromcol:tocol] = 0.0
    
    # convolve to smoothen edges somewhat, in case hills were cut off
    K = np.ones((10,10)) / 100.0
    s = convolve2d(hfield[fromrow-9:torow+9, fromcol-9:tocol+9], K, mode='same', boundary='symm')
    hfield[fromrow-9:torow+9, fromcol-9:tocol+9] = s
    
    return hfield
    
def _checkpath(path_):
    if path_ is None:
        path_ = DEFAULT_PATH
    if not os.path.exists(path_):
        os.makedirs(path_)
    return path_
        
def save_heightfield(x, y, hfield, fname, path=None):
    '''
    @param path, str (optional). If not provided, DEFAULT_PATH is used. Make sure the path + fname match the <file> attribute
        of the <asset> element in the env XML where the height field is defined
    '''
    path = _checkpath(path)
    plt.figure()
    plt.contourf(x, y, -hfield, 100, cmap=TERRAIN_CMAP) # terrain_cmap is necessary to make sure tops get light color
    plt.savefig(os.path.join(path, fname), bbox_inches='tight')
    plt.close()

def save_texture(x, y, hfield, fname, path=None):
    '''
    @param path, str (optional). If not provided, DEFAULT_PATH is used. Make sure this matches the <texturedir> of the
        <compiler> element in the env XML
    '''
    path = _checkpath(path)
    plt.figure()
    plt.contourf(x, y, -hfield, 100, cmap=TERRAIN_CMAP)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    # for some reason plt.grid does not work here, so generate gridlines manually
    for i in np.arange(xmin,xmax,0.5):
        plt.plot([i,i], [ymin,ymax], 'k', linewidth=0.1)
    for i in np.arange(ymin,ymax,0.5):
        plt.plot([xmin,xmax],[i,i], 'k', linewidth=0.1)
    plt.savefig(os.path.join(path, fname), bbox_inches='tight')
    plt.close()