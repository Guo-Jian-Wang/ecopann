# -*- coding: utf-8 -*-

from . import cosmic_params
import numpy as np
import torch

#%% data conversion
def numpy2torch(data):
    """ Transfer data from the numpy array (on CPU) to the torch tensor (on CPU). """
    dtype = torch.FloatTensor
    data = torch.from_numpy(data).type(dtype)
    return data

def numpy2cuda(data, device=None):
    """ Transfer data from the numpy array (on CPU) to the torch tensor (on GPU). """
    if device is None:
        dtype = torch.cuda.FloatTensor
        data = torch.from_numpy(data).type(dtype)
    else:
        data = numpy2torch(data)
        data = torch2cuda(data, device=device)
    return data

def torch2cuda(data, device=None):
    """ Transfer data (torch tensor) from CPU to GPU. """
    return data.cuda(device=device)

def torch2numpy(data):
    """ Transfer data from the torch tensor (on CPU) to the numpy array (on CPU). """
    return data.numpy()

def cuda2torch(data):
    """ Transfer data (torch tensor) from GPU to CPU. """
    return data.cpu()

def cuda2numpy(data):
    """ Transfer data from the torch tensor (on GPU) to the numpy array (on CPU). """
    return data.cpu().numpy()

def cpu2cuda(data):
    """Transfer data from CPU to GPU.

    Parameters
    ----------
    data : array-like or tensor
        Numpy array or torch tensor.

    Raises
    ------
    TypeError
        The data type should be :class:`np.ndarray` or :class:`torch.Tensor`.

    Returns
    -------
    Tensor
        Torch tensor.

    """
    d_type = type(data)
    if d_type is np.ndarray:
        return numpy2cuda(data)
    elif d_type is torch.Tensor:
        return torch2cuda(data)
    else:
        raise TypeError('The data type should be numpy.ndarray or torch.Tensor')

#%% parameter scaling
#updated
class ParamsScaling(object):
    """Data preprocessing of cosmological parameters.
    
    Parameters
    ----------
    params_base : array-like
        A 1-D array that contains the base values of the cosmological parameters.
    """
    def __init__(self, params_base):
        self.params_base = params_base
   
    def scaling(self, params):
        return params / self.params_base
   
    def inverseScaling(self, params):
        return params * self.params_base

#%% statistic of a numpy array
class Statistic(object):
    """ Statistics of an array. """
    def __init__(self, x):
        self.x = x
        self.dtype = type(x)
    
    @property
    def mean(self):
        if self.dtype==np.ndarray:
            return float(np.mean(self.x))
        elif self.dtype==torch.Tensor:
            return torch.mean(self.x)
    
    @property
    def xmin(self):
        if self.dtype==np.ndarray:
            return float(np.min(self.x))
        elif self.dtype==torch.Tensor:
            return torch.min(self.x)
    
    @property
    def xmax(self):
        if self.dtype==np.ndarray:
            return float(np.max(self.x))
        elif self.dtype==torch.Tensor:
            return torch.max(self.x)
    
    @property
    def std(self):
        if self.dtype==np.ndarray:
            return float(np.std(self.x))
        elif self.dtype==torch.Tensor:
            return torch.std(self.x)
    
    def statistic(self):
        st = {'min' : self.xmin,
              'max' : self.xmax,
              'mean': self.mean,
              'std' : self.std,
              }
        return st

#%% normalization & inverse normalization
class Normalize(object):
    """ Normalize data. """
    def __init__(self, x, statistic={}, norm_type='z_score', a=0, b=1):
        self.x = x
        self.stati = statistic
        self.norm_type = norm_type
        self.a = a #only for minmax
        self.b = b #only for minmax
    
    def minmax(self):
        """min-max normalization
        
        Rescaling the range of features to scale the range in [0, 1] or [a,b]
        https://en.wikipedia.org/wiki/Feature_scaling
        """
        return self.a + (self.x-self.stati['min'])*(self.b-self.a) / (self.stati['max']-self.stati['min'])
    
    def mean(self):
        """ mean normalization """
        return (self.x-self.stati['mean'])/(self.stati['max']-self.stati['min'])
    
    def z_score(self):
        """ standardization/z-score/zero-mean normalization """
        return (self.x-self.stati['mean'])/self.stati['std']
    
    def norm(self):
        return eval('self.%s()'%self.norm_type)

class InverseNormalize(object):
    """ Inverse transformation of class :class:`~Normalize`. """
    def __init__(self, x1, statistic={}, norm_type='z_score', a=0, b=1):
        self.x = x1
        self.stati = statistic
        self.norm_type = norm_type
        self.a = a #only for minmax
        self.b = b #only for minmax
    
    def minmax(self):
        return (self.x-self.a) * (self.stati['max']-self.stati['min']) / (self.b-self.a) + self.stati['min']
    
    def mean(self):
        return self.x * (self.stati['max']-self.stati['min']) + self.stati['mean']
    
    def z_score(self):
        return self.x * self.stati['std'] + self.stati['mean']
    
    def inverseNorm(self):
        return eval('self.%s()'%self.norm_type)
