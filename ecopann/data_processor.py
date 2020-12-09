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
class ParamsScaling(object):
    """Data preprocessing of cosmological parameters.
    
    Parameters
    ----------
    parameters : array-like
        Cosmological parameters of the training set, with shape (N, n), where N is the number of the training set and n is the number of parameters.
    param_names : list
        A list which contains the parameter names, e.g. ['H0','Omega_m','ombh2','omch2','tau','As','ns'].
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the base values, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    """
    def __init__(self, parameters, param_names, params_dict=None):
        self.params = parameters
        self.params_base = cosmic_params.ParamsProperty(param_names,params_dict=params_dict).params_base
   
    def scaling(self):
        return self.params / self.params_base
   
    def inverseScaling(self):
        return self.params * self.params_base

#%% statistic of a numpy array
class Statistic(object):
    """ Statistics of an array. """
    def __init__(self, x):
        self.x = x
    
    @property
    def mean(self):
        return np.mean(self.x)
    
    @property
    def xmin(self):
        return np.min(self.x)
    
    @property
    def xmax(self):
        return np.max(self.x)
    
    @property
    def std(self):
        return np.std(self.x)
    
    def statistic(self):
        st = {'min' : float(self.xmin),
              'max' : float(self.xmax),
              'mean': float(self.mean),
              'std' : float(self.std),
              }
        return st

#%% normalization & inverse normalization
class Normalize(object):
    """ Normalize data. """
    def __init__(self, x, statistic={}, norm_type='z_score'):
        self.x = x
        self.stati = statistic
        self.norm_type = norm_type
    
    def minmax(self):
        """min-max normalization
        
        Rescaling the range of features to scale the range in [0, 1] or [a,b]
        https://en.wikipedia.org/wiki/Feature_scaling
        """
        return (self.x-self.stati['min'])/(self.stati['max']-self.stati['min'])
    
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
    def __init__(self, x1, statistic={}, norm_type='z_score'):
        self.x = x1
        self.stati = statistic
        self.norm_type = norm_type
    
    def minmax(self):
        return self.x * (self.stati['max']-self.stati['min']) + self.stati['min']
    
    def mean(self):
        return self.x * (self.stati['max']-self.stati['min']) + self.stati['mean']
    
    def z_score(self):
        return self.x * self.stati['std'] + self.stati['mean']
    
    def inverseNorm(self):
        return eval('self.%s()'%self.norm_type)
