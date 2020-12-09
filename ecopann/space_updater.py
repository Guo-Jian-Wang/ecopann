# -*- coding: utf-8 -*-

from . import cosmic_params
import numpy as np


#%% obtain best-fit parameters from chains
#!!! to be updated, build a class for this part, move this function to plc and refer plc
def _quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples (a function in corner)
    
    This is a copy of quantile in corner (https://github.com/dfm/corner.py). Copyright (c) 2013-2015 Daniel Foreman-Mackey.
    
    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array-like[nsamples,]
       The samples.

    q : array-like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array-like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array-like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, 100.0 * q)
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()

class Chains(object):
    
    @staticmethod
    def bestFit(chain, best_values=None, symmetry_error=True):
        """Get the best-fit parameters from the chain.

        Parameters
        ----------
        chain : array-like
            The ANN chain.
        best_values : 1-dimension array, optional
            The best values of parameters. Default: None
        symmetry_error : bool, optional
            If True, obtain symmetrical errors, otherwise, obtain unsymmetrical errors. Default: True
        """
        n_dim = len(chain[0,:]) #the number of parameters (n dimension)
        if best_values is None:
            best_values = np.array([_quantile(chain[:,i], 0.5) for i in range(n_dim)])
            
        params = np.zeros((n_dim, 3))
        for i in range(n_dim):
            params[i, 0] = best_values[i]
            params[i, 1] = best_values[i] - _quantile(chain[:,i], 0.1587) # 1sigma left error
            params[i, 2] = _quantile(chain[:,i], 0.8413) - best_values[i] # 1sigma right error
        
        params_symmetry = np.c_[params[:,0], (params[:,1]+params[:,2])/2.]
        if symmetry_error:
            return params_symmetry
        else:
            return params
    
    @staticmethod
    def param_devs(chain_1, chain_2):
        """Get deviations of parameters obtained from two chains.
        """
        best_1 = Chains.bestFit(chain_1, best_values=None, symmetry_error=True)
        best_2 = Chains.bestFit(chain_2, best_values=None, symmetry_error=True)
        dev = abs(best_1[:,0]-best_2[:,0]) / np.sqrt(best_1[:,1]**2+best_2[:,1]**2)
        dev = [round(i, 4) for i in dev]
        return dev
    
    @staticmethod
    def error_devs(chain_1, chain_true):
        """Get the absolute values of the relative deviations of error of parameters obtained from two chains.
        """
        err_1 = Chains.bestFit(chain_1, best_values=None, symmetry_error=True)[:,1]
        err_true = Chains.bestFit(chain_true, best_values=None, symmetry_error=True)[:,1]
        dev = abs( (err_1 - err_true)/err_true )
        dev = [round(i, 4) for i in dev]
        return dev

#%% update parameter space
class CheckParameterSpace(object):
    
    @staticmethod
    def check_limit(p_space, limit_space):
        """Check the parameter space, to ensure that the parameter space does not exceed its limit range.
        
        Parameters
        ----------
        p_space : array-like
            The parameter space to be checked.
        limit_space : array-like
            The limit range of parameter space.
        
        Returns
        -------
        array-like
            A parameter space being limited by its limit range.
        """
        params_lower = p_space[:,0]
        params_upper = p_space[:,1]
        params_min = limit_space[:,0]
        params_max = limit_space[:,1]
        for i in range(len(params_lower)):
            if not np.isnan(params_min[i]):
                if params_lower[i] < params_min[i]:
                    params_lower[i] = params_min[i]
            if not np.isnan(params_max[i]):
                if params_upper[i] > params_max[i]:
                    params_upper[i] = params_max[i]
        return np.c_[params_lower, params_upper]

class UpdateParameterSpace(CheckParameterSpace):
    """Update parameter space.        
    
    Parameters
    ----------
    param_names : list
        A list that contains parameter names.
    init_params : array-like
        The initial settings of the parameter space.
    chain_1 : array-like
        The ANN chain of the i-th step, where :math:`i\geq2`.
    chain_0 : array-like, optional
        The ANN chain of the (i-1)-th step, where :math:`i\geq2`, if step :math:`\leq2`, ``chain_0`` should be set to None, 
        otherwise, ``chain_0`` should be an array. Default: None
    step : int, optional
        The number of step in the training process. Default: 2
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the base values, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    match_init : bool, optional
        If True, the new parameter space will in the initial set parameter space.
        If False, the new parameter space is only determined according to the prediction in the previous step, 
        in this case, the parameter may jump out of the initial set parameter space. Default: False
    """
    def __init__(self, param_names, init_params, chain_1, chain_0=None, step=2, params_dict=None, match_init=False):
        self.param_names = param_names
        self.params_limit = cosmic_params.ParamsProperty(param_names, params_dict=params_dict).params_limit        
        self.init_params = init_params
        self.chain_1 = chain_1
        self.chain_0 = chain_0
        self._step = step
        self.match_init = match_init
        self.spaceSigma_min = 5
    
    @property
    def step(self):
        if self._step>=2:
            return self._step
        else:
            raise ValueError('Parameter space can only be updated after the first step (step>=2)')
    
    @property
    def posterior_params(self):
        return Chains.bestFit(self.chain_1, symmetry_error=True)
    
    @property
    def params_dev(self):
        if self.step==2:
            best_1 = Chains.bestFit(self.chain_1)
            dev = abs((np.mean(self.init_params, axis=1) - best_1[:,0]) / best_1[:,1])
            dev = [round(i, 2) for i in dev]
            return dev
        elif self.step>=3:
            return Chains.param_devs(self.chain_1, self.chain_0)
    
    @property
    def params_error_dev(self):
        if self.step==2:
            best_1 = Chains.bestFit(self.chain_1)
            err_1 = best_1[:,1]
            #here we assume prior parameter space is [-5\sigma, +5\sigma]
            err_2 = (np.mean(self.init_params, axis=1) - self.init_params[:,0])/5.
            dev = abs((err_1 - err_2) / err_2)
            dev = [round(i, 2) for i in dev]
            return dev
        elif self.step>=3:
            return Chains.error_devs(self.chain_1, self.chain_0)
    
    #this method should be updated, deviation can not fully indecate the similarity between two chains, other method should be used
    #this method may should be removed ?, by adding the number of chains to be obtained?
    def small_dev(self, limit_dev=0.01):
        """A small value of deviation of parameters between two steps used to end the training process.
        
        Parameters
        ----------
        limit_dev : float, optional
            If the deviation of parameters between two steps smaller than this value, the training process will be over. Default: 0.01 (the deviation < 1%)

        Returns
        -------
        bool
            True (dev :math:`\leq` limit_dev) or False (dev>limit_dev)
        """
        if max(self.params_dev) <= limit_dev:
            if self.step==2:
                print('\nThe deviations of parameters between the first step and the initial parameters are: {}\\sigma'.format(self.params_dev))
            else:
                print('\nThe deviations of parameters between the above two steps are: {}\\sigma'.format(self.params_dev))
            print('The parameters are well constrained based on the deviation smaller than %s\\sigma\n'%limit_dev)
            return True
        else:
            if self.step==2:
                print('\nThe minimum/maximum deviations of parameters between the first step and the initial parameters are: {}/{}\\sigma'.format(min(self.params_dev), max(self.params_dev)))
            else:
                print('\nThe minimum/maximum deviations of parameters between the above two steps are: {}/{}\\sigma'.format(min(self.params_dev), max(self.params_dev)))
            return False
    
    def _auto_sigma(self, param_dev, critical_5=2/np.sqrt(2), critical_10=7/np.sqrt(2)):
        # the values of critical_* here are based on a hypothesis
        
        # critical_10: int, default: (10-3)/2**0.5, if max(self.param_dev)>critical_10, 'sigma' will be set to 20, 
        #              if critical_5<max(self.param_dev)<=critical_10, 'sigma' will be set to 10
        # critical_5: int, default: (5-3)/2**0.5, if max(self.param_dev)<=critical_5, 'sigma' will be set to 5
        if param_dev<critical_5:
            # sigma = 5
            sigma = self.spaceSigma_min
        elif critical_5<=param_dev<critical_10:
            sigma = 10
        elif param_dev>=critical_10:
            sigma = 20
        return sigma
    
    @property
    def sigmas(self):
        _sigmas = []
        for i in range(len(self.param_names)):
            _sigmas.append(self._auto_sigma(self.params_dev[i]))
        return np.array(_sigmas)
    
    def params_space(self, limiting=True):
        """Obtain the parameter space to be learned from chain.

        Parameters
        ----------
        limiting : bool, optional
            If True, the parameters space will be limited by the parameter limit space, otherwise, do nothing. Default: True
            
        Returns
        -------
        array-like
            Limited parameter space.
        """
        p_space = np.c_[self.posterior_params[:,0]-self.posterior_params[:,1]*self.sigmas, self.posterior_params[:,0]+self.posterior_params[:,1]*self.sigmas]
        
        if limiting:
            # params_limit = cosmic_params.params_limit(self.param_names)
            p_space = self.check_limit(p_space, self.params_limit)
        
        if self.match_init:
            p_space = self.check_limit(p_space, self.init_params)
        return p_space
    
    def limited_sigmas(self):
        p_space_limited = self.params_space(limiting=True)
        p_space_lower = p_space_limited[:,0]
        p_space_upper = p_space_limited[:,1]
        left_sigmas = (self.posterior_params[:,0]-p_space_lower) / self.posterior_params[:,1]
        right_sigmas = (p_space_upper-self.posterior_params[:,0]) / self.posterior_params[:,1]
        lim_sigmas = np.c_[left_sigmas, right_sigmas]
        return lim_sigmas
    
    @property
    def sigma_max(self):
        return round(np.max(self.limited_sigmas()), 1)
    
    @property
    def sigma_min(self):
        return round(np.min(self.limited_sigmas()), 1)
    
    def print_learningRange(self):
        p_space_limited = self.params_space(limiting=True)
        lim_sigmas = self.limited_sigmas()
        for i in range(len(self.param_names)):
            print('Learning range of %s: [%.6f, %.6f] ~ [-%.1f\sigma, +%.1f\sigma]'%(self.param_names[i], p_space_limited[i][0], p_space_limited[i][1], lim_sigmas[i][0], lim_sigmas[i][1]))
            # print('Note: if min(sigma)<3, this parameter will not be estimated well!!!') #???
        print('')
    
    def update(self):        
        print('Updating parameter space to be learned ...')
        p_space = self.params_space(limiting=True)
        self.print_learningRange()
        #!!!
        # check the leraning range, if min(abs(\sigma))<4 or <3.?, the prediction to params error will be not good
        # add warning
        return p_space
