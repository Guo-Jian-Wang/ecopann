# -*- coding: utf-8 -*-

from . import cosmic_params
import numpy as np
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None


'''
This file should be updated !!!
'''

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

def pdf_1(X, bins=100, smooth=5):
    """Estimate the probability density function for the given data.
    """
    hist = np.histogram(X, bins, density=True) #normed=True --> density=True
    P, x = hist[0], hist[1]
    x = (x[:-1]+x[1:])/2.0
    P = gaussian_filter(P, smooth)
    return x, P

def get_cov(X, Y, mean=None):
    """Calculate covariance or variance.

    Parameters
    ----------
    X : array-like
        Random variable X.
    Y : array-like
         Random variable Y.
    mean : array-like or list, optional
        The mean values of X and Y. Default: None

    Returns
    -------
    float
        Covariance or variance.
    """
    #https://en.wikipedia.org/wiki/Covariance
    if mean is None:
        mu_X = np.mean(X)
        mu_Y = np.mean(Y)
    else:
        mu_X, mu_Y = mean[0], mean[1]
    return np.mean((X-mu_X)*(Y-mu_Y))

def get_CovMatrix(chain, params_n, best_values=None):
    """Calculate covariance matrix from a chain.
    
    Parameters
    ----------
    chain : array-like
        The ANN or MCMC chain with shape (N, M), where N is the number of chain and M is the number of parameters.
    params_n : array-like
        The number of parameters.
    best_values : array-like
        The best-fit values.
    
    Returns
    -------
    cov_matrix : array-like
        Covariance matrix.
    """
    if best_values is None:
        best_values = np.mean(chain, axis=0)
    cov_matrix = np.ones((params_n, params_n))
    for i in range(params_n):
        for j in range(params_n):
            cov_matrix[i, j] = get_cov(chain[:,i], chain[:,j], mean=[best_values[i], best_values[j]])
    return cov_matrix

class Chains(object):
    
    @staticmethod
    def reshape_chain(chain):
        if len(chain.shape)==1:
            chain = chain.reshape(-1, 1)
        return chain
    
    @staticmethod
    def params_n(chain):
        if len(chain.shape)==1:
            return 1
        elif len(chain.shape)==2:
            return chain.shape[1]
    
    @staticmethod
    def best_mode(chain, bins=100, smooth=5):
        """Take the mode as the best value.
        """
        chain = Chains.reshape_chain(chain)
        params_n = Chains.params_n(chain)
        best = np.zeros(params_n)
        for i in range(params_n):
            x, prob = pdf_1(chain[:,i], bins, smooth)
            best_idx = np.where(prob==np.max(prob))
            best[i] = x[best_idx]
        return best
    
    @staticmethod
    def best_median(chain):
        """Take the median as the best value.
        """
        chain = Chains.reshape_chain(chain)
        params_n = Chains.params_n(chain)
        return np.array([_quantile(chain[:,i], 0.5)[0] for i in range(params_n)])

    @staticmethod
    def sigma(chain, best_values, out_sigma=1):
        """Calculate the standard deviations.
        
        Parameters
        ----------
        chain : array-like
            The ANN chain.
        best_values : 1-dimension array
            The best values of parameters.
        out_sigma : int
            The output sigma, which can be 1, 2, or 3. Default: 1

        Returns
        -------
        sigma_1l, sigma_2l, sigma_3l : 1-dimension array
            The left 1 sigma, 2 sigma, or 3sigma deviations.
        sigma_1r, sigma_2r, sigma_3r : 1-dimension array
            The right 1 sigma, 2 sigma, or 3sigma deviations.
        """
        chain = Chains.reshape_chain(chain)
        params_n = Chains.params_n(chain)
        sigma1_left = np.zeros(params_n)
        sigma2_left = np.zeros(params_n)
        sigma3_left = np.zeros(params_n)
        sigma1_right = np.zeros(params_n)
        sigma2_right = np.zeros(params_n)
        sigma3_right = np.zeros(params_n)
        for i in range(params_n):
            sigma1_left_edge = _quantile(chain[:,i][np.where(chain[:,i]<best_values[i])], 0.3174)
            sigma2_left_edge = _quantile(chain[:,i][np.where(chain[:,i]<best_values[i])], 0.0456)
            sigma3_left_edge = _quantile(chain[:,i][np.where(chain[:,i]<best_values[i])], 0.0026)
            sigma1_right_edge = _quantile(chain[:,i][np.where(chain[:,i]>best_values[i])], 0.6826)
            sigma2_right_edge = _quantile(chain[:,i][np.where(chain[:,i]>best_values[i])], 0.9544)
            sigma3_right_edge = _quantile(chain[:,i][np.where(chain[:,i]>best_values[i])], 0.9974)
            sigma1_left[i] = best_values[i] - sigma1_left_edge
            sigma2_left[i] = best_values[i] - sigma2_left_edge
            sigma3_left[i] = best_values[i] - sigma3_left_edge
            sigma1_right[i] = sigma1_right_edge - best_values[i]
            sigma2_right[i] = sigma2_right_edge - best_values[i]
            sigma3_right[i] = sigma3_right_edge - best_values[i]
        if out_sigma==1:
            return sigma1_left, sigma1_right
        elif out_sigma==2:
            return sigma1_left, sigma1_right, sigma2_left, sigma2_right
        elif out_sigma==3:
            return sigma1_left, sigma1_right, sigma2_left, sigma2_right, sigma3_left, sigma3_right
        
    @staticmethod
    def bestFit(chain, best_type='mode', out_sigma=1, symmetry_error=True):
        """Get the best-fit parameters from the chain.

        Parameters
        ----------
        chain : array-like
            The ANN chain.
        best_type : str, optional
            The type of the best values of parameters, 'mode' or 'median'. If 'mode', it will take the mode as the best value. 
            If 'median', it will take the median as the best value. Default: 'mode'
        out_sigma : int
            The output sigma, which can be 1, 2, or 3. Default: 1
        symmetry_error : bool, optional
            If True, obtain symmetrical errors, otherwise, obtain unsymmetrical errors. Default: True
        """
        if best_type=='mode':
            best_values = Chains.best_mode(chain, bins=100, smooth=5)
        elif best_type=='median':
            best_values = Chains.best_median(chain)
        sigma1_left, sigma1_right, sigma2_left, sigma2_right, sigma3_left, sigma3_right = Chains.sigma(chain, best_values, out_sigma=3) #1sigma, 2sigma, 3sigma left/right error
        if out_sigma==1:
            best_fit = np.c_[best_values, sigma1_left, sigma1_right]
            best_fit_symError = np.c_[best_values, (sigma1_left+sigma1_right)/2.0]
        elif out_sigma==2:
            best_fit = np.c_[best_values, sigma1_left, sigma1_right, sigma2_left, sigma2_right]
            best_fit_symError = np.c_[best_values, (sigma1_left+sigma1_right)/2.0, (sigma2_left+sigma2_right)/2.0]
        elif out_sigma==3:
            best_fit = np.c_[best_values, sigma1_left, sigma1_right, sigma2_left, sigma2_right, sigma3_left, sigma3_right]
            best_fit_symError = np.c_[best_values, (sigma1_left+sigma1_right)/2.0, (sigma2_left+sigma2_right)/2.0, (sigma3_left+sigma3_right)/2.0]
        if symmetry_error:
            return best_fit_symError
        else:
            return best_fit
        
    @staticmethod
    def param_devs(chain_1, chain_2):
        """Get deviations of parameters obtained from two chains.
        """
        best_fit_1 = Chains.bestFit(chain_1, best_type='mode', symmetry_error=True)
        best_fit_2 = Chains.bestFit(chain_2, best_type='mode', symmetry_error=True)
        dev = abs(best_fit_1[:,0]-best_fit_2[:,0]) / np.sqrt(best_fit_1[:,1]**2+best_fit_2[:,1]**2)
        dev = [round(i, 4) for i in dev]
        return dev
    
    @staticmethod
    def error_devs(chain_1, chain_true):
        """Get the absolute values of the relative deviations of error of parameters obtained from two chains.
        """
        err_1 = Chains.bestFit(chain_1, best_type='mode', symmetry_error=True)[:,1]
        err_true = Chains.bestFit(chain_true, best_type='mode', symmetry_error=True)[:,1]
        dev = abs( (err_1 - err_true)/err_true )
        dev = [round(i, 4) for i in dev]
        return dev
    
    @staticmethod
    def cov_matrix(chain, max_error=True, expand_factor=0):
        """Get the covariance matrix.
        
        Parameters
        ----------
        chain : array-like
            The ANN chain.
        max_error : bool, optional
            If True, the diagonal elements of the covariance matrix will be replaced by the estimated maximum errors, 
            which is useful for non-Gaussian distribution. Default: True
        expand_factor : float, optional
            The expansion factor that is used to expand the error (the standard deviation) of each cosmological parameter.
            For example, if expand_factor=0.05, the error will has 5% expansion. It only works when max_error is True. Default: 0
        
        Returns
        -------
        cov : array-like
            The covariance matrix.
        """
        #remove expand_factor?
        best_fit = Chains.bestFit(chain, best_type='mode', symmetry_error=False)
        cov = get_CovMatrix(chain, chain.shape[1], best_values=best_fit[:,0])
        if max_error:
            sigma_max = np.max(best_fit[:,1:], axis=1)
            # this will ensure the Pearson correlation coefficient is in the range of [-1,1], and also ensure matrix is positive definite
            idx_big = np.where(sigma_max/np.sqrt(cov.diagonal()) > 1.0)[0]
            cov[idx_big, idx_big] = sigma_max[idx_big]**2
            diag_idx = np.diag_indices_from(cov)
            cov[diag_idx] = cov[diag_idx] * (1+expand_factor)**2
        
        #test
        # print(cov.shape)
        # cov[1,0] = cov[1,0] * 1.5
        # cov[0,1] = cov[0,1] * 1.5
        
        #test 2
        # factor_sigma = 0.1
        # error_factor = np.abs(np.random.normal(1, factor_sigma, len(cov)))
        # # error_factor = np.abs(np.random.normal(0, factor_sigma, len(cov))) + 1
        # print(error_factor)
        # idx = np.where(cov)
        # cov[idx] = cov[idx] * error_factor[idx[0]] * error_factor[idx[1]]
        
        
        #test 3
        # factor_sigma = 0.3
        # error_factor = np.abs(np.random.normal(1, factor_sigma, 1))
        # # error_factor = np.abs(np.random.normal(0, factor_sigma, 1)) + 1
        # print(error_factor)
        # cov = cov * error_factor * error_factor
        

        return cov

#%% update parameter space
class CheckParameterSpace(object):
    
    @staticmethod
    def check_limit(p_space, limit_space):
        """Check the parameter space to ensure that the parameter space does not exceed its limit range.
        
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
    step : int
        The number of step in the training process.
    param_names : list
        A list that contains parameter names.
    chain_1 : array-like
        The ANN chain of the i-th step, where :math:`i\geq2`.
    chain_0 : None or array-like, optional
        The ANN chain of the (i-1)-th step, where :math:`i\geq2`, if step :math:`\leq2`, ``chain_0`` should be set to None, 
        otherwise, ``chain_0`` should be an array. Default: None
    init_params : None or array-like
        The initial settings of the parameter space. If ``chain_0`` is given, ``init_params`` will be ignored. Default: None
    spaceSigma : int or array-like, optional
        The size of parameter space to be learned. It is a int or a numpy array with shape of (n,), where n is the number of parameters, 
        e.g. for spaceSigma=5, the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    """
    def __init__(self, step, param_names, chain_1, chain_0=None, init_params=None, spaceSigma=5, params_dict=None):
        self.step = self._step(step)
        self.param_names = param_names
        self.params_limit = cosmic_params.ParamsProperty(param_names, params_dict=params_dict).params_limit
        self.chain_1 = self._chain_1(chain_1)
        self.chain_0 = chain_0
        self.init_params = init_params
        if type(spaceSigma) is int:
            self.spaceSigma = np.array([spaceSigma for i in range(len(param_names))])
        else:
            self.spaceSigma = spaceSigma
        
    def _step(self, step):
        if step>=2:
            return step
        else:
            raise ValueError('Parameter space can only be updated after the first step (step>=2)')
    
    def _chain_1(self, chain):
        chain = Chains.reshape_chain(chain)
        self.best_fit_1 = Chains.bestFit(chain, best_type='mode', symmetry_error=False)
        self.best_1 = self.best_fit_1[:,0]
        self.sigma_max_1 = np.max(self.best_fit_1[:,1:], axis=1)
        self.sigma_mean_1 = (self.best_fit_1[:,1] + self.best_fit_1[:,2])/2.0
        return chain
    
    @property
    def param_devs(self):
        if self.step==2:
            if self.chain_0 is None:
                dev = abs((np.mean(self.init_params, axis=1) - self.best_1) / self.best_1)
                dev = [round(i, 4) for i in dev]
                return dev
            else:
                return Chains.param_devs(self.chain_1, self.chain_0)
        elif self.step>=3:
            return Chains.param_devs(self.chain_1, self.chain_0)
    
    @property
    def error_devs(self):
        if self.step==2:
            if self.chain_0 is None:
                #here we assume initial parameter space is [-5\sigma, +5\sigma]
                err_0 = (np.mean(self.init_params, axis=1) - self.init_params[:,0])/5.0
                dev = abs((self.sigma_mean_1 - err_0) / err_0)
                dev = [round(i, 4) for i in dev]
                return dev
            else:
                return Chains.error_devs(self.chain_1, self.chain_0)
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
        if max(self.param_devs) <= limit_dev:
            if self.step==2:
                print('\nThe maximum deviation of parameters between the first step and the initial parameters is: {:.2f}\\sigma'.format(max(self.param_devs)))
            else:
                print('\nThe maximum deviation of parameters between the above two steps is: {:.2f}\\sigma'.format(max(self.param_devs)))
            print('The parameters are well constrained based on the deviation smaller than %s\\sigma\n'%limit_dev)
            return True
        else:
            if self.step==2:
                print('\nThe minimum/maximum deviations of parameters (relative deviations of errors) between the first step and the initial parameters are: {:.2f}/{:.2f}\\sigma; ({:.2f}/{:.2f})'.format(min(self.param_devs), max(self.param_devs), min(self.error_devs), max(self.error_devs)))
            else:
                print('\nThe minimum/maximum deviations of parameters (relative deviations of errors) between the above two steps are: {:.2f}/{:.2f}\\sigma; ({:.2f}/{:.2f})'.format(min(self.param_devs), max(self.param_devs), min(self.error_devs), max(self.error_devs)))
            return False
    
    def _auto_sigma(self, space_sigma, param_dev, critical_5=2/np.sqrt(2), critical_10=7/np.sqrt(2)):
        # the values of critical_* here are based on a hypothesis
        
        # critical_10: int, default: (10-3)/2**0.5, if max(self.param_dev)>critical_10, 'sigma' will be set to 20, 
        #              if critical_5<max(self.param_dev)<=critical_10, 'sigma' will be set to 10
        # critical_5: int, default: (5-3)/2**0.5, if max(self.param_dev)<=critical_5, 'sigma' will be set to 5
        # we can also use the following settings: critical_5=(5-4)/2**0.5, critical_10=(10-4)/2**0.5
        if param_dev<critical_5:
            # sigma = 5
            sigma = space_sigma #also use error_dev???
        elif critical_5<=param_dev<critical_10:
            sigma = 10
        elif param_dev>=critical_10:
            sigma = 15
        return sigma
    
    @property
    def spaceSigma_all(self):
        _sigmas = []
        for i in range(len(self.param_names)):
            # _sigmas.append(self._auto_sigma(self.spaceSigma[i], self.param_devs[i]))
            _sigmas.append(self._auto_sigma(self.spaceSigma[i], self.param_devs[i], critical_5=1/np.sqrt(2), critical_10=6/np.sqrt(2)))
        return np.array(_sigmas)
    
    def params_space(self):
        """Obtain the parameter space to be learned from chain.
        
        Returns
        -------
        array-like
            Limited parameter space.
        """
        p_space = np.c_[self.best_1-self.sigma_max_1*self.spaceSigma_all, self.best_1+self.sigma_max_1*self.spaceSigma_all]
        return self.check_limit(p_space, self.params_limit)
    
    def limited_spaceSigma_all(self):
        p_space_limited = self.params_space()
        p_space_lower = p_space_limited[:,0]
        p_space_upper = p_space_limited[:,1]
        left_sigmas = (self.best_1 - p_space_lower) / self.sigma_max_1
        right_sigmas = (p_space_upper - self.best_1) / self.sigma_max_1
        lim_sigmas = np.c_[left_sigmas, right_sigmas]
        return lim_sigmas
    
    @property
    def spaceSigma_max(self):
        return round(np.max(self.limited_spaceSigma_all()), 1)
    
    @property
    def spaceSigma_min(self):
        return round(np.min(self.limited_spaceSigma_all()), 1)
    
    def print_learningRange(self):
        print('Updating parameter space to be learned ...')
        p_space_limited = self.params_space()
        lim_sigmas = self.limited_spaceSigma_all()
        for i in range(len(self.param_names)):
            print('Learning range of %s: [%.6f, %.6f] ~ [-%.1f\sigma, +%.1f\sigma]'%(self.param_names[i], p_space_limited[i][0], p_space_limited[i][1], lim_sigmas[i][0], lim_sigmas[i][1]))
            # print('Note: if min(sigma)<3, this parameter will not be estimated well!!!') #???
        print('')

