# -*- coding: utf-8 -*-
import coplot.plot_contours as plc

from . import space_updater as su
from . import cosmic_params, utils
import numpy as np
import warnings
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from smt.sampling_methods import LHS
from matplotlib.path import Path
import matplotlib.pyplot as plt
import itertools
import math


#%% select samples from the local data sets or previous step samples
#to be updated to use new best-fit (best values & errors)???
#it seems this works well for hypercube & LHS, but not for hypersphere & hyperellipsoid, test this!!!
#because the params_space of hypersphere & hyperellipsoid will be a bit larger
class ParametersFilter(object):
    """Select cosmological parameters from a data set according to a given parameter space.
    
    Parameters
    ----------
    param_names : list
        A list that contains parameter names.
    sim_params : array-like
        The simulated cosmological parameters with the shape of (N, n), where N is the number of samples and n is the number of parameters.
    params_space : array-like
        The parameter space with the shape of (n, 2), where n is the number of parameters. For each parameter, it is: [lower_limit, upper_limit].
    prev_space : array-like
        The parameter space of local simulated data (or mock data in previous step), with shape of (n, 2), where n is the number of parameters. For each parameter, it is: [lower_limit, upper_limit].
    check_include : bool, optional
        If True, it will check whether ``params_space`` is in the space of ``sim_params``, otherwise, do nothing. Default: True
    rel_dev_limit : float, optional
        The limit of the relative deviation when ``params_space`` is not in the space of ``sim_params``, the default is 20% (this means if ``params_space`` is 
        :math:`[-5\sigma, +5\sigma]`, it can deviate :math:`<1\sigma` from ``sim_params``), note that it should be :math:`<0.4` (the deviation :math:`<2\sigma` for parameter space 
        :math:`[-5\sigma, +5\sigma]`). Default: 0.2
    """
    def __init__(self, param_names, sim_params, params_space, prev_space, check_include=True, rel_dev_limit=0.2):
        self.param_names = param_names
        self.sim_params = sim_params
        self.params_lower = params_space[:,0]
        self.params_upper = params_space[:,1]
        self.prev_space = prev_space
        self.check_include = check_include
        self.relDev_limit = rel_dev_limit
    
    def _check_relDev(self):
        if self.relDev_limit >= 0.4:
            warnings.warn('"rel_dev_limit" must be <0.4 for parameter space [-5\sigma, +5\sigma], it is better to set to 0.2', Warning)
    
    @property
    def include(self):
        """Check whether ``params_space`` is in the space of the ``sim_params``.
        
        Returns
        -------
        bool
            If ``params_space`` is in the space of the ``sim_params``, return True, otherwise, return False.
        """
        self._check_relDev()
        #old,
        # sim_params_min = np.min(self.sim_params, axis=0)
        # sim_params_max = np.max(self.sim_params, axis=0)
        # sim_params_mean = np.mean(self.sim_params, axis=0)
        #new
        sim_params_min = self.prev_space[:,0]
        sim_params_max = self.prev_space[:,1]
        sim_params_mean = np.mean(self.prev_space, axis=1)
        residual_lower_min = np.min(self.params_lower - sim_params_min)
        residual_upper_max = np.max(self.params_upper - sim_params_max)
        relativeDev_lower = (self.params_lower - sim_params_min)/(sim_params_mean - sim_params_min)
        relativeDev_upper = (self.params_upper - sim_params_max)/(sim_params_max - sim_params_mean)
        # print(min(relativeDev_lower), max(relativeDev_upper), '!!!!')
        if residual_lower_min>=0 and residual_upper_max<=0:
            return True
        elif min(relativeDev_lower)>=-self.relDev_limit and max(relativeDev_upper)<=self.relDev_limit:
            return True
        else:
            out_index_lower = np.where(relativeDev_lower<self.relDev_limit)[0]
            out_index_upper = np.where(relativeDev_upper>self.relDev_limit)[0]
            out_index = np.union1d(out_index_lower, out_index_upper)
            if len(out_index)==len(self.param_names):
                print('The parameter space to be learned is not included in the parameter space of the local data sets')
            else:
                for i in out_index:
                    print("Learning range of %s is not included in the parameter space of the local data sets"%(self.param_names[i]))
            return False
    
    def filter_index(self):
        residual_lower = self.sim_params - self.params_lower
        residual_upper = self.sim_params - self.params_upper
        residual_lower_min = np.min(residual_lower, axis=1)
        residual_upper_max = np.max(residual_upper, axis=1)
        index_min = np.where(residual_lower_min >= 0)
        index_max = np.where(residual_upper_max <= 0)
        self.index = np.intersect1d(index_min, index_max)
        return self.index
        
    def filter_params(self):
        if self.check_include:
            if self.include:
                self.filter_index()
                return self.sim_params[self.index]
        else:
            self.filter_index()
            return self.sim_params[self.index]

#%% simulate parameters
class CutParams(object):
    """Cut parameter samples that crossed the parameter limits.
    
    Parameters
    ----------
    param_names : list
        A list that contains parameter names.
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    """
    def __init__(self, param_names, params_dict=None):
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict)
        self.params_limit = p_property.params_limit
    
    def cut_params(self, params, params_limit=None):
        if params_limit is None:
            params_limit = self.params_limit
        idx_params, idx_edge = np.where(~np.isnan(params_limit))
        edge_values = params_limit[idx_params, idx_edge]
        for i in range(len(idx_params)):
            #left edge
            if idx_edge[i]==0:
                params = params[np.where(params[:,idx_params[i]]>edge_values[i])]
            #right edge
            elif idx_edge[i]==1:
                params = params[np.where(params[:,idx_params[i]]<edge_values[i])]
        return params

#updated
class SimParameters(CutParams):
    """Simulate parameters.
        
    Parameters
    ----------
    param_names : list
        A list that contains parameter names.
    chain : array-like or None
        The predicted ANN chain in the previous step. If ``chain`` is an array, ``params_space`` will be ignored.
        If ``chain`` is None, ``params_space`` should be given. Default: None
    params_space : array-like or None
        The parameter space with the shape of (n, 2), where n is the number of parameters. 
        For each parameter, it is: [lower_limit, upper_limit]. This is only used for space_type='hypercube' and space_type='LHS'
        If ``chain`` is an array, ``params_space`` will be ignored. If ``chain`` is None, ``params_space`` should be given. Default: None
    spaceSigma : int or array-like, optional
        The size of parameter space to be learned. It is a int or a numpy array with shape of (n,), where n is the number of parameters, 
        e.g. for spaceSigma=5, the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    space_type : str, optional
        The type of parameter space. It can be 'hypercube', 'LHS', 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid'. Default: 'hypercube'
    cut_crossedLimit : bool, optional
        If True, the data points that cross the parameter limits will be cut. This only works when space_type is 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid'. Default: True
    cut_crossedBest : bool, optional
        If True, the folded data points that cross the best values will be cut. It is recommended to set it to True. 
        This only works when space_type is 'hypersphere', 'hyperellipsoid', 'or 'posterior_hyperellipsoid', and when ``cut_crossedLimit=False``. Default: True
    cross_best : bool, optional
        If True, the folded data points will cross the best values, otherwise, the folded data points will not cross the best values. 
        This only works when space_type is 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid', and when ``cut_crossedLimit=False`` and ``cut_crossedBest=False``. Default: False
    
    Attributes
    ----------
    seed : None or int, optional
        Seed number which controls random draws. Default: None
        
    Note
    ----
    Either ``chain`` or ``params_space`` should be given to simulate samples.
    """
    def __init__(self, param_names, chain=None, params_space=None, spaceSigma=5, params_dict=None, 
                 space_type='hypercube', cut_crossedLimit=True, cut_crossedBest=True, cross_best=False):
        self.param_names = param_names
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict) #
        self.params_limit = p_property.params_limit #
        self.chain = self._chain(chain) #the input chain should be limited using the params_limit, modify in other palce???
        self.spaceSigma = spaceSigma
        self.space_type = self._space_type(space_type)
        self.params_space = self._params_space(params_space) #define after self.space_type
        self.cut_crossedLimit = cut_crossedLimit #
        self.cut_crossedBest = cut_crossedBest
        self.cross_best = cross_best
        self.seed = None
        self.max_error = True #test
        self.expand_factor = 0 #test, only works for hyperellipsoid
        
    @property
    def params_n(self):
        return len(self.param_names)
    
    def _chain(self, chain):
        if chain is None:
            return None
        else:
            chain = su.Chains.reshape_chain(chain)
            # using asymmetric error, modify space_updater???
            self.best_fit = su.Chains.bestFit(chain, best_type='mode', symmetry_error=False)
            # self.best_fit = su.Chains.bestFit(chain, best_type='median', symmetry_error=False) #further test???
            self.best = self.best_fit[:,0]
            self.sigma_max = np.max(self.best_fit[:,1:], axis=1)
            return chain
    
    def _space_type(self, space_type):
        if self.chain is None:
            if space_type=='hypersphere' or space_type=='hyperellipsoid' or space_type=='posterior_hyperellipsoid':
                raise ValueError("A chain should be given if space_type is set to 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid', otherwise, set space_type to 'hypercube' or 'LHS'")
        return space_type
    
    def _params_space(self, p_space):
        if self.chain is not None:
            if self.space_type=='posterior_hyperellipsoid':
                # this params_space will be used in self.posterior_hyperellipsoid
                return self.get_edge_space()
            else:
                p_space = np.c_[self.best-self.sigma_max*self.spaceSigma, self.best+self.sigma_max*self.spaceSigma]
                return su.CheckParameterSpace.check_limit(p_space, self.params_limit)
        else:
            if p_space is None:
                raise ValueError("An ANN chain should be given, otherwise, params_space should not be set to None.")
            else:
                return su.CheckParameterSpace.check_limit(p_space, self.params_limit)
    
    def uniform_params(self, N, p_space):
        if len(p_space.shape)==1:
            p_space = p_space.reshape(-1, 2) #for one parameter
        params_lower = p_space[:,0]
        params_upper = p_space[:,1]
        params_n = len(params_lower)
        return np.random.rand(N, params_n) * (params_upper-params_lower) + params_lower
    
    def hypercube(self, N):
        """Generate samples uniformly in a hypercube parameter space using uniform distribution.
        
        Parameters
        ----------
        N : int
            The number of data to be simulated.

        Returns
        -------
        array-like
            Parameters.
        """
        #see also https://ls11-www.cs.tu-dortmund.de/people/swessing/diversipy/doc/hycusampling.html
        #Beachkofski, B.; Grandhi, R. (2002). Improved Distributed Hypercube Sampling.
        print('Generating samples uniformly in a hypercube parameter space using uniform distribution')
        return self.uniform_params(N, self.params_space)

    def lhs(self, N):
        """Generate samples uniformly in a hypercube parameter space using Latin hypercube sampling.
        
        https://en.wikipedia.org/wiki/Latin_hypercube_sampling
        https://blog.csdn.net/yuxeaotao/article/details/108952326
        
        Parameters
        ----------
        N : int
            The number of data to be simulated.

        Returns
        -------
        array-like
            Parameters.
        """
        print('Generating samples uniformly in a hypercube parameter space using Latin hypercube sampling')
        #Note: When using the LHS method, it is necessary to simulate the data of training set size at one time.
        #pay attention to this when applying it???
        return LHS(xlimits=self.params_space, criterion='center', random_state=self.seed)(N)

    def random_ball(self, N, dimension, radius=1):
        """Generate samples uniformly in a ball with N dimension (hypersphere).
        
        https://www.cnpython.com/qa/349434
        https://www.zhihu.com/question/277712372
        https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html
        https://arxiv.org/pdf/1404.1347.pdf
        https://www.sciencedirect.com/science/article/pii/S0047259X10001211
        """
        # First generate random directions by normalizing the length of a vector of random-normal values (these distribute evenly on ball).
        random_directions = np.random.normal(size=(dimension, N))
        random_directions = random_directions / np.linalg.norm(random_directions, axis=0)
        # Second generate a random radius with probability proportional to the surface area of a ball with a given radius.
        random_radii = np.random.random(N)**(1.0/dimension)
        # Return the list of random (direction & length) points.
        return radius * (random_directions * random_radii).T
    
    def fold_sphere(self, params):
        """Fold the simulated parameters using the extremum of the parameters.
        
        https://en.wikipedia.org/wiki/Folded_normal_distribution
        """
        idx_params, idx_edge = np.where(~np.isnan(self.params_limit))
        edge_values = self.params_limit[idx_params, idx_edge]
        best_values = self.best[idx_params]
        for i in range(len(idx_params)):
            #left edge
            if idx_edge[i]==0:
                if np.min(params[:,idx_params[i]]) < edge_values[i]:
                    if self.cut_crossedBest:
                        cut_edge_l = 2*edge_values[i] - best_values[i]
                        params = params[np.where(params[:,idx_params[i]]>cut_edge_l)]
                    params_l = params[np.where(params[:,idx_params[i]]<edge_values[i])]
                    params_r = params[np.where(params[:,idx_params[i]]>edge_values[i])]
                    if not self.cross_best and max(abs(params_l[:,idx_params[i]]-edge_values[i]))>abs(best_values[i]-edge_values[i]):
                        a = 2*edge_values[i] - best_values[i]
                        b = edge_values[i]
                        p_min = min(params_l[:,idx_params[i]])
                        p_max = max(params_l[:,idx_params[i]])
                        params_l[:,idx_params[i]] = a + (params_l[:,idx_params[i]]-p_min)*(b-a) / (p_max-p_min)
                    params_l[:,idx_params[i]] = 2*edge_values[i] - params_l[:,idx_params[i]]
                    params = np.r_[params_l, params_r]
            #right edge
            elif idx_edge[i]==1:
                if np.max(params[:,idx_params[i]]) > edge_values[i]:
                    if self.cut_crossedBest:
                        cut_edge_r = 2*edge_values[i] - best_values[i]
                        params = params[np.where(params[:,idx_params[i]]<cut_edge_r)]
                    params_l = params[np.where(params[:,idx_params[i]]<edge_values[i])]
                    params_r = params[np.where(params[:,idx_params[i]]>edge_values[i])]
                    if not self.cross_best and max(abs(params_r[:,idx_params[i]]-edge_values[i]))>abs(best_values[i]-edge_values[i]):
                        a = edge_values[i]
                        b = 2*edge_values[i] - best_values[i]
                        p_min = min(params_r[:,idx_params[i]])
                        p_max = max(params_r[:,idx_params[i]])
                        params_r[:,idx_params[i]] = a + (params_r[:,idx_params[i]]-p_min)*(b-a) / (p_max-p_min)
                    params_r[:,idx_params[i]] = 2*edge_values[i] - params_r[:,idx_params[i]]
                    params = np.r_[params_l, params_r]
        return params
    
    def hypersphere(self, N):
        """Generate samples uniformly in a hypersphere parameter space.
        
        Parameters
        ----------
        N : int
            The number of data to be simulated.

        Returns
        -------
        array-like
            Parameters.
        """
        print('Generating samples uniformly in a hypersphere parameter space')
        radius = 1.0
        params_uncorr = self.random_ball(N, self.params_n, radius=radius)
        params_uncorr = params_uncorr * self.sigma_max / radius * self.spaceSigma + self.best #note the radius
        #cut or fold parameters
        if self.cut_crossedLimit:
            params_uncorr = self.cut_params(params_uncorr)
        else:
            params_uncorr = self.fold_sphere(params_uncorr)
        return params_uncorr
    
    def normal_params(self, N, best, sigma_max, spaceSigma):
        print('Generating Gaussian parameter space using standard deviation of the parameter')
        if isinstance(best, np.ndarray):
            params_n = len(best)
        else:
            params_n = 1
        return np.random.randn(N, params_n)*sigma_max/3.0 * spaceSigma + best
    
    def hyperellipsoid(self, N):
        """Generate samples uniformly in a hyperellipsoid parameter space using covariance between parameters.
        
        https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html
        https://blogs.sas.com/content/iml/2012/02/08/use-the-cholesky-transformation-to-correlate-and-uncorrelate-variables.html
        
        Parameters
        ----------
        N : int
            The number of data to be simulated.

        Returns
        -------
        array-like
            Parameters.
        
        Note
        ----
        For Cholesky decomposition, the covariance matrix :math:`C = LL^T`. So, the transformation relationship between 
        correlated parameters :math:`P_{corr}` and uncorrelated parameters :math:`P_{uncorr}` is 
        :math:`P_{corr} = LP_{uncorr}`, :math:`P_{uncorr} = L^{-1}P_{corr}`
        """
        #See also https://www.zhihu.com/question/268718682/answer/471626663 , note that u = L^-1 x should be u = L x 
        #check again for this method?
        if self.params_n==1:
            # params_corr = self.normal_params(N, self.best, self.sigma_max, self.spaceSigma) #normal distribution
            params_corr = self.hypersphere(N) #uniform distribution, use this, more reasonable
        else:
            print('Generating samples uniformly in a hyperellipsoid parameter space using covariance between parameters')
            radius = 1.0
            #Matrix will not positive definite when using max_error???, try to fix this
            # cov = su.Chains.cov_matrix(self.chain, max_error=True) #max_error is for using sigma_max
            cov = su.Chains.cov_matrix(self.chain, max_error=self.max_error, expand_factor=self.expand_factor) #test !!!
            params_uncorr = self.random_ball(N, self.params_n, radius=radius)
            #method 1
            # params_uncorr = params_uncorr / radius * self.spaceSigma #note the radius
            # L = np.linalg.cholesky(cov)
            # params_corr = L.dot(params_uncorr.T).T  + self.best
            #method 2, use this?
            sigma_max = np.sqrt(np.diagonal(cov))
            sigma_scale_factor = radius / sigma_max #note the radius
            idx = np.where(np.ones_like(cov, dtype=bool))
            cov_scaled = np.zeros_like(cov)
            cov_scaled[idx] = cov[idx] * sigma_scale_factor[idx[0]] * sigma_scale_factor[idx[1]]
            L = np.linalg.cholesky(cov_scaled)
            
            #test
            # factor_sigma = 0.2
            # # error_factor = np.abs(np.random.normal(1, factor_sigma, (len(L),len(L))))
            # error_factor = np.abs(np.random.normal(0, factor_sigma, (len(L),len(L)))) + 1
            # L = L * error_factor

            
            params_corr = L.dot(params_uncorr.T).T
            params_corr = params_corr / sigma_scale_factor / radius * self.spaceSigma + self.best #note the radius

        # cut or fold parameters
        if self.cut_crossedLimit:
            params_corr = self.cut_params(params_corr)
        else:
            params_corr = self.fold_sphere(params_corr)
        return params_corr
    
    #to be updated for plc._hist2d, do not plot figure???
    def _get_contour_edge(self, p1, p2, sigma=3):
        """Get the boundary of an ellipse using the chain.
        
        Parameters
        ----------
        p1 : array-like
            1-D array with shape of (N,). The chain of the first parameter.
        p2 : array-like
            1-D array with shape of (N,). The chain of the second parameter.
        sigma : int, optional
            Multiple of the standard deviation, which can be 1, 2, 3, 4, or 5. Default: 3
        
        Returns
        -------
        edge : array-like
            The boundary of the ellipse.
        """
        if sigma==1:
            level = 0.6826
        elif sigma==2:
            level = 0.9544
        elif sigma==3:
            level = 0.9974
        elif sigma==4:
            level = 0.99994
        elif sigma==5:
            level = 0.9999994
        contour_edge = plc._hist2d(p1, p2, bins=100, smooth=0, levels=(level,), line_width=1) #smooth: 0 or 1
        plt.close() #close the figure plotted in plc._hist2d
        segs = contour_edge.allsegs
        lens = np.array([len(segs[0][i]) for i in range(len(segs[0]))])
        idx_max = np.where(lens==max(lens))[0][0]
        edge = segs[0][idx_max]
        return edge
    
    @property
    def combinations(self):
        return [c for c in itertools.combinations(range(0, self.params_n), 2)]
    
    def get_contour_edges(self,sigma=3):
        #change the chain to a new chain such as a 5sigma chain
        chain_new = (self.chain - self.best) / 3.0 * self.spaceSigma + self.best
        edge_ij = []
        for idx in self.combinations:
            _edge_ij = self._get_contour_edge(chain_new[:,idx[0]], chain_new[:,idx[1]], sigma=sigma)
            p_property_ij = cosmic_params.ParamsProperty([self.param_names[idx[0]], self.param_names[idx[1]]])
            params_limit_ij = p_property_ij.params_limit
            _edge_ij = self.cut_params(_edge_ij, params_limit_ij)
            edge_ij.append(_edge_ij)
        return edge_ij
    
    def get_edge_space(self, sigma=3):
        self.edge_ij = self.get_contour_edges(sigma=sigma)
        space_mins = [[] for i in range(self.params_n)]
        space_maxs = [[] for i in range(self.params_n)]
        for i, idx in zip(range(len(self.combinations)), self.combinations):
            range_i_min, range_j_min = min(self.edge_ij[i][:,0]), min(self.edge_ij[i][:,1])
            range_i_max, range_j_max = max(self.edge_ij[i][:,0]), max(self.edge_ij[i][:,1])
            space_mins[idx[0]].append(range_i_min)
            space_mins[idx[1]].append(range_j_min)
            space_maxs[idx[0]].append(range_i_max)
            space_maxs[idx[1]].append(range_j_max)
        space_mins = np.array([max(e) for e in space_mins]).reshape(-1, 1)
        space_maxs = np.array([min(e) for e in space_maxs]).reshape(-1, 1)
        space = np.c_[space_mins, space_maxs]
        return su.CheckParameterSpace.check_limit(space, self.params_limit)
    
    def in_polygon(self, edge, x, y, get_points=True):
        """Judge whether the given points are in the area surrounded by the polygon.
        
        Parameters
        ----------
        edge : array-like
            2-D array with shape (N, 2). The vertices of a polygon.
        x : array-like
            1-D array with shape (M,). The x coordinate of the data points.
        y : array-like
            1-D array with shape (M,). The y coordinate of the data points.
        get_points : bool, optional
            If True, it will return data points inside the area, if False, it will return 
            a bool array which is True if the (closed) path contains the corresponding point. Default: True

        Returns
        -------
        array-like
            Points in the polygon.
        """
        # https://blog.csdn.net/weixin_43794311/article/details/121027299
        _path = Path(edge)
        points = np.c_[x, y]
        _in = _path.contains_points(points) #Judge whether the point is in the area.
        # _in = _path.contains_points(points, radius=-1e-10) #Judge whether the point is in the area or on the curve.
        if get_points:
            idx_in = np.where(_in==True)
            x_in = x[idx_in]
            y_in = y[idx_in]
            return x_in, y_in
        else:
            return _in
    
    def unique_elements(self, list_array):
        """Find the unique elements of a list which contains various of arrays.
        
        Parameters
        ----------
        list_array : list
            A list that contais various of arrays.

        Returns
        -------
        array-like
            The sorted unique elements of the list.
        """
        if len(list_array)==1:
            return list_array[0]
        elif len(list_array)==2:
            return np.intersect1d(list_array[0], list_array[1])
        elif len(list_array)>=3:
            uniq = np.intersect1d(list_array[0], list_array[1])
            for i in range(len(list_array)-2):
                uniq = np.intersect1d(uniq, list_array[i+2])
            return uniq
    
    def posterior_hyperellipsoid(self, N, factor=float):
        if self.params_n==1:
            params_corr = self.hypersphere(N) #uniform distribution
        else:
            print('Generating samples in a deformed hyperellipsoid parameter space using the posterior density (ANN chain) directly')
            if factor==float:
                factor = 1/0.43**self.params_n
            sim_points = self.uniform_params(int(N*factor), self.params_space)
            points_in = []
            for i, idx in zip(range(len(self.combinations)), self.combinations):
                points_ij = self.in_polygon(self.edge_ij[i], sim_points[:,idx[0]], sim_points[:,idx[1]], get_points=False)
                idx_in_ij = np.where(points_ij==True)[0]
                points_in.append(idx_in_ij)
            points_in = self.unique_elements(points_in)
            params_corr = sim_points[points_in]
            idx_rand = np.random.choice(len(params_corr), len(params_corr), replace=False)
            params_corr = params_corr[idx_rand]
            
        # cut or fold parameters
        if self.cut_crossedLimit:
            params_corr = self.cut_params(params_corr)
        else:
            params_corr = self.fold_sphere(params_corr)
        return params_corr
    
    def _get_params(self, N):
        if self.space_type=='hypercube':
            return self.hypercube(N)
        elif self.space_type=='LHS':
            return self.lhs(N)
        elif self.space_type=='hypersphere':
            return self.hypersphere(N)
        elif self.space_type=='hyperellipsoid':
            return self.hyperellipsoid(N)
        elif self.space_type=='posterior_hyperellipsoid':
            #here 0.43 is based on experiments and can also be set to other values
            factor = 1/0.43**self.params_n
            params = self.posterior_hyperellipsoid(N, factor=factor)
            while len(params)<N:
                factor = factor * math.ceil(N/len(params))
                params = self.posterior_hyperellipsoid(N, factor=factor)
            return params[:N]
    
    def get_params(self, N):
        np.random.seed(self.seed)
        if self.cut_crossedLimit or self.cut_crossedBest:
            factor = 1
            params = self._get_params(N*factor)
            while len(params)<N:
                factor += 1
                params = self._get_params(N*factor)
            return params[:N]
        else:
            return self._get_params(N)
    
    #test, multi_params & data_seed should be changed, to let seed and parameters random combination
    def get_multiParams(self, N, multi_params=1, use_dataSeed=False, reorder=True):
        params = self.get_params(N)
        
        if multi_params==1:
            #!!! do not use data seed
            return params
        else:
            multi_P = []
            for i in range(multi_params):
                multi_P.append(params)
            multi_P = np.concatenate(multi_P, axis=0)
            if reorder:
                index = np.random.choice(len(multi_P), len(multi_P), replace=False)
                multi_P = multi_P[index]
            if use_dataSeed:
                data_seed = np.random.choice(100000000, multi_params, replace=False) #test
                self.data_seed = np.repeat(data_seed, N, axis=0) #test
                if reorder:
                    self.data_seed = self.data_seed[index] #test
            return multi_P

#%% simulate training set
#updated
class SimSpectra(SimParameters):
    """Simulate training set.
    
    Parameters
    ----------
    N : int
        The number of data to be simulated.
    model : cosmological (or theoretical) model instance
        A cosmological (or theoretical) model instance that is used to simulate training set,
        it should contains a 'simulate' method, and 'simulate' should accept input of cosmological parameters,
        if you use the local data sets, it should also contain 'load_params', 'load_params_space', and 'load_sample' methods.
    param_names : list
        A list that contains parameter names.
    chain : array-like or None
        The predicted ANN chain in the previous step. If ``chain`` is an array, ``params_space`` will be ignored.
        If ``chain`` is None, ``params_space`` should be given. Default: None
    params_space : array-like or None
        The parameter space with the shape of (n, 2), where n is the number of parameters. 
        For each parameter, it is: [lower_limit, upper_limit]. This is only used for space_type='hypercube' and space_type='LHS'
        If ``chain`` is an array, ``params_space`` will be ignored. If ``chain`` is None, ``params_space`` should be given. Default: None
    spaceSigma : int or array-like, optional
        The size of parameter space to be learned. It is a int or a numpy array with shape of (n,), where n is the number of parameters, 
        e.g. for spaceSigma=5, the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    space_type : str, optional
        The type of parameter space. It can be 'hypercube', 'LHS', 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid'. Default: 'hypercube'
    cut_crossedLimit : bool, optional
        If True, the data points that cross the parameter limits will be cut. This only works when space_type is 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid'. Default: True
    cut_crossedBest : bool, optional
        If True, the folded data points that cross the best values will be cut. It is recommended to set it to True. 
        This only works when space_type is 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid', and when ``cut_crossedLimit=False``. Default: True
    cross_best : bool, optional
        If True, the folded data points will cross the best values, otherwise, the folded data points will not cross the best values. 
        This only works when space_type is 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid', and when ``cut_crossedLimit=False`` and ``cut_crossedBest=False``. Default: False
    local_samples : None, str, or list, optional
        Path of local samples, None, 'sample' or ['sample'] or ['sample_1', 'sample_2', ...].
        If None, no local samples are used. Default: None
    prevStep_data : None or list, optional
        Samples simulated in the previous step, if list, it should be [spectra, params].
        The spectra or params has shape (N, n), where N is the number of spectra and n is the number of data points in a spectrum. Default: None
    check_include : bool, optional
        If True, will check whether ``params_space`` is in the space of ``local_samples``, otherwise, do nothing. Default: True
    rel_dev_limit : float, optional
        The limit of the relative deviation when ``params_space`` is not in the space of ``sim_params``, the default is 20% (this means if ``params_space`` is 
        :math:`[-5\sigma, +5\sigma]`, it can deviate :math:`<1\sigma` from ``sim_params``), note that it should be :math:`<0.4` (the deviation :math:`<2\sigma` for parameter space 
        :math:`[-5\sigma, +5\sigma]`). Default: 0.2
    
    Attributes
    ----------
    prev_space : array-like
        The parameter space of local simulated data (or mock data in previous step), with shape of (n, 2), where n is the number of parameters. For each parameter, it is: [lower_limit, upper_limit].
    
    seed : None or int, optional
        Seed number which controls random draws. Default: None
        
    Note
    ----
    Either ``chain`` or ``params_space`` should be given to simulate samples.
    """
    def __init__(self, N, model, param_names, chain=None, params_space=None, spaceSigma=5,
                 params_dict=None, space_type='hypercube', cut_crossedLimit=True, cut_crossedBest=True, cross_best=False, 
                 local_samples=None, prevStep_data=None, check_include=True, rel_dev_limit=0.2):
        self.N = N
        self.model = model
        self.param_names = param_names
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict) #
        self.params_limit = p_property.params_limit #
        self.param_fullNames = p_property.param_fullNames #
        self.chain = self._chain(chain)
        self.spaceSigma = spaceSigma
        self.space_type = self._space_type(space_type)
        self.params_space = self._params_space(params_space) #define after self.space_type
        self.cut_crossedLimit = cut_crossedLimit
        self.cut_crossedBest = cut_crossedBest
        self.cross_best = cross_best
        self.local_samples = self._local_samples(local_samples)
        self.prevStep_data = prevStep_data
        self.check_include = check_include
        self.rel_dev_limit = rel_dev_limit   
        self.prev_space = None
        self.seed = None
        self.max_error = True #test
        self.expand_factor = 0 #test, only works for hyperellipsoid
        
    def _local_samples(self, loc_samples):
        if loc_samples is None:
            return loc_samples
        else:
            return utils.makeList(loc_samples)
    
    def comb_spectra(self, spectra_1, spectra_2):
        # Combine two sets of spectral samples.
        is_list = isinstance(spectra_1, list)
        if is_list:
            spectra_c = []
            for i in range(len(spectra_1)):
                spectra_c.append(np.r_[spectra_1[i], spectra_2[i]])
            return spectra_c
        else:
            if np.ndim(spectra_1)==1:
                spectra_1 = spectra_1.reshape(1, -1)
            if np.ndim(spectra_2)==1:
                spectra_2 = spectra_2.reshape(1, -1)
            return np.r_[spectra_1, spectra_2]
    
    def _filter_localSample(self, local_sample, N_local, local_params, p_filter):
        #print('Note: parameter space of the local samples should be large enough to cover the true parameter values (it is better to larger than prior paramter space)')
        index_chosen = p_filter.filter_index()
        if len(index_chosen) >= N_local:
            index_chosen = index_chosen[:N_local]
        
        #to be removed
#        if len(index_chosen) >= 1:
#            print('Indices of the last five samples chosen from the local data sets are: {}'.format(index_chosen[-5:]))
        
        #when len(index_chosen)=1, selected_* should be reshaped to (1,-1), to be improved, but this may not happen
        selected_params = local_params[index_chosen]
        
        #old, to be removed
        # selected_spectra = []
        # for i in range(len(index_chosen)):
        #     selected_spectra.append(self.model.load_sample(local_sample, index_chosen[i]))
        # selected_spectra = np.array(selected_spectra)
        
        #new, updated
        local_spectra = self.model.load_sample(local_sample)
        selected_spectra = local_spectra[index_chosen]
        return selected_spectra, selected_params
    
    def filter_localSample(self, local_sample, N_local):
        """Select samples from the local data sets.
        
        Parameters
        ----------
        local_sample : str
            Folders of local samples.
        N_local : int
            The number of local samples to be selected.

        Returns
        -------
        array-like
            The selected spectra and parameters.
        
        Note
        ----
        Parameter space of the local samples should be in the initial parameter space.
        """
        local_params = self.model.load_params(local_sample)
        self.local_space = self.model.load_params_space(local_sample)
        p_filter = ParametersFilter(self.param_names, local_params, self.params_space, self.local_space, check_include=self.check_include, rel_dev_limit=self.rel_dev_limit)
        if self.check_include:
            if p_filter.include:
                selected_spectra, selected_params = self._filter_localSample(local_sample, N_local, local_params, p_filter)
            else:
#                print('The parameter space to be learned is not included in the parameter space of the local data sets')
                selected_spectra, selected_params = [], []
        else:
            selected_spectra, selected_params = self._filter_localSample(local_sample, N_local, local_params, p_filter)
        return selected_spectra, selected_params
    
    #to be improved, for the if... else..., other functions like this should also be improved
    def filter_localSamples(self, N_local):
        print('Loading samples from the local data sets ...')
        if len(self.local_samples)==1:
            print('Local samples: %s'%(self.local_samples[0]))
            local_spectra, local_params = self.filter_localSample(self.local_samples[0], N_local)
            print('The number of samples selected: %s'%(len(local_params)))
            return local_spectra, local_params
        elif len(self.local_samples)>=2:
            print('Local samples: %s'%(self.local_samples[0]))
            local_spectra, local_params = self.filter_localSample(self.local_samples[0], N_local)
            print('The number of samples selected: %s'%(len(local_params)))
            if len(local_params)==N_local:
                return local_spectra, local_params
            else:
                missing_N = N_local - len(local_params)
                for sample in self.local_samples[1:]:
                    print('\nLocal samples: %s'%(sample))
                    local_spectra_2, local_params_2 = self.filter_localSample(sample, missing_N)
                    print('The number of samples selected: %s'%(len(local_params_2)))
                    if len(local_params)==0:
                        local_spectra, local_params = local_spectra_2, local_params_2
                    else:
                        if len(local_params_2)!=0:
                            local_spectra = self.comb_spectra(local_spectra, local_spectra_2)
                            local_params = np.r_[local_params, local_params_2]
                    if len(local_params)==N_local:
                        return local_spectra, local_params
                    missing_N = N_local - len(local_params)
                return local_spectra, local_params
    
    def _filter_previousSamples(self, N_pre, pre_spectra, pre_params, p_filter):
        index_chosen = p_filter.filter_index()
        if len(index_chosen) >= N_pre:
            index_chosen = index_chosen[:N_pre]
        #when len(index_chosen)=1, selected_* should be reshaped to (1,-1), to be improved, but this may not happen
        selected_params = pre_params[index_chosen]
        selected_spectra = pre_spectra[index_chosen]
        return selected_spectra, selected_params
        
    def filter_previousSamples(self, N_pre):
        """Select samples from the mock data simulated in the previous step.

        Parameters
        ----------
        N_pre : int
            The number of samples to be selected.

        Returns
        -------
        array-like
            The selected spectra and parameters.
        """
        print('Selecting samples from the mock data of the previous step ...')
        pre_spectra, pre_params = self.prevStep_data[0], self.prevStep_data[1]
        p_filter = ParametersFilter(self.param_names, pre_params, self.params_space, self.prev_space, check_include=self.check_include, rel_dev_limit=self.rel_dev_limit)
        if self.check_include:
            if p_filter.include:
                selected_spectra, selected_params = self._filter_previousSamples(N_pre, pre_spectra, pre_params, p_filter)
            else:
                print('The parameter space to be learned is not included in the parameter space of the the previous step samples')
                selected_spectra, selected_params = [], []
        else:
            selected_spectra, selected_params = [], []
        return selected_spectra, selected_params
    
    def simulate_spectra(self, N):
        params = self.get_params(N)
        spectra = []
        for i in range(N):
            if (i+1)%200==0:
                print('Simulating the sample: {}/{}'.format(i+1,N))
            spectra.append(self.model.simulate(params[i])[1])
        spectra = np.array(spectra)
        return spectra, params
    
    def simulate(self):
        missing_N = self.N
        #local samples
        if self.local_samples is not None:
            local_spectra, local_params = self.filter_localSamples(missing_N)
            print('%s sets of local samples are added to the training set\n'%(len(local_params)))
            if len(local_params)!=0:
                self.sim_spectra, self.sim_params = local_spectra, local_params
                missing_N = missing_N - len(local_params)
        
        #previous step samples, Note: if 'local_samples' is not None, previous step samples will not be used
        if self.local_samples is None and self.prevStep_data is not None and missing_N!=0:
            pre_spectra, pre_params = self.filter_previousSamples(missing_N)
            print('%s sets of samples of previous step are added to the training set\n'%(len(pre_params)))
            if len(pre_params)!=0:
                if missing_N==self.N:
                    self.sim_spectra, self.sim_params = pre_spectra, pre_params
                else:
                    self.sim_spectra = self.comb_spectra(self.sim_spectra, pre_spectra)
                    self.sim_params = np.r_[self.sim_params, pre_params]
                missing_N = missing_N - len(pre_params)
        
        #simulate samples
        if missing_N!=0:
            print('%s sets of new samples to be simulated ...'%(missing_N))
            new_spectra, new_params = self.simulate_spectra(missing_N)
            if len(new_params)==self.N:
                self.sim_spectra, self.sim_params = new_spectra, new_params
            else:
                self.sim_spectra = self.comb_spectra(self.sim_spectra, new_spectra)
                self.sim_params = np.r_[self.sim_params, new_params]
        return self.sim_spectra, self.sim_params
    
    #old, to be removed
    # def save_samples(self, path='sim_data/sample'):
    #     params = self.get_params(self.N)
    #     utils.savenpy(path, 'parameters', params, dtype=np.float32)
    #     self._save_paramsName(path)
    #     for i in range(self.N):
    #         if (i+1)%100==0:
    #             print('Simulating the samples: {}/{}'.format(i+1,self.N))
    #         x, y = self.model.simulate(params[i])
    #         utils.savenpy(path, 'y_%s'%i, y, dtype=np.float32)
    #     utils.savenpy(path, 'x', x, dtype=np.float32)
    
    #updated
    def save_samples(self, path='sim_data/sample'):
        params = self.get_params(self.N)
        utils.savenpy(path, 'parameters', params, dtype=np.float32)
        self._save_paramsName(path)
        self._save_params_space(path)
        spectra = []
        for i in range(self.N):
            if (i+1)%100==0:
                print('Simulating the samples: {}/{}'.format(i+1,self.N))
            x, y = self.model.simulate(params[i])
            spectra.append(y)
        spectra = np.array(spectra)
        utils.savenpy(path, 'x', x, dtype=np.float32)
        utils.savenpy(path, 'y', spectra, dtype=np.float32)

    #test
    def save_samples_2(self, multi_params=1, path='sim_data/sample', use_dataSeed=False):
        # params = self.get_multiParams(self.N, multi_params=multi_params)
        params = self.get_multiParams(self.N, multi_params=multi_params, use_dataSeed=use_dataSeed, reorder=True) #test

        utils.savenpy(path, 'parameters', params, dtype=np.float32)
        self._save_paramsName(path)
        self._save_params_space(path)
        spectra = []
        for i in range(self.N*multi_params):
            if (i+1)%100==0:
                print('Simulating the samples: {}/{}'.format(i+1,self.N*multi_params))
            if use_dataSeed:
                np.random.seed(self.data_seed[i]) #test
            x, y = self.model.simulate(params[i])
            spectra.append(y)
        spectra = np.array(spectra)
        utils.savenpy(path, 'x', x, dtype=np.float32)
        utils.savenpy(path, 'y', spectra, dtype=np.float32)
        
    #test
    def save_samples_3(self, part_size=10, multi_params=1, path='sim_data/sample'):
        params = self.get_multiParams(self.N, multi_params=multi_params)
        utils.savenpy(path, 'parameters', params, dtype=np.float32)
        self._save_paramsName(path)
        self._save_params_space(path)

        part_n = len(params) // part_size
        utils.savenpy(path, 'part_n_size', [part_n, part_size])
        for p in range(part_n):
            params_part = params[part_size*p:part_size*(p+1)]
            utils.savenpy(path, 'parameters_%s'%p, params_part, dtype=np.float32)
            
            spectra = []
            for i in range(part_size):
                if (i+1)%100==0:
                    print('Simulating the samples: {}/{}'.format(part_size*p+i+1,self.N*multi_params))
                x, y = self.model.simulate(params_part[i])
                spectra.append(y)
            spectra = np.array(spectra)
            utils.savenpy(path, 'x', x, dtype=np.float32)
            utils.savenpy(path, 'y_%s'%p, spectra, dtype=np.float32)

    #test, test before use
    def save_samples_3_onePart(self, params, part_size=10, part_idx=0, path='sim_data/sample'):
        if self.seed is None:
            raise ValueError("The attribute 'seed' should be assigned to an integer")
        utils.savenpy(path, 'parameters', params, dtype=np.float32)
        self._save_paramsName(path)
        self._save_params_space(path)
        
        part_n = len(params) // part_size
        if part_idx+1>part_n:
            raise ValueError("The number of part is %s, so the value of 'part_idx' should be <%s"%(part_n, part_n))
        utils.savenpy(path, 'part_n_size', [part_n, part_size])
        
        params_part = params[part_size*part_idx:part_size*(part_idx+1)]
        utils.savenpy(path, 'parameters_%s'%part_idx, params_part, dtype=np.float32)
        spectra = []
        for i in range(part_size):
            if (i+1)%100==0:
                print('Simulating the samples: {}/{}'.format(part_size*part_idx+i+1, len(params)))
            x, y = self.model.simulate(params_part[i])
            spectra.append(y)
        spectra = np.array(spectra)
        utils.savenpy(path, 'x', x, dtype=np.float32)
        utils.savenpy(path, 'y_%s'%part_idx, spectra, dtype=np.float32)
            
    def _save_paramsName(self, path):
        np.savetxt(path+"/paramNames.txt", self.param_fullNames, fmt='%s', delimiter='     ')
    
    def _save_params_space(self, path):
        utils.savenpy(path, 'params_space', self.params_space, dtype=np.float32)


#updated
class SimMultiSpectra(SimSpectra):
    """Simulate training set containing multiple observations (for multi-branch network).
    
    Parameters
    ----------
    branch_n : int
        The number of branch of the network.
    N : int
        The number of data to be simulated.
    model : cosmological (or theoretical) model instance
        A cosmological (or theoretical) model instance that is used to simulate training set,
        it should contains a 'simulate' method, and 'simulate' should accept input of cosmological parameters,
        if you use the local data sets, it should also contain 'load_params', 'load_params_space', and 'load_sample' methods.
    param_names : list
        A list that contains parameter names.
    chain : array-like or None
        The predicted ANN chain in the previous step. If ``chain`` is an array, ``params_space`` will be ignored.
        If ``chain`` is None, ``params_space`` should be given. Default: None
    params_space : array-like
        The parameter space with the shape of (n, 2), where n is the number of parameters. For each parameter, it is: [lower_limit, upper_limit].
    spaceSigma : int or array-like, optional
        The size of parameter space to be learned. It is a int or a numpy array with shape of (n,), where n is the number of parameters, 
        e.g. for spaceSigma=5, the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    space_type : str, optional
        The type of parameter space. It can be 'hypercube', 'LHS', 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid'. Default: 'hypercube'
    cut_crossedLimit : bool, optional
        If True, the data points that cross the parameter limits will be cut. This only works when space_type is 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid'. Default: True
    cut_crossedBest : bool, optional
        If True, the folded data points that cross the best values will be cut. It is recommended to set it to True. 
        This only works when space_type is 'hypersphere', 'hyperellipsoid',  or 'posterior_hyperellipsoid', and when ``cut_crossedLimit=False``. Default: True
    cross_best : bool, optional
        If True, the folded data points will cross the best values, otherwise, the folded data points will not cross the best values. 
        This only works when space_type is 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid', and when ``cut_crossedLimit=False`` and ``cut_crossedBest=False``. Default: False
    local_samples : None, str, or list, optional
        Path of local samples, None, 'sample' or ['sample'] or ['sample_1', 'sample_2', ...].
        If None, no local samples are used. Default: None
    prevStep_data : None or list, optional
        Samples simulated in the previous step, if list, it should be [spectra, params].
        The spectra or params has shape (N, n), where N is the number of spectra and n is the number of data points in a spectrum. Default: None
    check_include : bool, optional
        If True, will check whether ``params_space`` is in the space of ``local_samples``, otherwise, do nothing. Default: True
    rel_dev_limit : float, optional
        The limit of the relative deviation when ``params_space`` is not in the space of ``sim_params``, the default is 20% (this means if ``params_space`` is 
        :math:`[-5\sigma, +5\sigma]`, it can deviate :math:`<1\sigma` from ``sim_params``), note that it should be :math:`<0.4` (the deviation :math:`<2\sigma` for parameter space 
        :math:`[-5\sigma, +5\sigma]`). Default: 0.2
    
    Attributes
    ----------
    prev_space : array-like
        The parameter space of local simulated data (or mock data in previous step), with shape of (n, 2), where n is the number of parameters. For each parameter, it is: [lower_limit, upper_limit].
    
    seed : None or int, optional
        Seed number which controls random draws. Default: None
        
    Note
    ----
    Either ``chain`` or ``params_space`` should be given to simulate samples.
    """
    def __init__(self, branch_n, N, model, param_names, chain=None, params_space=None, spaceSigma=5,
                 params_dict=None, space_type='hypercube', cut_crossedLimit=True, cut_crossedBest=True, cross_best=False, 
                 local_samples=None, prevStep_data=None, check_include=True, rel_dev_limit=0.2):
        self.branch_n = branch_n
        self.N = N
        self.model = model
        self.param_names = param_names
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict) #
        self.params_limit = p_property.params_limit #
        self.param_fullNames = p_property.param_fullNames #
        self.chain = self._chain(chain)
        self.spaceSigma = spaceSigma
        self.space_type = self._space_type(space_type)
        self.params_space = self._params_space(params_space) #define after self.space_type
        self.cut_crossedLimit = cut_crossedLimit
        self.cut_crossedBest = cut_crossedBest
        self.cross_best = cross_best
        self.local_samples = self._local_samples(local_samples)
        self.prevStep_data = prevStep_data
        self.check_include = check_include
        self.rel_dev_limit = rel_dev_limit  
        self.prev_space = None
        self.seed = None
        self.max_error = True #test
        self.expand_factor = 0 #test, only works for hyperellipsoid
        
    def _filter_localSample(self, local_sample, N_local, local_params, p_filter):
        #print('Note: parameter space of the local samples should be large enough to cover the true parameter values (it is better to larger than prior paramter space)')
        index_chosen = p_filter.filter_index()
        if len(index_chosen) >= N_local:
            index_chosen = index_chosen[:N_local]
        
        #to be removed
#        if len(index_chosen) >= 1:
#            print('Indices of the last five samples chosen from the local data sets are: {}'.format(index_chosen[-5:]))
        
        #when len(index_chosen)=1, selected_* should be reshaped to (1,-1), to be improved, but this may not happen
        selected_params = local_params[index_chosen]
        
        #old, to be removed
        # selected_spectra = []
        # for comp in range(self.branch_n):
        #     _spectra = []
        #     for i in range(len(index_chosen)):
        #         _spectra.append(self.model.load_sample(local_sample, index_chosen[i])[comp])
        #     selected_spectra.append(np.array(_spectra))
            
        #new, updated
        local_spectra = self.model.load_sample(local_sample)
        selected_spectra = [local_spectra[comp][index_chosen] for comp in range(self.branch_n)]
        return selected_spectra, selected_params

    def _filter_previousSamples(self, N_pre, pre_spectra, pre_params, p_filter):
        index_chosen = p_filter.filter_index()
        if len(index_chosen) >= N_pre:
            index_chosen = index_chosen[:N_pre]
        #when len(index_chosen)=1, selected_* should be reshaped to (1,-1), to be improved, but this may not happen
        selected_params = pre_params[index_chosen]
        selected_spectra = []
        for comp in range(self.branch_n):
            selected_spectra.append(pre_spectra[comp][index_chosen])
        return selected_spectra, selected_params

    def simulate_spectra(self, N):
        params = self.get_params(N)
        for i in range(self.branch_n):
            exec('spectra_%s=[]'%i)

        for i in range(N):
            if (i+1)%100==0:
                print('Simulating the sample: {}/{}'.format(i+1,N))
            data = self.model.simulate(params[i])[1]
            for j in range(self.branch_n):
                exec('spectra_%s.append(data[j])'%j)
        
        spectra = []
        for i in range(self.branch_n):
            exec('spectra_%s=np.array(spectra_%s)'%(i,i))
            exec('spectra.append(spectra_%s)'%i)
        return spectra, params
    
    #old, to be removed
    # def save_samples(self, root_path='sim_data', branch_paths=['comp1','comp2']):
    #     params = self.get_params(self.N)
    #     utils.savenpy(root_path, 'parameters', params, dtype=np.float32)
    #     self._save_paramsName(root_path)
    #     for i in range(self.N):
    #         if (i+1)%100==0:
    #             print('Simulating the samples: {}/{}'.format(i+1,self.N))
    #         xx, yy = self.model.simulate(params[i])
    #         for j in range(self.branch_n):
    #             utils.savenpy(root_path+'/'+branch_paths[j], 'y_%s'%i, yy[j], dtype=np.float32)
    #     for j in range(len(branch_paths)):
    #         utils.savenpy(root_path+'/'+branch_paths[j], 'x', xx[j], dtype=np.float32)

    #new, updated
    def save_samples(self, path='sim_data', branch_paths=['comp1','comp2']):
        params = self.get_params(self.N)
        utils.savenpy(path, 'parameters', params, dtype=np.float32)
        self._save_paramsName(path)
        self._save_params_space(path)
        for i in range(self.branch_n):
            exec('spectra_%s=[]'%i)
        for i in range(self.N):
            if (i+1)%100==0:
                print('Simulating the samples: {}/{}'.format(i+1,self.N))
            xx, yy = self.model.simulate(params[i])
            for j in range(self.branch_n):
                exec('spectra_%s.append(yy[j])'%j)
        for j in range(self.branch_n):
            utils.savenpy(path+'/'+branch_paths[j], 'x', xx[j], dtype=np.float32)
            exec('spectra_%s=np.array(spectra_%s)'%(j,j))
            utils.savenpy(path+'/'+branch_paths[j], 'y', eval('spectra_%s'%j), dtype=np.float32)


#%% Add Gaussian noise
class AddGaussianNoise(object):
    """Add Gaussian noise for simulated data.
    
    Parameters
    ----------
    spectra : torch tensor, or a list of torch tensor
        The simulated spectra (data) with shape (N, spectra_length), or a list of spectra with shape [(N,spectra_length_1), (N,spectra_length_2), ...]
    params : torch tensor or None
        The simulated cosmological parameters. Default: None
    obs_errors : torch tensor, or a list of torch tensor, optional
        Observational errors (standard deviation) with shape (spectra_length,), or a list of errors with shape [(spectra_length_1,), (spectra_length_2,), ...]. Default: None
    cholesky_factor : torch tensor, a list of torch tensor, or None, optional
        Cholesky factor of covariance matrix with shape (spectra_length, spectra_length), or a list of Cholesky factor of covariance matrix with shape 
        [(spectra_length_1, spectra_length_1), (spectra_length_2, spectra_length_2), ...]. Default: None
    noise_type : str, optional
        The type of Gaussian noise added to the training set, 'singleNormal' or 'multiNormal'. Default: 'multiNormal'
    factor_sigma : float, optional
        For the case of 'singleNormal', it is the factor of the observational error (standard deviation), 
        while for the case of 'multiNormal' it is the standard deviation of the coefficient of the observational error (standard deviation). Default: 0.5
    multi_noise : int, optional
        The number of realization of noise added to a spectrum. Default: 5
    use_GPU : bool, optional
        If True, the noise will be generated by GPU, otherwise, it will be generated by CPU. Default: True
    """
    #combine obs_errors & cholesky_factor?
    def __init__(self, spectra, params=None, obs_errors=None, cholesky_factor=None, 
                 noise_type='multiNormal', factor_sigma=0.5, multi_noise=5, use_GPU=True):
        self.spectra = spectra
        self.params = params
        self.is_list = isinstance(self.spectra, list)
        self.obs_errors = obs_errors
        self.cholesky_factor = self._cholesky_factor(cholesky_factor)
        self.noise_type = noise_type
        self.factor_sigma = factor_sigma
        self.multi_noise = multi_noise
        self.use_GPU = use_GPU
        if use_GPU:
            self.epsilon = torch.cuda.FloatTensor([1e-20])
        else:
            self.epsilon = torch.FloatTensor([1e-20])
    
    def _cholesky_factor(self, cholesky_factor):
        if self.is_list and cholesky_factor is None:
            return [None for i in range(len(self.spectra))]
        else:
            return cholesky_factor
    
    #to be updated for 2D/3D maps
    def obs_noise(self, spectrum, obs_error=None, cholesky_factor=None):
        # Note 1: "torch.FloatTensor(ell_num).normal_().cuda()"(a) is equivalent to "torch.randn(ell_num).cuda()"(b)
        # and equivalent to "torch.cuda.FloatTensor(ell_num).normal_()"(c), in which (c) is faster than (a) and (b)
        # ##########################################################################################
        # Note 2: in the method of cudaErr, if the input 'data' is in torch.cuda.FloatTensor type,
        #         then the input 'data' will equal to the output 'data' ! To avoid this bug,
        #         the input 'data' should be in torch.FloatTensor type !
        # def cudaErr(data):
        #     for i in range(data.size(0)):
        #         data[i] = data[i] * 0.1
        #     return data
        # ###########################################################################################
        # Note 3: (i):   m=MultivariateNormal(torch.zeros(spectrum_leng), cov_torch)
        #         (ii):  m=MultivariateNormal(torch.zeros(spectrum_leng).cuda(), cov_cuda)
        #         (iii): m=MultivariateNormal(torch.cuda.FloatTensor(spectrum_leng).zero_(), cov_cuda)
        #         the speed of (i), (ii) and (iii) is (ii) > (i) > (iii)
        # Note 4: even though (ii) is faster, it is slower than (i) when using:
        #         batch_index = np.random.choice(len(self.inputs), self.batch_size, replace=False)
        #         xx = self.inputs[batch_index] # this step is slower, why?
        # Note 5: in MultivariateNormal, using scale_tril will be more efficient than covariance_matrix and precision_matrix,
        #         see: https://pytorch.org/docs/stable/distributions.html?highlight=multivariatenormal#torch.distributions.multivariate_normal.MultivariateNormal
        # Note 6: (i):  m = MultivariateNormal(mean, covariance_matrix=cov_matrix) #slower than the case when using scale_tril
        #         (ii): L = torch.cholesky(cov_matrix, upper=False) #cov=LL^T
        #               m = MultivariateNormal(mean, scale_tril=L) #using scale_tril will be more efficient
        #         case (ii) is faster than case (i)
        if cholesky_factor is None:
            if self.use_GPU:
                noise = torch.cuda.FloatTensor(spectrum.size()).normal_(0,1) * obs_error
            else:
                noise = torch.FloatTensor(spectrum.size()).normal_(0,1) * obs_error
        else:
            if self.use_GPU:
                mean = torch.zeros(cholesky_factor.size(-1)).cuda()
            else:
                mean = torch.zeros(cholesky_factor.size(-1))
            m = MultivariateNormal(mean, scale_tril=cholesky_factor)
            noise = m.sample((spectrum.size(0),))
        return noise
    
    #to be updated for 2D/3D maps
    def _singleNormalSpectra(self, spectrum, obs_error=None, cholesky_factor=None, error_factor=1):
        if cholesky_factor is None:
            noise = self.obs_noise(spectrum, obs_error=obs_error*error_factor)
        else:
            noise = self.obs_noise(spectrum, cholesky_factor=cholesky_factor*error_factor)
        return spectrum + noise
    
    #to be updated for 2D/3D maps
    def _multiNormalSpectra(self, spectrum, obs_error=None, cholesky_factor=None, factor_sigma=0.5):
        if cholesky_factor is None:
            if self.use_GPU:
                error_factor = torch.abs(torch.cuda.FloatTensor(spectrum.size(1)).normal_(0, factor_sigma)) #A !!!+, use this
                # error_factor = torch.abs(torch.cuda.FloatTensor(spectrum.size(0), 1).normal_(0, factor_sigma))
                # error_factor = torch.abs(torch.cuda.FloatTensor(spectrum.size()).normal_(0, factor_sigma)) !!!
                # error_factor = torch.abs(torch.cuda.FloatTensor(1).normal_(0, factor_sigma)) #test for map-->params
                # print(error_factor.shape, (obs_error*error_factor).shape)
            else:
                error_factor = torch.abs(torch.FloatTensor(spectrum.size(1)).normal_(0, factor_sigma))
                # error_factor = torch.abs(torch.FloatTensor(spectrum.size(0), 1).normal_(0, factor_sigma))
                # error_factor = torch.abs(torch.FloatTensor(spectrum.size()).normal_(0, factor_sigma))
                # error_factor = torch.abs(torch.FloatTensor(1).normal_(0, factor_sigma))
            noise = self.obs_noise(spectrum, obs_error=obs_error*error_factor)
        else:
            if self.use_GPU:
                #try to update this to make the result better???
                # method 1
                #the epsilon is used to ensure error_factor>0, because the diagonal of cholesky_factor*error_factor must be positive
                error_factor = torch.abs(torch.cuda.FloatTensor(1).normal_(0, factor_sigma)) + self.epsilon #A !!, use this
                
                # method 2, less good, need further test?
                # mean = torch.zeros(cholesky_factor.size(-1)).cuda()
                # m = MultivariateNormal(mean, scale_tril=cholesky_factor)
                # obs_error = m.sample((spectrum.size(0),))
                # error_factor = torch.abs(torch.cuda.FloatTensor(spectrum.size(0), 1).normal_(0, factor_sigma)) #!, less good
                # # error_factor = torch.abs(torch.cuda.FloatTensor(spectrum.size(1)).normal_(0, factor_sigma)) #?
                # # error_factor = torch.abs(torch.cuda.FloatTensor(1).normal_(0, factor_sigma)) #?
                # noise = obs_error * error_factor
                
                # method 3, the same as method 2, further test?
                # cov = cholesky_factor.clone()
                # error_factor = torch.abs(torch.cuda.FloatTensor(spectrum.size(1)).normal_(0, factor_sigma)) + 1e-4 #self.epsilon #A !!, use this
                # idx = torch.where(cov)
                # cov[idx] = cov[idx] * error_factor[idx[0]] * error_factor[idx[1]]
                # cholesky_factor_2 = torch.cholesky(cov)
                # noise = self.obs_noise(spectrum, cholesky_factor=cholesky_factor_2)
                
                # method 4, the same as method 3, further test? is more reasonable than method 5?, better?
                # cov = cholesky_factor.clone()
                # # print('ddddd????')
                # factor_sigma_2 = 0.1
                # # factor_sigma_2 = factor_sigma
                # error_factor_2 = torch.abs(torch.cuda.FloatTensor(spectrum.size(1)).normal_(1, factor_sigma_2))
                # idx = torch.where(cov)
                # cov[idx] = cov[idx] * error_factor_2[idx[0]] * error_factor_2[idx[1]]
                
                # error_factor = torch.abs(torch.cuda.FloatTensor(1).normal_(0, factor_sigma)) + self.epsilon #A !!, use this
                # cov = cov * error_factor**2
                
                # cholesky_factor_2 = torch.cholesky(cov)
                # noise = self.obs_noise(spectrum, cholesky_factor=cholesky_factor_2)
                
                # method 5, the same as method 4, but random cholesky_factor, right?
                # # factor_sigma_2 = 0.01 #random < 5%
                # # factor_sigma_2 = 0.03 #random < 10%
                # factor_sigma_2 = 0.2 #random < 100%
                # # factor_sigma_2 = factor_sigma
                # # print(factor_sigma_2, 'ddd???')
                # error_factor_2 = torch.abs(torch.cuda.FloatTensor(spectrum.size(1),spectrum.size(1)).normal_(1, factor_sigma_2))
                # # cholesky_f = cholesky_f * error_factor_2 #right?
                # error_factor = torch.abs(torch.cuda.FloatTensor(1).normal_(0, factor_sigma)) + self.epsilon #A !!, use this
                # noise = self.obs_noise(spectrum, cholesky_factor=cholesky_factor*error_factor_2*error_factor)
            else:
                #the epsilon is used to ensure error_factor>0, because the diagonal of cholesky_factor*error_factor must be positive
                error_factor = torch.abs(torch.FloatTensor(1).normal_(0, factor_sigma)) + self.epsilon
            # method 1
            noise = self.obs_noise(spectrum, cholesky_factor=cholesky_factor*error_factor)
        return spectrum + noise
    
    def singleNormalSpectra(self, error_factor=1):
        if self.is_list:
            noisy_sp = []
            for i in range(len(self.spectra)):
                _noisy_sp = self._singleNormalSpectra(self.spectra[i], obs_error=self.obs_errors[i], cholesky_factor=self.cholesky_factor[i], error_factor=error_factor)
                noisy_sp.append(_noisy_sp)
        else:
            noisy_sp = self._singleNormalSpectra(self.spectra, obs_error=self.obs_errors, cholesky_factor=self.cholesky_factor, error_factor=error_factor)
        return noisy_sp
    
    def multiNormalSpectra(self, factor_sigma=0.5):
        if self.is_list:
            noisy_sp = []
            for i in range(len(self.spectra)):
                _noisy_sp = self._multiNormalSpectra(self.spectra[i], obs_error=self.obs_errors[i], cholesky_factor=self.cholesky_factor[i], factor_sigma=factor_sigma)
                noisy_sp.append(_noisy_sp)
        else:
            noisy_sp = self._multiNormalSpectra(self.spectra, obs_error=self.obs_errors, cholesky_factor=self.cholesky_factor, factor_sigma=factor_sigma)
        return noisy_sp
    
    def noisySpectra(self):
        if self.noise_type=='singleNormal':
            return self.singleNormalSpectra(error_factor=self.factor_sigma)
        elif self.noise_type=='multiNormal':
            return self.multiNormalSpectra(factor_sigma=self.factor_sigma)
        else:
            raise ValueError("noise_type must be 'singleNormal' or 'multiNormal'")
    
    def noisySample(self):
        return self.noisySpectra(), self.params
    
    def multiParams(self):
        multi_P = []
        for i in range(self.multi_noise):
            multi_P.append(self.params)
        multi_P = torch.cat(multi_P, dim=0)
        return multi_P
    
    def multiNoisySpectra(self):
        _multi_NS = []
        for i in range(self.multi_noise):
            _multi_NS.append(self.noisySpectra())
        
        if self.is_list:
            multi_NS = []
            for comp in range(len(self.spectra)):
                oneComp_NS = []
                for i in range(self.multi_noise):
                    oneComp_NS.append(_multi_NS[i][comp])
                oneComp_NS = torch.cat(oneComp_NS, dim=0)
                multi_NS.append(oneComp_NS)
        else:
            multi_NS = torch.cat(_multi_NS, dim=0)
        return multi_NS
    
    def multiNoisySample(self, reorder=True):
        multi_P = self.multiParams()
        multi_NS = self.multiNoisySpectra()
        if reorder:
            index = np.random.choice(len(multi_P), len(multi_P), replace=False)
            multi_P = multi_P[index]
            if self.is_list:
                for comp in range(len(self.spectra)):
                    multi_NS[comp] = multi_NS[comp][index]
            else:
                multi_NS = multi_NS[index]
        return multi_NS, multi_P

