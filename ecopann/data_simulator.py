# -*- coding: utf-8 -*-

from . import space_updater as su
from . import cosmic_params, utils
import numpy as np
import warnings
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


#%% select samples from the local data sets or previous step samples
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
    check_include : bool, optional
        If True, it will check whether ``params_space`` is in the space of ``sim_params``, otherwise, do nothing. Default: True
    rel_dev_limit : float, optional
        The limit of the relative deviation when ``params_space`` is not in the space of ``sim_params``, the default is 20% (this means if ``params_space`` is 
        :math:`[-5\sigma, +5\sigma]`, it can deviate :math:`<1\sigma` from ``sim_params``), note that it should be :math:`<0.4` (the deviation :math:`<2\sigma` for parameter space 
        :math:`[-5\sigma, +5\sigma]`). Default: 0.2
    """
    def __init__(self, param_names, sim_params, params_space, check_include=True, rel_dev_limit=0.2):
        self.param_names = param_names
        self.sim_params = sim_params
        self.params_lower = params_space[:,0]
        self.params_upper = params_space[:,1]
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
        sim_params_min = np.min(self.sim_params, axis=0)
        sim_params_max = np.max(self.sim_params, axis=0)
        sim_params_mean = np.mean(self.sim_params, axis=0)
        residual_lower_min = np.min(self.params_lower - sim_params_min)
        residual_upper_max = np.max(self.params_upper - sim_params_max)
        relativeDev_lower = (self.params_lower - sim_params_min)/(sim_params_mean - sim_params_min)
        relativeDev_upper = (self.params_upper - sim_params_max)/(sim_params_max - sim_params_mean)
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

#%% simulate training set
class Spectra(object):
    """Simulate training set.
    
    Parameters
    ----------
    model : cosmological (or theoretical) model instance
        A cosmological (or theoretical) model instance that is used to simulate training set,
        it should contains a 'simulate' method, and 'simulate' should accept input of cosmological parameters,
        if you use the local data sets, it should also contain 'load_params' and 'load_sample' methods.
    param_names : list
        A list that contains parameter names.
    params_space : array-like
        The parameter space with the shape of (n, 2), where n is the number of parameters. For each parameter, it is: [lower_limit, upper_limit].
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the base values, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    step : int, optional
        The number of step of the training process. Default: 1
    N : int, optional
        The number of data to be simulated. Default: 5000
    base_N : int, optional
        The basic or minimum number of data to be simulated, which works only when ``auto_N`` is set to True. Default: 2000
    auto_N : bool, optional
        If True, ``N`` is set automatically and the value of ``N`` will be ignored in some cases according to parameter space. Default: True
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
    """
    def __init__(self, model, param_names, params_space, params_dict=None, step=1, N=5000, base_N=2000, auto_N=True,
                 local_samples=None, prevStep_data=None, check_include=True, rel_dev_limit=0.2):
        self.model = model
        self.param_names = param_names
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict)
        self.params_limit = p_property.params_limit
        self.param_fullNames = p_property.param_fullNames        
        self.params_space = self._limit_params_space(params_space)
        self._step = step
        self.params_lower = self.params_space[:,0]
        self.params_upper = self.params_space[:,1]
        self.params_n = len(params_space)
        self.N = N
        self.base_N = base_N
        self.auto_N = auto_N
        self._local_samples = local_samples
        self.prevStep_data = prevStep_data
        self.check_include = check_include
        self.rel_dev_limit = rel_dev_limit
        self.sigma_max = 20
        self.burn_in = False
    
    @property
    def step(self):
        if self._step>=1:
            return self._step
        else:
            raise ValueError('"step" must be an integer of >=1')
    
    @property
    def local_samples(self):
        if self._local_samples is None:
            return self._local_samples
        else:
            return utils.makeList(self._local_samples)
    
    def _limit_params_space(self, p_space):
        return su.CheckParameterSpace.check_limit(p_space, self.params_limit)
    
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
        selected_spectra = []
        for i in range(len(index_chosen)):
            selected_spectra.append(self.model.load_sample(local_sample, index_chosen[i]))
        selected_spectra = np.array(selected_spectra)
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
        p_filter = ParametersFilter(self.param_names, local_params, self.params_space, check_include=self.check_include, rel_dev_limit=self.rel_dev_limit)
        if self.check_include:
            if p_filter.include:
                selected_spectra, selected_params = self._filter_localSample(local_sample, N_local, local_params, p_filter)
            else:
#                print('The parameter space to be learned is not included in the parameter space of the local data sets')
                selected_spectra, selected_params = [], []
        else:
            selected_spectra, selected_params = self._filter_localSample(local_sample, N_local, local_params, p_filter)
        return selected_spectra, selected_params
    
    #to be improved, for the if... else..., other code like this should also be improved
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
        p_filter = ParametersFilter(self.param_names, pre_params, self.params_space, check_include=self.check_include, rel_dev_limit=self.rel_dev_limit)
        if self.check_include:
            if p_filter.include:
                selected_spectra, selected_params = self._filter_previousSamples(N_pre, pre_spectra, pre_params, p_filter)
            else:
                print('The parameter space to be learned is not included in the parameter space of the the previous step samples')
                selected_spectra, selected_params = [], []
        else:
            selected_spectra, selected_params = [], []
        return selected_spectra, selected_params
    
    def uniform_params(self, N):
        return np.random.rand(N, self.params_n) * (self.params_upper-self.params_lower) + self.params_lower
    
    def simulate_spectra(self, N):
        params = self.uniform_params(N)
        spectra = []
        for i in range(N):
            if (i+1)%100==0:
                print('Simulating the sample: {}/{}'.format(i+1,N))
            spectra.append(self.model.simulate(params[i])[1])
        spectra = np.array(spectra)
        return spectra, params
    
    def _auto_N(self):
        if self.step==1:
            self.N = self.base_N
        else:
            if not self.burn_in and self.sigma_max>10:
                self.N = self.base_N
    
    def simulate(self):
        #set 'self.N' automatically
        if self.auto_N:
            self._auto_N()
        
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
    
    def save_samples(self, path='sim_data/sample'):
        params = self.uniform_params(self.N)
        utils.savenpy(path, 'parameters', params, dtype=np.float32)
        self._save_paramsName(path)
        for i in range(self.N):
            if (i+1)%100==0:
                print('Simulating the samples: {}/{}'.format(i+1,self.N))
            x, y = self.model.simulate(params[i])
            utils.savenpy(path, 'y_%s'%i, y, dtype=np.float32)
        utils.savenpy(path, 'x', x, dtype=np.float32)
    
    def _save_paramsName(self, path):
        np.savetxt(path+"/paramNames.txt", self.param_fullNames, fmt='%s', delimiter='     ')


#%%
class MultiSpectra(Spectra):
    """Simulate training set containing multiple observations (for multi-branch network).
    
    Parameters
    ----------
    branch_n : int
        The number of branch of the network.
    model : cosmological (or theoretical) model instance
        A cosmological (or theoretical) model instance that is used to simulate training set,
        it should contains a 'simulate' method, and 'simulate' should accept input of cosmological parameters,
        if you use the local data sets, it should also contain 'load_params' and 'load_sample' methods.
    param_names : list
        A list that contains parameter names.
    params_space : array-like
        The parameter space with the shape of (n, 2), where n is the number of parameters. For each parameter, it is: [lower_limit, upper_limit].
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the base values, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    step : int, optional
        The number of step of the training process. Default: 1
    N : int, optional
        The number of data to be simulated. Default: 5000
    base_N : int, optional
        The basic or minimum number of data to be simulated, which works only when ``auto_N`` is set to True. Default: 2000
    auto_N : bool, optional
        If True, ``N`` is set automatically and the value of ``N`` will be ignored in some cases according to parameter space. Default: True
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
    """
    def __init__(self, branch_n, model, param_names, params_space, params_dict=None, step=1, N=5000, base_N=2000, auto_N=True,
                 local_samples=None, prevStep_data=None, check_include=True, rel_dev_limit=0.2):
        self.branch_n = branch_n
        self.model = model
        self.param_names = param_names
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict)
        self.params_limit = p_property.params_limit
        self.param_fullNames = p_property.param_fullNames
        self.params_space = self._limit_params_space(params_space)
        self._step = step
        self.params_lower = self.params_space[:,0]
        self.params_upper = self.params_space[:,1]
        self.params_n = len(params_space)
        self.N = N
        self.base_N = base_N
        self.auto_N = auto_N
        self._local_samples = local_samples
        self.prevStep_data = prevStep_data
        self.check_include = check_include
        self.rel_dev_limit = rel_dev_limit
        self.sigma_max = 20
        self.burn_in = False
    
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
        selected_spectra = []
        for comp in range(self.branch_n):
            _spectra = []
            for i in range(len(index_chosen)):
                _spectra.append(self.model.load_sample(local_sample, index_chosen[i])[comp])
            selected_spectra.append(np.array(_spectra))
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
        params = self.uniform_params(N)
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
    
    def save_samples(self, root_path='sim_data', branch_paths=['comp1','comp2']):
        params = self.uniform_params(self.N)
        utils.savenpy(root_path, 'parameters', params, dtype=np.float32)
        self._save_paramsName(root_path)
        for i in range(self.N):
            if (i+1)%100==0:
                print('Simulating the samples: {}/{}'.format(i+1,self.N))
            xx, yy = self.model.simulate(params[i])
            for j in range(self.branch_n):
                utils.savenpy(root_path+'/'+branch_paths[j], 'y_%s'%i, yy[j], dtype=np.float32)
        for j in range(len(branch_paths)):
            utils.savenpy(root_path+'/'+branch_paths[j], 'x', xx[j], dtype=np.float32)


#%% Add Gaussian noise
class AddGaussianNoise(object):
    """Add Gaussian noise for simulated data.
    
    Parameters
    ----------
    spectra : torch tensor, or a list of torch tensor
        The simulated spectra (data) with shape (N, spectra_length), or a list of spectra with shape [(N,spectra_length_1), (N,spectra_length_2), ...]
    obs_errors : torch tensor, or a list of torch tensor, optional
        Observational errors with shape (spectra_length,), or a list of errors with shape [(spectra_length_1,), (spectra_length_2,), ...]. Default: None
    cov_matrix : torch tensor, a list of torch tensor, or None, optional
        Covariance matrix with shape (spectra_length, spectra_length), or a list of covariance matrix with shape 
        [(spectra_length_1, spectra_length_1), (spectra_length_2, spectra_length_2), ...]. Default: None
    use_GPU : bool, optional
        If True, the noise will be generated by GPU, otherwise, it will be generated by CPU. Default: True
    """
    def __init__(self, spectra, obs_errors=None, cov_matrix=None, use_GPU=True):
        self.spectra = spectra
        self.is_list = isinstance(self.spectra, list)
        self.obs_errors = obs_errors
        self.cov_matrix = self._cov_matrix(cov_matrix)
        self.use_GPU = use_GPU
    
    def _cov_matrix(self, cov_matrix):
        if self.is_list and cov_matrix is None:
            return [None for i in range(len(self.spectra))]
        else:
            return cov_matrix
        
    def _singleSigmaNormal(self, spectrum, error, sigma):
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
        if self.use_GPU:
            err = torch.cuda.FloatTensor(spectrum.size()).normal_()*error*sigma
        else:
            err = torch.FloatTensor(spectrum.size()).normal_()*error*sigma
        return spectrum + err
    
    def _singleSigmaMultiVariableNormal(self, spectrum, cov_matrix, sigma):
        # Note 1: (i):   m=MultivariateNormal(torch.zeros(spectrum_leng), cov_torch)
        #         (ii):  m=MultivariateNormal(torch.zeros(spectrum_leng).cuda(), cov_cuda)
        #         (iii): m=MultivariateNormal(torch.cuda.FloatTensor(spectrum_leng).zero_(), cov_cuda)
        #         the speed of (i), (ii) and (iii) is (ii) > (i) > (iii)
        # Note 2: even though (ii) is faster, it is slower than (i) when using:
        #         batch_index = np.random.choice(len(self.inputs), self.batch_size, replace=False)
        #         xx = self.inputs[batch_index] # this step is slower, why?
        if self.use_GPU:
            #to be further improved
            m = MultivariateNormal(torch.zeros(spectrum.size(1)).cuda(), covariance_matrix=cov_matrix)
            err = m.sample((spectrum.size(0),)) * sigma
        else:
            m = MultivariateNormal(torch.zeros(spectrum.size(1)), covariance_matrix=cov_matrix)
            err = m.sample((spectrum.size(0),)) * sigma
        return spectrum + err
    
    def _multiSigmaNormal(self, spectrum, error, sigma):
#        print('the random strategy may be further updated !!!')
        if self.use_GPU:
            _sigmas = torch.cuda.FloatTensor(spectrum.size(1)).normal_(0, sigma) #original, this will be a little better !!!
#            _sigmas = torch.cuda.FloatTensor(spectrum.size(0), 1).normal_(0, sigma) #new, less good
#            _sigmas = torch.cuda.FloatTensor(spectrum.size()).normal_(0, sigma) #new_new, less good
            
#            _sigmas = torch.cuda.FloatTensor(spectrum.size(1)).uniform_(0, sigma) #test uniform, may be better?
            err = torch.cuda.FloatTensor(spectrum.size()).normal_()*error * torch.abs(_sigmas)
        else:
            _sigmas = torch.FloatTensor(spectrum.size(1)).normal_(0, sigma) #original, this will be a little better !!!
#            _sigmas = torch.FloatTensor(spectrum.size(0), 1).normal_(0, sigma) #new, less good
#            _sigmas = torch.FloatTensor(spectrum.size()).normal_(0, sigma) #new_new, less good
            err = torch.FloatTensor(spectrum.size()).normal_()*error * torch.abs(_sigmas)
        return spectrum + err
    
    def _multiSigmaMultiVariableNormal(self, spectrum, cov_matrix, sigma):
#        print('the random strategy may be further updated !!!')
        if self.use_GPU:
            #to be further improved
            m = MultivariateNormal(torch.zeros(spectrum.size(1)).cuda(), covariance_matrix=cov_matrix)
#            m = MultivariateNormal(torch.cuda.FloatTensor(spectrum.size(1)).zero_(), covariance_matrix=cov_matrix)# not used, will this faster?
#            _sigmas = torch.cuda.FloatTensor(spectrum.size(1)).normal_(0, sigma) #original, less good
            _sigmas = torch.cuda.FloatTensor(spectrum.size(0), 1).normal_(0, sigma) #new, this will be better !!!
#            _sigmas = torch.cuda.FloatTensor(spectrum.size()).normal_(0, sigma) #original_new, less good
            
#            _sigmas = torch.cuda.FloatTensor(spectrum.size(0), 1).uniform_(0, sigma) #test uniform, may be better?
            err = m.sample((spectrum.size(0),)) * torch.abs(_sigmas)
        else:
            m = MultivariateNormal(torch.zeros(spectrum.size(1)), covariance_matrix=cov_matrix)
#            _sigmas = torch.FloatTensor(spectrum.size(1)).normal_(0, sigma) #original, less good
            _sigmas = torch.FloatTensor(spectrum.size(0), 1).normal_(0, sigma) #new, this will be better !!!
#            _sigmas = torch.FloatTensor(spectrum.size()).normal_(0, sigma) #original_new, less good
            err = m.sample((spectrum.size(0),)) * torch.abs(_sigmas)
        return spectrum + err
    
    def singleSigma(self, sigma=1):
        if self.is_list:
            noisy_sp = []
            for i in range(len(self.spectra)):
                if self.cov_matrix[i] is None:
                    _noisy_sp = self._singleSigmaNormal(self.spectra[i], self.obs_errors[i], sigma)
                else:
                    _noisy_sp = self._singleSigmaMultiVariableNormal(self.spectra[i], self.cov_matrix[i], sigma)
                noisy_sp.append(_noisy_sp)
        else:
            if self.cov_matrix is None:
                noisy_sp = self._singleSigmaNormal(self.spectra, self.obs_errors, sigma)
            else:
                noisy_sp = self._singleSigmaMultiVariableNormal(self.spectra, self.cov_matrix, sigma)
        return noisy_sp
    
    def multiSigma(self, sigma=0.5):
        if self.is_list:
            noisy_sp = []
            for i in range(len(self.spectra)):
                if self.cov_matrix[i] is None:
                    _noisy_sp = self._multiSigmaNormal(self.spectra[i], self.obs_errors[i], sigma)
                else:
                    # _noisy_sp = self._multiSigmaMultiVariableNormal(self.spectra[i], self.cov_matrix[i], sigma)#!!! check if possible to use this
                    _noisy_sp = self._singleSigmaMultiVariableNormal(self.spectra[i], self.cov_matrix[i], 1)#!!!can we use "_multiSigmaMultiVariableNormal"?
                noisy_sp.append(_noisy_sp)
        else:
            if self.cov_matrix is None:
                noisy_sp = self._multiSigmaNormal(self.spectra, self.obs_errors, sigma)
            else:
                # noisy_sp = self._multiSigmaMultiVariableNormal(self.spectra, self.cov_matrix, sigma)#!!! check if possible to use this
                noisy_sp = self._singleSigmaMultiVariableNormal(self.spectra, self.cov_matrix, 1)#!!!can we use "_multiSigmaMultiVariableNormal"?
        return noisy_sp
    
    def noisySpectra(self, noise_type='multiSigma', sigma=0.5):
        if noise_type=='singleSigma':
            return self.singleSigma(sigma=sigma)
        elif noise_type=='multiSigma':
            return self.multiSigma(sigma=sigma)
        else:
            raise ValueError("noise_type must be 'singleSigma' or 'multiSigma'")


class AddMultiGaussianNoise(AddGaussianNoise):
    """Add multiple Gaussian noise for simulated data.
    
    Parameters
    ----------
    spectra : torch tensor, or a list of torch tensor
        The simulated spectra (data) with shape (N, spectra_length), or a list of spectra with shape [(N,spectra_length_1), (N,spectra_length_2), ...]
    params : torch tensor
        The simulated cosmological parameters.
    obs_errors : torch tensor, or a list of torch tensor, optional
        Observational errors with shape (spectra_length,), or a list of errors with shape [(spectra_length_1,), (spectra_length_2,), ...]. Default: None
    cov_matrix : torch tensor, a list of torch tensor, or None, optional
        Covariance matrix with shape (spectra_length, spectra_length), or a list of covariance matrix with shape 
        [(spectra_length_1, spectra_length_1), (spectra_length_2, spectra_length_2), ...]. Default: None
    multi_noise : int, optional
        The number of realization of noise added to a spectrum. Default: 5
    use_GPU : bool, optional
        If True, the noise will be generated by GPU, otherwise, it will be generated by CPU. Default: True
    """
    def __init__(self, spectra, params, obs_errors=None, cov_matrix=None, multi_noise=5, use_GPU=True):
        super(AddMultiGaussianNoise, self).__init__(spectra, obs_errors=obs_errors, cov_matrix=cov_matrix, use_GPU=use_GPU)
        self.params = params
        self.multi_noise = multi_noise
        self.is_list = isinstance(self.spectra, list)
    
    def multiParams(self):
        multi_P = []
        for i in range(self.multi_noise):
            multi_P.append(self.params)
        multi_P = torch.cat(multi_P, dim=0)
        return multi_P
    
    def multiNoisySpectra(self, noise_type, sigma):
        _multi_NS = []
        for i in range(self.multi_noise):
            _multi_NS.append(self.noisySpectra(noise_type=noise_type, sigma=sigma))
        
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
    
    def multiNoisySample(self, noise_type='multiSigma', sigma=0.5, reorder=True):
        multi_P = self.multiParams()
        multi_NS = self.multiNoisySpectra(noise_type, sigma)
        if reorder:
            index = np.random.choice(len(multi_P), len(multi_P), replace=False)
            multi_P = multi_P[index]
            if self.is_list:
                for comp in range(len(self.spectra)):
                    multi_NS[comp] = multi_NS[comp][index]
            else:
                multi_NS = multi_NS[index]
        return multi_NS, multi_P
