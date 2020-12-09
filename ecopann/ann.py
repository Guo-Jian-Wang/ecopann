# -*- coding: utf-8 -*-

from . import data_simulator as ds
from . import space_updater as su
from . import models, utils, cosmic_params, plotter, evaluate
from decimal import Decimal
import numpy as np
import torch
import time


#%% ANN
class ANN(plotter.Plotters):
    """Estimating (cosmological) parameters with Artificial Neural Network.
    
    Parameters
    ----------
    data : array-like or list
        The observational spectra (data) with shape (spectra_length,3), or a list of spectra with shape [(spectra_length_1,3), (spectra_length_2,3), ...].
        The first column is the observational variable, the second column is the best values of the measurement, 
        and the third column is the error of the measurement.
    cosmic_model : cosmological (or theoretical) model instance
        A cosmological (or theoretical) model instance that is used to simulate training set,
        it should contains a 'simulate' method, and 'simulate' should accept input of cosmological parameters,
        if you use the local data sets, it should also contain 'load_params' and 'load_sample' methods.
    param_names : list
        A list which contains the parameter names, e.g. ['H0','Omega_m','ombh2','omch2','tau','As','ns'].
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the base values, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    cov_matrix : array-like, list, or None, optional
        Covariance matrix of the observational data. It should be an array with shape (spectra_length, spectra_length), 
        or a list of covariance matrix with shape [(spectra_length_1, spectra_length_1), (spectra_length_2, spectra_length_2), ...].
        If there is no covariance for some observations, the covariance matrix should be set to None. e.g. [cov_matrix_1, None, cov_matrix_3]. Default: None
    init_params : array-like, optional
        The initial settings of the parameter space. Default: None
    hidden_layer : int, optional
        The number of the hidden layer of the network (for a single branch network). Default: 3
    branch_hiddenLayer : int, optional
        The number of the hidden layer for the branch part of the network (for a multibranch network). Default: 2
    trunk_hiddenLayer : int, optional
        The number of the hidden layer for the trunk part of the network (for a multibranch network). Default: 1
    epoch : int, optional
        The number of epoch of the training process. Default: 2000
    epoch_branch : int, optional
        The number of epoch of the training process (for the branch part of the multibranch network). Default: 2000
    num_train : int, optional
        The number of samples of the training set. Default: 3000
    local_samples : None, str, or list, optional
        Path of local samples, None, or 'sample' or ['sample'] or ['sample_1', 'sample_2', ...].
        If None, no local samples are used. Default: None
    steps_n : int, optional
        The number of steps of the training process. Default: 10
    
    Attributes
    ----------
    hp_model : str, optional
        Hyperparameter model that contains hyperparameters (such as activation function, batch normalization, dropout, etc.) used in the network.
        It can be 'eco_1', 'eco_2', 'eco_3', 'eco_4', 'eco_5', 'eco_6', or 'eco_7' (see :func:`~.hpmodel.models`). Default: 'eco_3'
    lr : float, optional
        The learning rate setting of the network. Default: 1e-2
    lr_min : float, optional
        The minimum of the learning rate. Default: 1e-8
    batch_size : int, optional
        The batch size setting of the network. Default: 750
    auto_batchSize : bool, optional
        If True, the batch size will be set automatically in the training process, otherwise, use the setting of ``batch_size``. Default: True
    auto_epoch : bool, optional
        If True, the epoch will be set automatically in the training process, otherwise, use the setting of ``epoch``. Default: True
    base_N_max : int, optional
        The maximum value of the basic (or minimum) number of data to be simulated, which works only when ``auto_N`` is set to True. Default: 1500
    auto_N : bool, optional
        If True, the number of samples in the training set will be set automatically, otherwise, use the setting of ``num_train``. Default: True
    noise_type : str, optional
        The type of noise to be added to the training set, 'multiSigma' or 'singleSigma'. Default: 'multiSigma'
    noise_sigma : float, optional
        For the case of 'singleSigma', it is the standard deviation of the Gaussian noise, while for the case of 'multiSigma' it is 
        a coefficient of the standard deviation. Default: 0.5
    multi_noise : int, optional
        The number of realization of noise added to a spectrum. Default: 5
    norm_inputs : bool, optional
        If True, the input data of the network will be normalized. Default: True
    norm_target : bool, optional
        If True, the target data (cosmological parameters) will be normalized. Default: True
    norm_type : str, optional
        The method of normalization, 'z_score', 'minmax', or 'mean' (see :class:`~.data_processor.Normalize`). Default: 'z_score'
    set_numpySeed : bool, optional
        If True, a fixed random seed that works for ``numpy`` will be set before training the network. Default: True
    set_torchSeed : bool, optional
        If True, a fixed random seed that works for ``PyTorch`` will be set before training the network. Default: True
    train_branch : bool, optional
        If True, the branch part of the multibranch network will be trained before training the entire network. Default: False
    match_init : bool, optional
        If True, the parameter space to be learned is fixed to the initial parameter space, 
        otherwise, it can jump out of the initial setting of parameter space. Default: False
    spaceSigma_min : int, optional
        The minimum parameter space to be learned, e.g. for spaceSigma_min=5, the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    chain_leng : int, optional
        The number of samples to be generated by a network model when predicting ANN chain, which is equal to the length of the ANN chain. Default: 10000
    
    Note
    ----
    The number of samples of the training set should be large enough to ensure the network learns a reliable mapping.
    For example, set num_train to 3000, or a larger value like 4000 or 5000.
    
    The epoch should also be set large enough to ensure a well-learned network.
    e.g. set epoch to 2000, or a larger value like 3000, 4000, or 5000.
    
    The initial parameter space is suggested to set large enough to cover the true parameters.
    In this case, it be easier for the network to find the best-fit value of parameters.
    
    It is better to set step number a large value like 10, and this will minimize the effect of randomness on the results.
    However, it is also acceptable to set a smaller value like 5, because burn-in will be reached quickly (usually no more than 2 steps).
    The advantage of this method is that we can analyze the results before the end of the training process, and determine how many steps 
    can be used to estimate parameters.
    
    Local samples can be used as training set to save time, so when using this method, you can generate a sample library for later reuse.    
    """
    def __init__(self, data, cosmic_model, param_names, params_dict=None, cov_matrix=None, init_params=None,  
                 hidden_layer=3, branch_hiddenLayer=2, trunk_hiddenLayer=1, epoch=2000, epoch_branch=2000,
                 num_train=3000, local_samples=None, steps_n=10):
        #observational data & cosmological model
        self.data = data
        self.cosmic_model = cosmic_model
        self.param_names = param_names
        self.params_dict = params_dict
        self.cov_matrix = cov_matrix
        self.init_params = self._init_params(init_params)
        #ANN model
        self.hidden_layer = hidden_layer
        self.hp_model = 'eco_3'
        self.branch_n = self._branch_n()
        self.branch_hiddenLayer = branch_hiddenLayer
        self.trunk_hiddenLayer = trunk_hiddenLayer
        self.lr = 1e-2
        self.lr_min = 1e-8
        self.batch_size = 750
        self.auto_batchSize = True
        self.epoch = epoch
        self.epoch_branch = epoch_branch
        self.auto_epoch = True
        #training set
        self.num_train = num_train
        self.base_N_max = 1500
        self.auto_N = True
        self.local_samples = local_samples
        #data preprocessing
        self.noise_type = 'multiSigma'
        self.noise_sigma = 0.5
        self.multi_noise = 5
        self.norm_inputs = True
        self.norm_target = True
        self.norm_type = 'z_score'
        #training
        self.set_numpySeed = True
        self.set_torchSeed = True
        self.train_branch = False
        #updating
        self.steps_n = steps_n
        self.match_init = False
        self.spaceSigma_min = 5
        self.chain_leng = 10000
    
    @property
    def obs_variables(self):
        if self.branch_n==1:
            return self.data[:,0]
        else:
            obs_varis = []
            for i in range(self.branch_n):
                obs_varis.append(self.data[i][:,0])
            return obs_varis
    
    @property
    def obs_dtype(self):
        if self.branch_n==1:
            return self.data.dtype
        else:
            return self.data[0].dtype
    
    @property
    def obs_errors(self):
        if self.branch_n==1:
            return self.data[:,2]
        else:
            obs_errs = []
            for i in range(self.branch_n):
                obs_errs.append(self.data[i][:,2])
            return obs_errs
    
    @property
    def base_N(self):
        bn = self.num_train//2
        if bn <= self.base_N_max:
            return bn
        else:
            return self.base_N_max
    
    @property
    def base_epoch(self):
        return self.epoch//2
    
    def _init_params(self, prior):
        if prior is None:
            return np.array([[-100, 100] for i in range(len(self.param_names))])
        else:
            return prior
    
    def _branch_n(self):
        if type(self.data) is list:
            return len(self.data)
        else:
            return 1
    
    def _randn_nums(self):
        self.randn_num = round(abs(np.random.randn()/5.), 5)
        if self.randn_num<1:
            self.randn_num = round(self.randn_num+1, 5)
        return [round(self.randn_num+i, 5) for i in range(self.steps_n)]
    
    def print_hparams(self):
        #print all useful hyper parameters
        #only print the first randn_num
        pass
    
    @property
    def cov_copy(self):
        if self.cov_matrix is None:
            return None
        else:
            return np.copy(self.cov_matrix)
    
    def save_variables(self, path='ann', sample=None):
        if sample is None:
            fileName = 'variables_%s'%self.randn_num
        else:
            fileName = 'variables-%s_%s'%(sample, self.randn_num)
        utils.savenpy(path+'/variables', fileName, self.obs_variables, dtype=self.obs_dtype)
    
    def train(self, path='ann', sample=None):
        """Train the network and save the results.
        
        Parameters
        ----------
        path : str, optional
            The path of the results to be saved. Default: 'ann'
        sample : str or None, optional
            Symbol mark of observational data or measurements. Default: None

        Returns
        -------
        list
            A list of chains.
        """
        self.path = path
        start_time = time.time()
        self.randn_nums = self._randn_nums()
        
        #logs & variables
        logName = 'log_%s'%(str(self.randn_nums[0]).split('.')[-1])
        utils.logger(path=self.path+'/logs', fileName=logName)
        self.save_variables(path=self.path, sample=sample)
        
        if self.set_numpySeed:
            np.random.seed(1000)
        if self.set_torchSeed:
            torch.manual_seed(1000)
        self.chain_all = []
        burn_in = False
        self.burnIn_step = None
        for step in range(1, self.steps_n+1):
            i = step-1
            #simulate data & update parameters space
            if step==1:
                print('\n'+'#'*25+' step {}/{} '.format(step, self.steps_n)+'#'*25)
                if self.branch_n==1:
                    simor = ds.Spectra(self.cosmic_model, self.param_names, self.init_params, params_dict=self.params_dict, step=step, N=self.num_train, base_N=self.base_N, auto_N=self.auto_N, local_samples=self.local_samples)
                else:
                    simor = ds.MultiSpectra(self.branch_n, self.cosmic_model, self.param_names, self.init_params, params_dict=self.params_dict, step=step, N=self.num_train, base_N=self.base_N, auto_N=self.auto_N, local_samples=self.local_samples)
                sim_y, sim_params = simor.simulate()
            else:
                if step==2:
                    chain_0 = None
                elif step>=3:
                    chain_0 = self.chain_all[-2]
                updater = su.UpdateParameterSpace(self.param_names, self.init_params, self.chain_all[-1], chain_0=chain_0, step=step, params_dict=self.params_dict, match_init=self.match_init)
                updater.spaceSigma_min = self.spaceSigma_min
                if updater.small_dev(limit_dev=0.001):
                    break
                #this is based on experiments
                if self.burnIn_step is None and max(updater.params_dev)<1 and max(updater.params_error_dev)<0.5:
                    burn_in = True
                    self.burnIn_step = step - 1 #let good chain contain burn-in step chain
                    print('\n\n'+'='*73)
                    if self.burnIn_step>=10:
                        print('*'*5+' '*24+'Burn-In step: %s'%(self.burnIn_step)+' '*23+'*'*5)
                    else:
                        print('*'*5+' '*24+'Burn-In step: %s'%(self.burnIn_step)+' '*24+'*'*5)
                    print('*'*5+' '*11+'The parameters have reached stable values'+' '*11+'*'*5)
                    print('*'*5+' '*1+'The chains of later steps can be used for parameter inference'+' '*1+'*'*5)
                    print('='*73+'\n')
        
                print('\n'+'#'*25+' step {}/{} '.format(step, self.steps_n)+'#'*25)
                params_space = updater.update()
                # check whether it has problems when using previous_data
                if self.branch_n==1:
                    simor = ds.Spectra(self.cosmic_model, self.param_names, params_space, params_dict=self.params_dict, step=step, N=self.num_train, base_N=self.base_N, auto_N=self.auto_N, local_samples=self.local_samples, prevStep_data=[sim_y, sim_params])
                else:
                    simor = ds.MultiSpectra(self.branch_n, self.cosmic_model, self.param_names, params_space, params_dict=self.params_dict, step=step, N=self.num_train, base_N=self.base_N, auto_N=self.auto_N, local_samples=self.local_samples, prevStep_data=[sim_y, sim_params])
                simor.sigma_max = updater.sigma_max
                simor.burn_in = burn_in
                sim_y, sim_params = simor.simulate()
                
            #training
            if self.branch_n==1:
                self.eco = models.MLP(sim_y, sim_params, self.param_names, params_dict=self.params_dict, obs_errors=self.obs_errors, cov_matrix=self.cov_copy,
                                      hidden_layer=self.hidden_layer, hp_model=self.hp_model, loss_func='L1',
                                      noise_type=self.noise_type, noise_sigma=self.noise_sigma)
            else:
                self.eco = models.MultiBranchNet(sim_y, sim_params, self.param_names, params_dict=self.params_dict, obs_errors=self.obs_errors, cov_matrix=self.cov_copy,
                                                 branch_hiddenLayer=self.branch_hiddenLayer, trunk_hiddenLayer=self.trunk_hiddenLayer, hp_model=self.hp_model, loss_func='L1',
                                                 noise_type=self.noise_type, noise_sigma=self.noise_sigma)
            self.eco.lr = self.lr
            self.eco.lr_min = self.lr_min
            self.eco.batch_size = self.batch_size
            self.eco.auto_batchSize = self.auto_batchSize
            self.eco.epoch = self.epoch
            self.eco.base_epoch = self.base_epoch
            self.eco.auto_epoch = self.auto_epoch
            if step==1:
                self.eco.print_info = True
            self.eco.multi_noise = self.multi_noise
            self.eco.norm_inputs = self.norm_inputs
            self.eco.norm_target = self.norm_target
            self.eco.norm_type = self.norm_type
            if step>=2:
                self.eco.spaceSigma_min = updater.sigma_min
            self.eco.auto_repeat_n = False
            self.eco.burn_in = burn_in
            self.eco.burnIn_step = self.burnIn_step
            self.eco.transfer_learning = False
            self.eco.randn_num = self.randn_nums[i]
            
            if self.branch_n==1:
                self.eco.train(repeat_n=3)
            else:
                self.eco.lr_branch = self.lr
                self.eco.epoch_branch = self.epoch_branch
                self.eco.train(repeat_n=3, train_branch=self.train_branch, parallel=False) ## reset parallel
            
            #predict chain
            #Note: here use self.cov_copy is to avoid data type error in "eco"
            chain_1 = self.eco.predict_chain(self.data, cov_matrix=self.cov_copy, chain_leng=self.chain_leng)
            self.chain_all.append(chain_1)
            
            #save results
            sample_i = '%s_step%s'%(sample, step) if sample is not None else None
            self.eco.save_net(path=self.path, sample=sample_i)
            self.eco.save_chain(path=self.path, sample=sample_i)            
            self.eco.save_hparams(path=self.path, sample=sample_i)
        print("\nTime elapsed for the training process: %.3f minutes"%((time.time()-start_time)/60))
        return self.chain_all
    
    def chains_good(self):
        if self.burnIn_step is None:
            raise ValueError('The number of steps is too small to find the Burn-In step and good chains!')
        else:
            return self.chain_all[self.burnIn_step:]
    
    @property
    def chain_ann(self):
        """Combined ANN chain using the result of steps after burn-in.
        """
        _chains_good = self.chains_good()
        chain_comb = _chains_good[0]
        if len(_chains_good)>1:
            for i in range(len(_chains_good)-1):
                chain_comb = np.r_[chain_comb, _chains_good[i+1]]
        return chain_comb

    @property
    def param_labels(self):
        return cosmic_params.ParamsProperty(self.param_names,params_dict=self.params_dict).labels


#%% repredict
class RePredict(plotter.Plotters):
    """Reanalysis using the saved chains or the well-trained networks.
    
    Parameters
    ----------
    data : array-like or list
        The observational spectra (data) with shape (spectra_length,3), or a list of spectra with shape [(spectra_length_1,3), (spectra_length_2,3), ...].
        The first column is the observational variable, the second column is the best values of the measurement, 
        and the third column is the error of the measurement.
    cov_matrix : array-like, list, or None, optional
        Covariance matrix of the observational data. It should be an array with shape (spectra_length, spectra_length), 
        or a list of covariance matrix with shape [(spectra_length_1, spectra_length_1), (spectra_length_2, spectra_length_2), ...].
        If there is no covariance for some observations, the covariance matrix should be set to None. e.g. [cov_matrix_1, None, cov_matrix_3]. Default: None
    path : str, optional
        The path of the results saved. Default: 'ann'
    randn_num : str or int, optional
        A random number that identifies the saved results. Default: ''
    steps_n : int, optional
        The number of steps of the training process. Default: 10
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the base values, the minimum values, 
        and the maximum values. See params_dict_zoo. Default: None
    
    Attributes
    ----------
    norm_inputs : bool, optional
        If True, the input data of the network will be normalized. Default: True
    norm_target : bool, optional
        If True, the target data (cosmological parameters) will be normalized. Default: True
    norm_type : str, optional
        The method of normalization, 'z_score', 'minmax', or 'mean' (see :class:`~.data_processor.Normalize`). Default: 'z_score'
    chain_leng : int, optional
        The number of samples to be generated by a network model when predicting ANN chain, which is equal to the length of the ANN chain. Default: 10000
    """
    def __init__(self, data, cov_matrix=None, path='ann', randn_num='', steps_n=10, params_dict=None):
        self.data = data
        self.cov_matrix = cov_matrix
        self.branch_n = self._branch_n()
        self.path = path
        self.randn_num = str(randn_num)
        self.randn_nums = [str(Decimal(str(randn_num)) + Decimal(str(i))) for i in range(steps_n)]
        self.steps_n = steps_n
        self.params_dict = params_dict
        self.norm_inputs = True
        self.norm_target = True
        self.norm_type = 'z_score'
        self.chain_leng = 10000
        if self.branch_n==1:
            self.eco = models.RePredictMLP(path=path, params_dict=params_dict)
        else:
            self.eco = models.RePredictMBNet(path=path, params_dict=params_dict)
    
    def _branch_n(self):
        if type(self.data) is list:
            return len(self.data)
        else:
            return 1
    
    @property
    def obs_variables(self):
        if self.branch_n==1:
            return self.data[:,0]
        else:
            obs_varis = []
            for i in range(self.branch_n):
                obs_varis.append(self.data[i][:,0])
            return obs_varis
    
    @property
    def trained_variables(self):
        file_path = evaluate.FilePath(filedir=self.path+'/variables', randn_num=self.randn_num, suffix='.npy').filePath()
        return np.load(file_path, allow_pickle=True)
    
    @property
    def same_variables(self):
        if self.branch_n==1:
            return np.all(self.obs_variables==self.trained_variables)
        else:
            same_varis = [np.all(self.obs_variables[i]==self.trained_variables[i]) for i in range(self.branch_n)]
            return np.all(same_varis)

    @property
    def cov_copy(self):
        if self.cov_matrix is None:
            return None
        else:
            return np.copy(self.cov_matrix)
    
    def from_chain(self):
        """Predict using saved chains.
        
        Raises
        ------
        ValueError
            If variables of the input observational data are different from those used to train the network, an error will be raised.
        """
        if not self.same_variables:
            raise ValueError('Variables of the input observational data are different from those used to train the network!')
        self.chain_all = []
        self.chains_good = []
        for i in range(self.steps_n):
            self.eco.randn_num = self.randn_nums[i]
            self.eco.load_chain()
            self.eco.load_hparams()
            self.chain_all.append(self.eco.chain)
            if self.eco.burnIn_step is not None:
                self.chains_good.append(self.eco.chain)
    
    def from_net(self):
        """Predict using saved networks.
        
        Raises
        ------
        ValueError
            If variables of the input observational data are different from those used to train the network, an error will be raised.
        """
        if not self.same_variables:
            raise ValueError('Variables of the input observational data are different from those used to train the network!')
        self.chain_all = []
        self.chains_good = []
        torch.manual_seed(1000)#
        for i in range(self.steps_n):
            self.eco.randn_num = self.randn_nums[i]
            self.eco.load_net()
            self.eco.load_hparams()
            self.eco.predict_chain(self.data, cov_matrix=self.cov_copy, chain_leng=self.chain_leng)
            self.chain_all.append(self.eco.chain)
            if self.eco.burnIn_step is not None:
                self.chains_good.append(self.eco.chain)
    
    @property
    def chain_ann(self):
        """Combined ANN chain using the result of steps after burn-in.
        """
        if len(self.chains_good)==0:
            raise ValueError('The number of steps is too small to find the Burn-In step and good chains!')
        chain_comb = self.chains_good[0]
        if len(self.chains_good)>1:
            for i in range(len(self.chains_good)-1):
                chain_comb = np.r_[chain_comb, self.chains_good[i+1]]
        return chain_comb
    
    @property
    def best_fit(self):
        return su.Chains.bestFit(self.chain_ann, symmetry_error=False)
    
    @property
    def param_labels(self):
        return cosmic_params.ParamsProperty(self.eco.param_names,params_dict=self.params_dict).labels
    
    @property
    def burnIn_step(self):
        return self.eco.burnIn_step


#%% try BNN(Bayesian Neural Networks) & RNN
