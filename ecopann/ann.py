# -*- coding: utf-8 -*-

from . import data_simulator as ds
from . import space_updater as su
from . import models, utils, cosmic_params, plotter, evaluate
from decimal import Decimal
import numpy as np
import torch
import time


#%% ANN
class ANN(plotter.PlotPosterior):
    """Estimating (cosmological) parameters with Artificial Neural Network.
    
    Parameters
    ----------
    obs_data : array-like or list
        The observational spectra (data) with shape (spectra_length,3), or a list of spectra with shape [(spectra_length_1,3), (spectra_length_2,3), ...].
        The first column is the observational variable, the second column is the best values of the measurement, 
        and the third column is the error of the measurement.
    model : cosmological (or theoretical) model instance
        A cosmological (or theoretical) model instance that is used to simulate training set,
        it should contains a 'simulate' method, and 'simulate' should accept input of cosmological parameters,
        if you use local data sets, it should also contain 'load_params' and 'load_sample' methods.
    param_names : list
        A list which contains the parameter names, e.g. ['H0','Omega_m','ombh2','omch2','tau','As','ns'].
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    cov_matrix : array-like, list, or None, optional
        Covariance matrix of the observational data. It should be an array with shape (spectra_length, spectra_length), 
        or a list of covariance matrix with shape [(spectra_length_1, spectra_length_1), (spectra_length_2, spectra_length_2), ...].
        If there is no covariance for some observations, the covariance matrix should be set to None. e.g. [cov_matrix_1, None, cov_matrix_3]. Default: None
    init_chain : array-like, optional
        The initial ANN or MCMC chain, which is usually based on prvious parameter estimation. Default: None
    init_params : array-like, optional
        The initial settings of the parameter space. If ``init_chain`` is given, ``init_params`` will be ignored. Default: None
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
    num_vali : int, optional
        The number of samples of the validation set. Default: 500
    spaceSigma : int or array-like, optional
        The size of parameter space to be learned. It is a int or a numpy array with shape of (n,), where n is the number of parameters, 
        e.g. for spaceSigma=5, the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    space_type : str, optional
        The type of parameter space. It can be 'hypercube', 'LHS', 'hypersphere', 'hyperellipsoid', or 'posterior_hyperellipsoid'. Default: 'hyperellipsoid'
    local_samples : None, str, or list, optional
        Path of local samples, None, or 'sample' or ['sample'] or ['sample_1', 'sample_2', ...].
        If None, no local samples are used. Default: None
    stepStop_n : int, optional
        If the number of steps after burn-in reached stepStop_n, it will stop the whole training process.
        This only works after burn_in. Default: 3
    
    Attributes
    ----------
    activation_func : str, optional
        Activation function, which can be 'relu', 'leakyrelu', 'prelu', 'rrelu', 'relu6', 'elu', 'celu', 'selu', 'silu', 
        'sigmoid', 'logsigmoid', 'tanh', 'tanhshrink', 'softsign', or 'softplus' (see :func:`~.element.activation`). Default: 'rrelu'
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
        The type of Gaussian noise added to the training set, 'singleNormal' or 'multiNormal'. Default: 'multiNormal'
    factor_sigma : float, optional
        For the case of 'singleNormal', it is the factor of the observational error (standard deviation), 
        while for the case of 'multiNormal' it is the standard deviation of the coefficient of the observational error (standard deviation). Default: 0.2
    multi_noise : int, optional
        The number of realization of noise added to a spectrum. Default: 5
    scale_spectra : bool, optional
        If True, the input data (measurements) will be scaled based on the base values of the data. It is recommended to set to True. Default: True
    scale_params : bool, optional
        If True, the target data (cosmological parameters) will be scaled based on the base values of parameters. 
        See :class:`~.data_processor.ParamsScaling`. It is recommended to set to True. Default: True
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
    repeat_n : int, optional
        The number of iterations using the same batch of data during network training, which is usually set to 1 or 3. Default: 3
    expectedBurnIn_step : int, optional
        The expected burn-in step number. Default: 10
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
    def __init__(self, obs_data, model, param_names, params_dict=None, cov_matrix=None, init_chain=None, init_params=None,
                 hidden_layer=3, branch_hiddenLayer=2, trunk_hiddenLayer=1, epoch=2000, epoch_branch=2000,
                 num_train=3000, num_vali=500, spaceSigma=5, space_type='hyperellipsoid', local_samples=None, stepStop_n=3):
        #observational data & cosmological model
        self.obs_data = obs_data
        self.model = model
        self.param_names = param_names
        self.params_dict = params_dict
        self.cov_matrix = cov_matrix
        self.init_chain = init_chain
        self.init_params = self._init_params(init_params)
        #ANN model
        self.hidden_layer = hidden_layer
        self.activation_func = 'rrelu'
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
        self.num_vali = num_vali
        self.spaceSigma = spaceSigma
        self.space_type = space_type
        self.base_N_max = 1500
        self.auto_N = True
        self.local_samples = local_samples
        #data preprocessing
        self.noise_type = 'multiNormal'
        self.factor_sigma = 0.2
        self.multi_noise = 5
        self.scale_spectra = True
        self.scale_params = True
        self.norm_inputs = True
        self.norm_target = True
        self.independent_norm = False
        self.norm_type = 'z_score'
        #training
        self.set_numpySeed = False #remove?
        self.set_torchSeed = False #remove?
        self.train_branch = False
        self.repeat_n = 3
        #updating
        self.stepStop_n = stepStop_n
        self.expectedBurnIn_step = 10
        self.chain_leng = 10000
        
    @property
    def obs_variables(self):
        if self.branch_n==1:
            return self.obs_data[:,0]
        else:
            obs_varis = []
            for i in range(self.branch_n):
                obs_varis.append(self.obs_data[i][:,0])
            return obs_varis
    
    @property
    def obs_dtype(self):
        if self.branch_n==1:
            return self.obs_data.dtype
        else:
            return self.obs_data[0].dtype
    
    @property
    def obs_errors(self):
        if self.branch_n==1:
            return self.obs_data[:,2]
        else:
            obs_errs = []
            for i in range(self.branch_n):
                obs_errs.append(self.obs_data[i][:,2])
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
            prior = np.array([[-100, 100] for i in range(len(self.param_names))])
        params_limit = cosmic_params.ParamsProperty(self.param_names, params_dict=self.params_dict).params_limit
        return su.CheckParameterSpace.check_limit(prior, params_limit)
    
    def _branch_n(self):
        if type(self.obs_data) is list:
            return len(self.obs_data)
        else:
            return 1
    
    def _randn_nums(self):
        self.randn_num = round(abs(np.random.randn()/5.), 5)
        if self.randn_num<1:
            self.randn_num = round(self.randn_num+1, 5)
        return [round(self.randn_num+i, 5) for i in range(self.expectedBurnIn_step+self.stepStop_n)]
    
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
    
    def save_variables(self, sample=None):
        if sample is None:
            fileName = 'variables_%s'%self.randn_num
        else:
            fileName = 'variables-%s_%s'%(sample, self.randn_num)
        utils.savenpy(self.path+'/variables', fileName, self.obs_variables, dtype=self.obs_dtype)

    def simulate(self, step=1, burn_in=False, burnIn_step=None, space_type_all=[], prev_space=None,
                 chain_all=[], sim_spectra=None, sim_params=None):
        """Simulate data and update parameter space.
        """
        if step==1:
            # set training number
            training_n = self.base_N
            # training_n = 3000 #test, because using hypercube
            # set space_type
            if self.init_chain is None:
                if self.space_type=='hypersphere' or self.space_type=='hyperellipsoid' or self.space_type=='posterior_hyperellipsoid':
                    s_type = 'hypercube'
                    # s_type = 'LHS' #test
                    # self.factor_sigma_i = 0.4 #test, 0.5, 0.4, 0.35
                    self.factor_sigma_i = self.factor_sigma
                else:
                    s_type = self.space_type
                    self.factor_sigma_i = self.factor_sigma
            else:
                s_type = self.space_type
                self.factor_sigma_i = self.factor_sigma
            space_type_all.append(s_type)
            print('\n'+'#'*25+' step {} '.format(step)+'#'*25)
            if self.branch_n==1:
                simor = ds.SimSpectra(training_n, self.model, self.param_names, chain=self.init_chain, params_space=self.init_params, 
                                      spaceSigma=self.spaceSigma, params_dict=self.params_dict, space_type=s_type, 
                                      cut_crossedLimit=True, local_samples=self.local_samples, prevStep_data=None)
            else:
                simor = ds.SimMultiSpectra(self.branch_n, training_n, self.model, self.param_names, chain=self.init_chain, params_space=self.init_params, 
                                           spaceSigma=self.spaceSigma, params_dict=self.params_dict, space_type=s_type, 
                                           cut_crossedLimit=True, local_samples=self.local_samples, prevStep_data=None)
            sim_spectra, sim_params = simor.simulate()
            prev_space = simor.params_space #used for next step
        else:
            if step==2:
                chain_0 = self.init_chain
            elif step>=3:
                chain_0 = chain_all[-2]
            updater = su.UpdateParameterSpace(step,self.param_names,chain_all[-1],chain_0=chain_0,init_params=self.init_params,spaceSigma=self.spaceSigma,params_dict=self.params_dict)
            if updater.small_dev(limit_dev=0.001):
                #to be improved to get chain_ann after exit()???, or remove these two lines???
                exit()
            #this is based on experiments, update this??? eg. max(updater.param_devs)<0.5?
            # if burnIn_step is None and max(updater.param_devs)<1 and max(updater.error_devs)<0.5:
            # if burnIn_step is None and max(updater.param_devs)<0.5 and max(updater.error_devs)<0.5: #test!!!
            if burnIn_step is None and max(updater.param_devs)<=0.25 and max(updater.error_devs)<=0.25: #test!!!
                burn_in = True
                burnIn_step = step - 1 #let good chain contain burn-in step chain, should be step-2? the good chain will not contain burn-in step chain!
                print('\n\n'+'='*73)
                if burnIn_step>=10:
                    print('*'*5+' '*24+'Burn-In step: %s'%(burnIn_step)+' '*23+'*'*5)
                else:
                    print('*'*5+' '*24+'Burn-In step: %s'%(burnIn_step)+' '*24+'*'*5)
                print('*'*5+' '*11+'The parameters have reached stable values'+' '*11+'*'*5)
                print('*'*5+' '*1+'The chains of later steps can be used for parameter inference'+' '*1+'*'*5)
                print('='*73+'\n')
            
            if burn_in:
                print('\n'+'#'*25+' step {}/{} '.format(step, burnIn_step+self.stepStop_n)+'#'*25)
            else:
                print('\n'+'#'*25+' step {} '.format(step)+'#'*25)
            self.spaceSigma_min = updater.spaceSigma_min
            updater.print_learningRange()
            
            # set training number, should this (spaceSigma_max>=10) be updated??? 
            # if not burn_in and updater.spaceSigma_max>10:
            #     training_n = self.base_N
            # else:
            #     training_n = self.num_train
            #test
            if burn_in:
                training_n = self.num_train + self.num_vali
            else:
                if max(updater.param_devs)<=0.5 and max(updater.error_devs)<=0.25:
                    training_n = self.num_train
                else:
                    training_n = self.base_N
                    
            # set space_type
            if burn_in:
                s_type = self.space_type
                self.factor_sigma_i = self.factor_sigma
            else:
                # s_type = 'hypercube'
                s_type = self.space_type #test!!!
                self.factor_sigma_i = self.factor_sigma
                # # # s_type = 'LHS' #test !!!
                # # if max(updater.param_devs)<1 and max(updater.error_devs)<0.5: #test
                # if max(updater.param_devs)<0.5 and max(updater.error_devs)<0.5: #test
                #     s_type = self.space_type #test!!!
                # else:
                #     s_type = 'hypercube' #test
                    
            space_type_all.append(s_type)
            if space_type_all[-1]==space_type_all[-2]:
                prevStep_data = [sim_spectra, sim_params]
            else:
                prevStep_data = None
            # #test
            # if burn_in:
            #     prevStep_data = None ##test!!!
            # else:
            #     prevStep_data = [sim_spectra, sim_params] #test !!!
            
            rel_dev_limit = 0.1 #0.1 #test
            
            cut_crossedLimit = True
            # cut_crossedLimit = False #test !!!
            
            
            # check whether it has problems when using previous_data???
            if self.branch_n==1:
                simor = ds.SimSpectra(training_n, self.model, self.param_names, chain=chain_all[-1], params_space=None, spaceSigma=updater.spaceSigma_all,
                                      params_dict=self.params_dict, space_type=s_type, cut_crossedLimit=cut_crossedLimit, local_samples=None, prevStep_data=prevStep_data, rel_dev_limit=rel_dev_limit) #reset local_samples & previous_data???
            else:
                simor = ds.SimMultiSpectra(self.branch_n, training_n, self.model, self.param_names, chain=chain_all[-1], params_space=None, spaceSigma=updater.spaceSigma_all,
                                           params_dict=self.params_dict, space_type=s_type, cut_crossedLimit=cut_crossedLimit, local_samples=None, prevStep_data=prevStep_data, rel_dev_limit=rel_dev_limit) #reset local_samples & previous_data???
            simor.prev_space = prev_space
            sim_spectra, sim_params = simor.simulate()
            prev_space = simor.params_space #used for next step
        
        
        #test, to be added to the code???
        # good_index = np.where(~np.isnan(sim_spectra[:,0])) #test
        # sim_spectra = sim_spectra[good_index] #test
        # sim_params = sim_params[good_index] #test
        return sim_spectra, sim_params, burn_in, burnIn_step, space_type_all, prev_space
    
    def _train(self, sim_spectra, sim_params, step=1, burn_in=False, burnIn_step=None, 
               randn_num=0.123, sample=None, save_items=True, 
               showIter_n=100):
        if burn_in:
            idx = np.random.choice(self.num_train+self.num_vali, self.num_train+self.num_vali, replace=False)
            if self.branch_n==1:
                train_set = [sim_spectra[idx[:self.num_train]], sim_params[idx[:self.num_train]]]
                vali_set = [sim_spectra[idx[self.num_train:]], sim_params[idx[self.num_train:]]]
            else:
                sim_spectra_train = [sim_spectra[i][idx[:self.num_train]] for i in range(self.branch_n)]
                sim_params_train = sim_params[idx[:self.num_train]]
                sim_spectra_vali = [sim_spectra[i][idx[self.num_train:]] for i in range(self.branch_n)]
                sim_params_vali = sim_params[idx[self.num_train:]]
                train_set = [sim_spectra_train, sim_params_train]
                vali_set = [sim_spectra_vali, sim_params_vali]
        else:
            train_set = [sim_spectra, sim_params]
            vali_set = [None, None]
        if self.branch_n==1:
            self.eco = models.OneBranchMLP(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, cov_matrix=self.cov_copy, params_dict=self.params_dict,
                                           hidden_layer=self.hidden_layer, activation_func=self.activation_func, loss_func='L1',
                                           noise_type=self.noise_type, factor_sigma=self.factor_sigma_i, multi_noise=self.multi_noise)
        else:
            self.eco = models.MultiBranchMLP(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, cov_matrix=self.cov_copy, params_dict=self.params_dict,
                                             branch_hiddenLayer=self.branch_hiddenLayer, trunk_hiddenLayer=self.trunk_hiddenLayer, activation_func=self.activation_func, loss_func='L1',
                                             noise_type=self.noise_type, factor_sigma=self.factor_sigma_i, multi_noise=self.multi_noise)
        self.eco.lr = self.lr
        self.eco.lr_min = self.lr_min
        self.eco.batch_size = self.batch_size
        self.eco.auto_batchSize = self.auto_batchSize
        self.eco.epoch = self.epoch
        self.eco.base_epoch = self.base_epoch
        self.eco.auto_epoch = self.auto_epoch
        if step==1:
            self.eco.print_info = True
        self.eco.scale_spectra = self.scale_spectra
        self.eco.scale_params = self.scale_params
        self.eco.norm_inputs = self.norm_inputs
        self.eco.norm_target = self.norm_target
        self.eco.independent_norm = self.independent_norm
        self.eco.norm_type = self.norm_type
        if step>=2:
            self.eco.spaceSigma_min = self.spaceSigma_min
        self.eco.auto_repeat_n = False
        self.eco.burn_in = burn_in
        self.eco.burnIn_step = burnIn_step
        self.eco.transfer_learning = False
        self.eco.randn_num = randn_num
        
        if self.branch_n==1:
            self.eco.train(repeat_n=self.repeat_n, showIter_n=showIter_n)
        else:
            self.eco.lr_branch = self.lr
            self.eco.epoch_branch = self.epoch_branch
            self.eco.train(repeat_n=self.repeat_n, train_branch=self.train_branch, parallel=False, showIter_n=showIter_n) #reset parallel???
        
        #predict chain
        #Note: here use self.cov_copy is to avoid data type error in "eco"
        chain_1 = self.eco.predict_chain(self.obs_data, cov_matrix=self.cov_copy, chain_leng=self.chain_leng)
        # chain_1 = self.eco.chain_ann #updated, remove?
        
        #save results
        if save_items:
            sample_i = '%s_step%s'%(sample, step) if sample is not None else None
            self.eco.save_net(path=self.path, sample=sample_i)
            self.eco.save_loss(path=self.path, sample=sample_i)
            self.eco.save_chain(path=self.path, sample=sample_i)            
            self.eco.save_hparams(path=self.path, sample=sample_i)
        return chain_1
    
    # to be updated, add stop_step_n ???
    def train(self, path='ann', sample=None, save_items=True, showIter_n=100):
        """Train the network and save the results.
        
        Parameters
        ----------
        path : str, optional
            The path of the results to be saved. Default: 'ann'
        sample : str or None, optional
            Symbol mark of observational data or measurements. Default: None
        save_items : bool, optional
            If True, results will be saved to disk, otherwise, results will not be saved
        showIter_n : int, optional
            The number of iterations interval for printing. Default: 100
        
        Returns
        -------
        list
            A list of chains.
        """
        self.path = path
        #should the parameter sample above be changed ???
        randn_nums = self._randn_nums()
        
        #logs & variables
        if save_items:
            # logName = 'log_%s'%(str(randn_nums[0]).split('.')[-1])
            logName = 'log_%s'%(randn_nums[0])
            utils.logger(path=self.path+'/logs', fileName=logName)
            self.save_variables(sample=sample)
            print('randn_num: %s'%randn_nums[0])
        
        if self.set_numpySeed:
            np.random.seed(1000)
        if self.set_torchSeed:
            torch.manual_seed(1000)
        self.chain_all = []
        burn_in = False
        self.burnIn_step = None
        space_type_all = []
        self.sim_spectra, self.sim_params, prev_space = None, None, None
        start_time = time.time()
        for step in range(1, self.expectedBurnIn_step+self.stepStop_n+1):
            #simulate data & update parameter space
            self.sim_spectra, self.sim_params, burn_in, self.burnIn_step, space_type_all, prev_space = self.simulate(step=step, burn_in=burn_in, burnIn_step=self.burnIn_step, 
                                                                                                                     space_type_all=space_type_all, prev_space=prev_space, 
                                                                                                                     chain_all=self.chain_all, sim_spectra=self.sim_spectra, sim_params=self.sim_params)
            
            #training
            chain_1 = self._train(self.sim_spectra, self.sim_params, step=step, burn_in=burn_in, burnIn_step=self.burnIn_step, 
                                  randn_num=randn_nums[step-1], sample=sample, save_items=save_items, 
                                  showIter_n=showIter_n)
            self.chain_all.append(chain_1)
            
            #test
            if burn_in and step-self.burnIn_step==self.stepStop_n:
                break
                
        print("\nTime elapsed for the training process: %.3f minutes"%((time.time()-start_time)/60))
        return self.chain_all

    @property
    def chains_good(self):
        if self.burnIn_step is None:
            raise ValueError('The number of steps is too small to find the Burn-In step and good chains!')
        else:
            return self.chain_all[self.burnIn_step:]
    
    @property
    def chain_ann(self):
        """Combined ANN chain using the result of steps after burn-in.
        """
        return np.concatenate(self.chains_good, axis=0)

    @property
    def param_labels(self):
        return cosmic_params.ParamsProperty(self.param_names,params_dict=self.params_dict).labels


#%% repredict
class RePredict(plotter.PlotPosterior):
    """Reanalysis using the saved chains or the well-trained networks.
    
    Parameters
    ----------
    obs_data : array-like or list
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
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See params_dict_zoo. Default: None
    
    Attributes
    ----------
    chain_leng : int, optional
        The number of samples to be generated by a network model when predicting ANN chain, which is equal to the length of the ANN chain. Default: 10000
    """
    def __init__(self, obs_data, cov_matrix=None, path='ann', randn_num='', steps_n=10, params_dict=None):
        self.obs_data = obs_data
        self.cov_matrix = cov_matrix
        self.branch_n = self._branch_n()
        self.path = path
        self.randn_num = str(randn_num)
        self.randn_nums = [str(Decimal(str(randn_num)) + Decimal(str(i))) for i in range(steps_n)]
        self.steps_n = steps_n
        self.params_dict = params_dict
        self.chain_leng = 10000
        if self.branch_n==1:
            self.eco = models.RePredictOBMLP(path=path)
        else:
            self.eco = models.RePredictMBMLP(path=path)
    
    def _branch_n(self):
        if type(self.obs_data) is list:
            return len(self.obs_data)
        else:
            return 1
    
    @property
    def obs_variables(self):
        if self.branch_n==1:
            return self.obs_data[:,0]
        else:
            obs_varis = []
            for i in range(self.branch_n):
                obs_varis.append(self.obs_data[i][:,0])
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
            self.eco.load_loss()
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
            self.eco.load_loss()
            self.eco.predict_chain(self.obs_data, cov_matrix=self.cov_copy, chain_leng=self.chain_leng)
            self.chain_all.append(self.eco.chain)
            if self.eco.burnIn_step is not None:
                self.chains_good.append(self.eco.chain)
    
    @property
    def chain_ann(self):
        """Combined ANN chain using the result of steps after burn-in.
        """
        if len(self.chains_good)==0:
            raise ValueError('The number of steps is too small to find the Burn-In step and good chains!')
        return np.concatenate(self.chains_good, axis=0)
    
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
