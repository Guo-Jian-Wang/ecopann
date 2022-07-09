# -*- coding: utf-8 -*-
import coplot.plot_settings as pls

from . import data_processor as dp
from . import data_simulator as ds
from . import space_updater as su
from . import train, evaluate, optimize, fcnet, nodeframe, utils, cosmic_params

import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt


#%% multilayer perceptron (MLP)
class OneBranchMLP(train.Train, dp.ParamsScaling, ds.CutParams):
    """Multilayer perceptron (MLP) that is used to predict cosmological parameters with one set of datasets.
        
    Parameters
    ----------
    train_set : list
        The training set that contains simulated observational spectra (data) with shape (N, spectra_length) 
        and simulated parameters of a specific cosmological (or theoretical) model. i.e. [spectra, parameters]
    param_names : list
        A list which contains the parameter names, e.g. ['H0','Omega_m','ombh2','omch2','tau','As','ns'].
    vali_set : list, optional
        The validation set that contains simulated observational spectra (data) with shape (N, spectra_length) 
        and simulated parameters of a specific cosmological (or theoretical) model. The validation set can also be set to None.
        i.e. [spectra, parameters] or [None, None]
    obs_errors : array-like, optional
        Observational errors with shape (spectra_length,). Default: None
    cov_matrix : array-like or None, optional
        Covariance matrix of the observational data. Default: None
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    hidden_layer : int, optional
        The number of the hidden layer of the network. Default: 3
    activation_func : str, optional
        Activation function, which can be 'relu', 'leakyrelu', 'prelu', 'rrelu', 'relu6', 'elu', 'celu', 'selu', 'silu', 
        'sigmoid', 'logsigmoid', 'tanh', 'tanhshrink', 'softsign', or 'softplus' (see :func:`~.element.activation`). Default: 'rrelu'
    loss_func : str, optional
        The loss function used in the network. Default: 'L1'
    noise_type : str, optional
        The type of Gaussian noise added to the training set, 'singleNormal' or 'multiNormal'. Default: 'multiNormal'
    factor_sigma : float, optional
        For the case of 'singleNormal', it is the factor of the observational error (standard deviation), 
        while for the case of 'multiNormal' it is the standard deviation of the coefficient of the observational error (standard deviation). Default: 0.5
    multi_noise : int, optional
        The number of realization of noise added to a spectrum. Default: 5
    
    Attributes
    ----------
    spectra_base : array-like, optional
        The base value of spectra that is used for data normalization when training the network to ensure that the scaled spectra are ~ 1., 
        it is suggested to set the mean of the simulated spectra. The default is the mean of the simulated spectra.
    params_base : array-like, optional
        The base value of parameters that is used for data normalization when training the network to ensure that the scaled parameters are ~ 1., 
        it is suggested to set the mean of the posterior distribution (or the simulated parameters). The default is the mean of the simulated parameters.
    lr : float, optional
        The learning rate setting of the network. Default: 1e-2
    lr_min : float, optional
        The minimum of the learning rate. Default: 1e-8
    batch_size : int, optional
        The batch size setting of the network. Default: 750
    auto_batchSize : bool, optional
        If True, the batch size will be set automatically in the training process, otherwise, use the setting of ``batch_size``. Default: True
    epoch : int, optional
        The number of epoch of the training process. Default: 2000
    base_epoch : int, optional
        The base number (or the minimum number) of epoch. Default: 1000
    auto_epoch : bool, optional
        If True, the epoch will be set automatically in the training process, otherwise, use the setting of ``epoch``. Default: True
    scale_spectra : bool, optional
        If True, the input data (measurements) will be scaled based on the base values of the data. It is recommended to set to True. Default: True
    scale_params : bool, optional
        If True, the target data (cosmological parameters) will be scaled based on the base values of parameters. 
        See :class:`~.data_processor.ParamsScaling`. It is recommended to set to True. Default: True
    norm_inputs : bool, optional
        If True, the input data of the network will be normalized. Default: True
    norm_target : bool, optional
        If True, the target data (cosmological parameters) will be normalized. Default: True
    independent_norm : bool, optional
        If True, the target data (cosmological parameters) will be normalized independently. This only works when ``norm_target`` is True. 
        It is recommended to set to False. Default: False
    norm_type : str, optional
        The method of normalization, 'z_score', 'minmax', or 'mean' (see :class:`~.data_processor.Normalize`). Default: 'z_score'
    spaceSigma_min : int, optional
        The minimum parameter space to be learned, e.g. for spaceSigma_min=5, the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    transfer_learning : bool, optional
        If True, the network will be initialized using the well-trained network of the previous step. Default: False
    """
    
    def __init__(self, train_set, param_names, vali_set=[None,None], obs_errors=None, cov_matrix=None, params_dict=None,
                 hidden_layer=3, activation_func='rrelu', loss_func='L1',
                 noise_type='multiNormal', factor_sigma=0.5, multi_noise=5):
        #data
        self.spectra, self.params = train_set
        self.spectra_base = np.mean(self.spectra, axis=0)
        self.params_base = np.mean(self.params, axis=0)
        self.param_names = param_names
        self.params_n = len(param_names)
        self.spectra_vali, self.params_vali = vali_set
        # self.obs_data = obs_data #updated, update for multi-branch network !!!, remove?
        # self.obs_errors = obs_data[:,2], remove?
        self.obs_errors = obs_errors
        self.cov_matrix = cov_matrix
        self.cholesky_factor = self._cholesky_factor(cov_matrix)
        self.params_dict = params_dict
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict)
        self.params_limit = p_property.params_limit
        #ANN model
        self.hidden_layer = hidden_layer
        self.activation_func = activation_func
        self.loss_func = train.loss_funcs(name=loss_func) #train.Train should be updated
        self.lr = 1e-2
        self.lr_min = 1e-8
        self.batch_size = 750
        self.auto_batchSize = True
        self.epoch = 2000
        self.base_epoch = 1000
        self.auto_epoch = True
        self.fix_initialize = False
        self.print_info = False
        #data preprocessing
        self.noise_type = noise_type
        self.factor_sigma = factor_sigma
        self.multi_noise = multi_noise
        self.scale_spectra = True
        self.scale_params = True
        self.norm_inputs = True
        self.norm_target = True
        self.independent_norm = False
        self.norm_type = 'z_score'
        #training
        self.spaceSigma_min = 5
        self.auto_repeat_n = False
        self.burn_in = False
        self.burnIn_step = None
        self.transfer_learning = False
        self.randn_num = round(abs(np.random.randn()/5.), 5)
    
    def _cholesky_factor(self, cov_matrix):
        if cov_matrix is None:
            return None
        else:
            return np.linalg.cholesky(cov_matrix) #cov=LL^T
            # return cov_matrix #test
    
    def _get_params_tot(self):
        if self.params_vali is None:
            return self.params
        else:
            return np.concatenate((self.params, self.params_vali), axis=0)
    
    def _nodes(self):
        self.node_in = self.spectra.shape[1]
        self.node_out = self.params.shape[1]
        return nodeframe.decreasingNode(node_in=self.node_in,node_out=self.node_out,hidden_layer=self.hidden_layer)
    
    def _net(self):
        if self.fix_initialize:
            torch.manual_seed(1000) #Fixed parameter initialization
        self.nodes = self._nodes()
        self.net = fcnet.FcNet(nodes=self.nodes, activation_func=self.activation_func)
        if self.print_info:
            print(self.net)
    
    def _check_batchSize(self):
        if self.batch_size > len(self.params)*self.multi_noise:
            self.batch_size = len(self.params)*self.multi_noise
            print('The batch size is set to %s'%(self.batch_size))
    
    def _auto_batchSize(self):
        if self.burn_in:
            #here <=2.5 is based on experiments
            if self.spaceSigma_min<=2.5:
                self.batch_size = 500
            else:
                self.batch_size = len(self.params)//4
        else:
            self.batch_size = len(self.params)//2
        #make sure batch size will not too large
        if self.batch_size>1250:
            self.batch_size = 1250
    
    def _auto_epoch(self):
        if not self.burn_in:
            self.epoch = self.base_epoch
    
    def _auto_repeat_n(self, repeat_n):
        if self.burn_in:
            return repeat_n
        else:
            return 1
    
    def statistic(self):
        if self.scale_spectra:
            self.spectra_statistic = dp.Statistic(self.spectra/self.spectra_base).statistic()
        else:
            self.spectra_statistic = dp.Statistic(self.spectra).statistic()
        if self.scale_params:
            self.params = self.scaling(self.params)
            if self.spectra_vali is not None:
                self.params_vali = self.scaling(self.params_vali)
        self.params_tot = self._get_params_tot() #updated
        if self.independent_norm:
            # self.params_statistic = [dp.Statistic(self.params[:,i]).statistic() for i in range(self.params_n)]
            #updated
            self.params_statistic = [dp.Statistic(self.params_tot[:,i]).statistic() for i in range(self.params_n)]
        else:
            # self.params_statistic = dp.Statistic(self.params).statistic()
            #updated
            self.params_statistic = dp.Statistic(self.params_tot).statistic()

    def load_wellTrainedNet(self, path='ann', randn_num='0.123'):
        randn_num = str(randn_num)
        print('\nLoading the well trained network that has random number {}'.format(randn_num))
        file_path = evaluate.FilePath(filedir=path, randn_num=randn_num).filePath()
        self.trained_net = torch.load(file_path)[0]#
        self.transfer_learning = True
    
    def copyLayers_fromTrainedNet(self):
        print('\nCopying hyperparameters of a well trained network to the network')
        self.net.load_state_dict(self.trained_net.state_dict())

    def transfer_data(self):
        if self.use_GPU:
            self.spectra = dp.numpy2cuda(self.spectra)
            self.params = dp.numpy2cuda(self.params)
            if self.cholesky_factor is None:
                self.obs_errors = dp.numpy2cuda(self.obs_errors)
            else:
                self.cholesky_factor = dp.numpy2cuda(self.cholesky_factor)
            self.spectra_base_torch = dp.numpy2cuda(self.spectra_base)
            #vali_set
            if self.spectra_vali is not None:
                self.spectra_vali = dp.numpy2cuda(self.spectra_vali)
                self.params_vali = dp.numpy2cuda(self.params_vali)
        else:
            self.spectra = dp.numpy2torch(self.spectra)
            self.params = dp.numpy2torch(self.params)
            if self.cholesky_factor is None:
                self.obs_errors = dp.numpy2torch(self.obs_errors)
            else:
                self.cholesky_factor = dp.numpy2torch(self.cholesky_factor)
            self.spectra_base_torch = dp.numpy2torch(self.spectra_base)
            #vali_set
            if self.spectra_vali is not None:
                self.spectra_vali = dp.numpy2torch(self.spectra_vali)
                self.params_vali = dp.numpy2torch(self.params_vali)
    
    def preprocessing_input(self, spectra, spectra_base, a=0, b=1):
        if self.scale_spectra:
            spectra = spectra / spectra_base
        if self.norm_inputs:
            spectra = dp.Normalize(spectra, self.spectra_statistic, norm_type=self.norm_type, a=a, b=b).norm()
        return spectra

    def preprocessing_target(self, params, a=0, b=1):
        if self.norm_target:
            if self.independent_norm:
                for i in range(self.params_n):
                    params[:,i] = dp.Normalize(params[:,i], self.params_statistic[i], norm_type=self.norm_type, a=a, b=b).norm()
            else:
                params = dp.Normalize(params, self.params_statistic, norm_type=self.norm_type, a=a, b=b).norm()
        return params
    
    def preprocessing_target_inv(self, params, a=0, b=1):
        if self.norm_target:
            if self.independent_norm:
                for i in range(self.params_n):
                    params[:,i] = dp.InverseNormalize(params[:,i], self.params_statistic[i], norm_type=self.norm_type, a=a, b=b).inverseNorm()
            else:
                params = dp.InverseNormalize(params, self.params_statistic, norm_type=self.norm_type, a=a, b=b).inverseNorm()
        if self.scale_params:
            params = self.inverseScaling(params)
        return params
    
    def train(self, repeat_n=3, showIter_n=100):
        """Train the network

        Parameters
        ----------
        repeat_n : int, optional
            The number of repeat feed to the network for each batch size data, which will increase the number of iterations in each epoch. Default: 3
        showIter_n : int, optional
            The number of epoch that show the training information. Default: 100

        Returns
        -------
        object
            The network object.
        array-like
            The loss.
        """
        # change showIter_n to showEpoch_n???
        self._net()
        if self.transfer_learning:
            self.copyLayers_fromTrainedNet()
        self.transfer_net()
        
        self.optimizer = self._optimizer(name='Adam')
        
        if self.auto_batchSize:
            self._auto_batchSize()
        self._check_batchSize()
        # print('batch size: %s'%self.batch_size)
        if self.auto_epoch:
            self._auto_epoch()
        if self.auto_repeat_n:
            repeat_n = self._auto_repeat_n(repeat_n)
        self.iteration = self.multi_noise * len(self.spectra)//self.batch_size * repeat_n

        self.statistic()
        self.transfer_data()
                
        self.train_loss = []
        self.vali_loss = []
        # np.random.seed(1000)#
        print('randn_num: %s'%self.randn_num)
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.spectra,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            self.inputs = self.preprocessing_input(self.inputs, self.spectra_base_torch)
            self.target = self.preprocessing_target(self.target)
            _, loss_ = self.train_1(self.inputs, self.target, repeat_n=1, set_seed=False, lr_decay=False, print_info=False)            
            loss_mean = np.mean(loss_)
            self.train_loss.append(loss_mean)
            
            #vali_loss
            if self.spectra_vali is not None:
                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.spectra_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
                self.inputs_vali = self.preprocessing_input(self.inputs_vali, self.spectra_base_torch)
                self.target_vali = self.preprocessing_target(self.target_vali)
                self.net.eval()
                pred_vali = self.net(Variable(self.inputs_vali))
                _vali_loss = self.loss_func(pred_vali, Variable(self.target_vali, requires_grad=False))
                self.vali_loss.append(_vali_loss.item())
                self.net.train()
            
            if subsample_num%showIter_n==0:
                if self.spectra_vali is None:
                    print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.optimizer.param_groups[0]['lr']))
                else:
                    print('(epoch:%s/%s; train_loss/vali_loss:%.5f/%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.vali_loss[-1], self.optimizer.param_groups[0]['lr']))
            lrdc = optimize.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[0]['lr'] = lrdc.exp()

        if self.use_multiGPU:
            self.net = self.net.module.cpu()
        else:
            self.net = self.net.cpu()
        self.train_loss = np.array(self.train_loss)
        self.vali_loss = np.array(self.vali_loss)            
        return self.net, self.train_loss, self.vali_loss

    def _predict(self, inputs, use_GPU=False, in_type='torch'):
        """Make predictions using a well-trained network.
        
        Parameters
        ----------
        inputs : numpy array or torch tensor
            The inputs of the network.
        use_GPU : bool
            If True, calculate using GPU, otherwise, calculate using CPU.
        in_type : str
            The data type of the inputs, 'numpy' or 'torch'.
        """
        if use_GPU:
            self.net = self.net.cuda()
            if in_type=='numpy':
                inputs = dp.numpy2cuda(inputs)
            elif in_type=='torch':
                inputs = dp.torch2cuda(inputs)
        else:
            if in_type=='numpy':
                inputs = dp.numpy2torch(inputs)
        self.net.eval() #this works for the batch normalization layers
        pred = self.net(Variable(inputs))
        if use_GPU:
            pred = dp.cuda2numpy(pred.data)
        else:
            pred = dp.torch2numpy(pred.data)
        #reshape chain
        if len(pred.shape)==1:
            pred = pred.reshape(-1, 1)
        return pred

    def predict(self, spectra, use_GPU=False, in_type='torch'):
        if len(spectra.shape)==1:
            spectra = spectra.reshape(1, -1) #for one spectrum
        spectra = self.preprocessing_input(spectra, dp.numpy2torch(self.spectra_base))
        self.pred_params = self._predict(spectra, use_GPU=use_GPU, in_type=in_type)
        self.pred_params = self.preprocessing_target_inv(self.pred_params)
        return self.pred_params
    
    def predict_chain(self, obs_data, cov_matrix=None, chain_leng=10000, use_GPU=False):
        # torch.manual_seed(1000)#
        # limit parameters using params_limit???, do not set limit here, set limit in chain_ann
        obs_data = dp.numpy2torch(obs_data)
        obs_best, obs_errors = obs_data[:,1], obs_data[:,2]
        self.obs_best_multi = torch.ones((chain_leng, len(obs_best))) * obs_best
        if cov_matrix is None:
            cholesky_factor = None
        else:
            cholesky_factor = dp.numpy2torch(np.linalg.cholesky(cov_matrix))
        self.obs_best_multi = ds.AddGaussianNoise(self.obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisySpectra()
        self.chain = self.predict(self.obs_best_multi, use_GPU=use_GPU, in_type='torch')
        self.chain = self.cut_params(self.chain) #remove non-physical parameters
        return self.chain
    
    def predict_params(self, sim_spectra, use_GPU=False):
        sim_spectra = dp.numpy2torch(sim_spectra)
        params = self.predict(sim_spectra, use_GPU=use_GPU, in_type='torch')
        return params
    
    def save_net(self, path='ann', sample=None):
        if sample is None:
            fileName = 'net_train%s_batch%s_epoch%s_%s.pt'%(len(self.params),self.batch_size,self.epoch,self.randn_num)
        else:
            fileName = 'net-%s_train%s_batch%s_epoch%s_%s.pt'%(sample,len(self.params),self.batch_size,self.epoch,self.randn_num)
        utils.saveTorchPt(path+'/net', fileName, self.net)
    
    def save_loss(self, path='ann', sample=None):
        if sample is None:
            fileName = 'loss_train%s_batch%s_epoch%s_%s'%(len(self.params),self.batch_size,self.epoch,self.randn_num)
        else:
            fileName = 'loss-%s_train%s_batch%s_epoch%s_%s'%(sample,len(self.params),self.batch_size,self.epoch,self.randn_num)
        utils.savenpy(path+'/loss', fileName, [self.train_loss, self.vali_loss])
    
    def save_chain(self, path='ann', sample=None):
        if sample is None:
            fileName = 'chain_train%s_batch%s_epoch%s_%s'%(len(self.params),self.batch_size,self.epoch,self.randn_num)
        else:
            fileName = 'chain-%s_train%s_batch%s_epoch%s_%s'%(sample,len(self.params),self.batch_size,self.epoch,self.randn_num)
        utils.savenpy(path+'/chains', fileName, self.chain)
    
    def save_hparams(self, path='ann', sample=None):
        if sample is None:
            fileName = 'hparams_train%s_batch%s_epoch%s_%s'%(len(self.params),self.batch_size,self.epoch,self.randn_num)
        else:
            fileName = 'hparams-%s_train%s_batch%s_epoch%s_%s'%(sample,len(self.params),self.batch_size,self.epoch,self.randn_num)
        utils.savenpy(path+'/hparams', fileName, [self.spectra_statistic, self.params_statistic, self.spectra_base, self.params_base, self.param_names, self.params_dict,
                                                  self.scale_spectra, self.scale_params, self.norm_inputs, self.norm_target, self.independent_norm, self.norm_type, 
                                                  self.burnIn_step])
    
    def plot_loss(self, alpha=0.6, show_logLoss=False):
        vali_loss_size = len(self.vali_loss)
        train_loss_mean = np.mean(self.train_loss[-200:])
        train_loss_min = np.min(self.train_loss[-200:])
        train_loss_max = np.max(self.train_loss[-200:])
        if vali_loss_size==0:
            print ('The average of last 200 training set losses: %.5f\n'%(train_loss_mean))
        else:
            print ('The aveage of last 200 training/validation set losses: %.5f/%.5f\n'%(train_loss_mean, np.mean(self.vali_loss[-200:])))
            vali_loss_min = np.min(self.vali_loss[-200:])
            vali_loss_max = np.min(self.vali_loss[-200:])
        x = np.linspace(1, len(self.train_loss), len(self.train_loss))
        if show_logLoss:
            panel = pls.PlotSettings(set_fig=True, figsize=(6*2, 4.5))
            panel.setting(location=[1,2,1], labels=['Epochs','Loss'])
        else:
            panel = pls.PlotSettings(set_fig=True)
            panel.setting(location=[1,1,1], labels=['Epochs','Loss'])
        plt.plot(x, self.train_loss, label='Training set')
        if vali_loss_size==0:
            loss_min, loss_max = train_loss_min, train_loss_max
        else:
            plt.plot(x, self.vali_loss, label='Validation set', alpha=alpha)
            loss_min, loss_max = min(train_loss_min, vali_loss_min), max(train_loss_max, vali_loss_max)
        loss_diff = loss_max - loss_min
        
        fraction_loss = 0.18
        fraction_low = 0.08
        ylim_tot = loss_diff / fraction_loss
        delta_low = fraction_low * ylim_tot
        ylim_min = loss_min - delta_low
        ylim_max = ylim_min + ylim_tot
        plt.xlim(1, len(self.train_loss))
        plt.ylim(ylim_min, ylim_max)
        panel.set_legend()
        
        if show_logLoss:
            panel.setting(location=[1,2,2], labels=['Epochs','Loss'])
            plt.loglog(x, self.train_loss, label='Training set')
            if vali_loss_size!=0:
                plt.loglog(x, self.vali_loss, label='Validation set', alpha=alpha)
            plt.xlim(1, len(self.train_loss))
            panel.set_legend()
        return panel.fig, panel.ax
    

class LoadNet(object):
    def __init__(self, path='ann', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)

    def load_net(self):
        file_path = evaluate.FilePath(filedir=self.path+'/net', randn_num=self.randn_num).filePath()
        self.net = torch.load(file_path)
    
class LoadLoss(object):
    def __init__(self, path='ann', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)

    def load_loss(self):
        file_path = evaluate.FilePath(filedir=self.path+'/loss', randn_num=self.randn_num, suffix='.npy').filePath()
        self.train_loss, self.vali_loss = np.load(file_path, allow_pickle=True)
    
class LoadChain(object):
    def __init__(self, path='ann', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)
    
    def load_chain(self):
        file_path = evaluate.FilePath(filedir=self.path+'/chains', randn_num=self.randn_num, suffix='.npy').filePath()
        self.chain = np.load(file_path)
       
class LoadHparams(object):
    def __init__(self, path='ann', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)

    def load_hparams(self):
        file_path = evaluate.FilePath(filedir=self.path+'/hparams', randn_num=self.randn_num, suffix='.npy').filePath()
        self.spectra_statistic, self.params_statistic, self.spectra_base, self.params_base, self.param_names, self.params_dict, self.scale_spectra, self.scale_params, self.norm_inputs, self.norm_target, self.independent_norm, self.norm_type, self.burnIn_step = np.load(file_path, allow_pickle=True)
        self.params_n = len(self.param_names)
        p_property = cosmic_params.ParamsProperty(self.param_names, params_dict=self.params_dict)
        self.params_limit = p_property.params_limit

class RePredictOBMLP(OneBranchMLP, LoadNet, LoadLoss, LoadChain, LoadHparams):
    """Repredict cosmological parameters using the saved networks.
    
    Parameters
    ----------
    path : str
        The path of the results saved. Default: 'ann'
    randn_num : str or int
        A random number that identifies the saved results.

    """
    def __init__(self, path='ann', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)
    

#%% multibranch network
class MultiBranchMLP(OneBranchMLP):
    """Multibranch network that is used to predict cosmological parameters with multiple sets of datasets.
        
    Parameters
    ----------
    train_set : list
        The training set that contains simulated observational spectra (data) which is a list spectra with shape [(N,spectra_length_1), (N,spectra_length_2), ...]
        and simulated parameters of a specific cosmological (or theoretical) model. i.e. [spectra, parameters]
    param_names : list
        A list which contains the parameter names, e.g. ['H0','Omega_m','ombh2','omch2','tau','As','ns'].
    vali_set : list, optional
        The validation set that contains simulated observational spectra (data) which is a list spectra with shape [(N,spectra_length_1), (N,spectra_length_2), ...] 
        and simulated parameters of a specific cosmological (or theoretical) model. The validation set can also be set to None.
        i.e. [spectra, parameters] or [None, None]
    obs_errors : list, optional
        Observational errors, it is a list of errors with shape [(spectra_length_1,), (spectra_length_2,), ...]. Default: None
    cov_matrix : list or None, optional
        A list of covariance matrix with shape [(spectra_length_1, spectra_length_1), (spectra_length_2, spectra_length_2), ...].
        If there is no covariance for some observations, the covariance matrix should be set to None. e.g. [cov_matrix_1, None, cov_matrix_3]. Default: None
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    branch_hiddenLayer : int, optional
        The number of the hidden layer for the branch part of the network. Default: 2
    trunk_hiddenLayer : int, optional
        The number of the hidden layer for the trunk part of the network. Default: 1
    activation_func : str, optional
        Activation function, which can be 'relu', 'leakyrelu', 'prelu', 'rrelu', 'relu6', 'elu', 'celu', 'selu', 'silu', 
        'sigmoid', 'logsigmoid', 'tanh', 'tanhshrink', 'softsign', or 'softplus' (see :func:`~.element.activation`). Default: 'rrelu'
    loss_func : str, optional
        The loss function used in the network. Default: 'L1'
    noise_type : str, optional
        The type of Gaussian noise added to the training set, 'singleNormal' or 'multiNormal'. Default: 'multiNormal'
    factor_sigma : float, optional
        For the case of 'singleNormal', it is the factor of the observational error (standard deviation), 
        while for the case of 'multiNormal' it is the standard deviation of the coefficient of the observational error (standard deviation). Default: 0.5
    multi_noise : int, optional
        The number of realization of noise added to a spectrum. Default: 5
    
    Attributes
    ----------
    spectra_base : array-like, optional
        The base value of spectra that is used for data normalization when training the network to ensure that the scaled spectra are ~ 1., 
        it is suggested to set the mean of the simulated spectra. The default is the mean of the simulated spectra.
    params_base : array-like, optional
        The base value of parameters that is used for data normalization when training the network to ensure that the scaled parameters are ~ 1., 
        it is suggested to set the mean of the posterior distribution (or the simulated parameters). The default is the mean of the simulated parameters.
    lr : float, optional
        The learning rate setting of the network. Default: 1e-2
    lr_branch : float, optional
        The learning rate setting of the branch part. Default: 1e-2
    lr_min : float, optional
        The minimum of the learning rate. Default: 1e-8
    batch_size : int, optional
        The batch size setting of the network. Default: 750
    auto_batchSize : bool, optional
        If True, the batch size will be set automatically in the training process, otherwise, use the setting of ``batch_size``. Default: True
    epoch : int, optional
        The number of epoch of the training process. Default: 2000
    epoch_branch : int, optional
        The number of epoch for the branch part. This only works when training the branch part. Default: 2000
    base_epoch : int, optional
        The base number (or the minimum number) of epoch. Default: 1000
    auto_epoch : bool, optional
        If True, the epoch will be set automatically in the training process, otherwise, use the setting of ``epoch``. Default: True
    scale_spectra : bool, optional
        If True, the input data (measurements) will be scaled based on the base values of the data. It is recommended to set to True. Default: True
    scale_params : bool, optional
        If True, the target data (cosmological parameters) will be scaled based on the base values of parameters. 
        See :class:`~.data_processor.ParamsScaling`. It is recommended to set to True. Default: True
    norm_inputs : bool, optional
        If True, the input data of the network will be normalized. Default: True
    norm_target : bool, optional
        If True, the target data (cosmological parameters) will be normalized. Default: True
    independent_norm : bool, optional
        If True, the target data (cosmological parameters) will be normalized independently. This only works when ``norm_target`` is True.
        It is recommended to set to False. Default: False
    norm_type : str, optional
        The method of normalization, 'z_score', 'minmax', or 'mean' (see :class:`~.data_processor.Normalize`). Default: 'z_score'
    spaceSigma_min : int, optional
        The minimum parameter space to be learned, e.g. for spaceSigma_min=5, the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    transfer_learning : bool, optional
        If True, the network will be initialized using the well-trained network of the previous step. Default: False
    
    Note
    ----
    It is suggested to set lr and lr_branch the same value.
    """
    def __init__(self, train_set, param_names, vali_set=[None,None], obs_errors=None, cov_matrix=None, params_dict=None,
                 branch_hiddenLayer=2, trunk_hiddenLayer=1, activation_func='rrelu', loss_func='L1',
                 noise_type='multiNormal', factor_sigma=0.5, multi_noise=5):
        #data
        self.spectra, self.params = train_set
        self.branch_n = len(train_set[0])
        self.spectra_base = [np.mean(self.spectra[i], axis=0) for i in range(self.branch_n)]
        self.params_base = np.mean(self.params, axis=0)
        self.param_names = param_names
        self.params_n = len(param_names)
        self.spectra_vali, self.params_vali = vali_set
        self.obs_errors = self._obs_errors(obs_errors)
        self.cholesky_factor = self._cholesky_factor(cov_matrix)
        self.params_dict = params_dict
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict)
        self.params_limit = p_property.params_limit
        #ANN model
        self.branch_hiddenLayer = branch_hiddenLayer
        self.trunk_hiddenLayer = trunk_hiddenLayer
        self.activation_func = activation_func
        self.loss_func = train.loss_funcs(name=loss_func)
        self.lr = 1e-2
        self.lr_branch = 1e-2
        self.lr_min = 1e-8
        self.batch_size = 750
        self.auto_batchSize = True
        self.epoch = 2000
        self.epoch_branch = 2000
        self.base_epoch = 1000
        self.auto_epoch = True
        self.fix_initialize = False
        self.print_info = False
        #data preprocessing
        self.noise_type = noise_type
        self.factor_sigma = factor_sigma
        self.multi_noise = multi_noise
        self.scale_spectra = True
        self.scale_params = True
        self.norm_inputs = True
        self.norm_target = True
        self.independent_norm = False
        self.norm_type = 'z_score'
        #training
        self.spaceSigma_min = 5
        self.auto_repeat_n = False
        self.burn_in = False
        self.burnIn_step = None
        self.transfer_learning = False
        self.randn_num = round(abs(np.random.randn()/5.), 5)
    
    def _obs_errors(self, errors):
        if errors is None:
            return [None for i in range(self.branch_n)]
        else:
            return errors

    def _cholesky_factor(self, cov_matrix):
        if cov_matrix is None:
            return [None for i in range(self.branch_n)]
        else:
            cholesky_f = []
            for i in range(self.branch_n):
                if cov_matrix[i] is None:
                    cholesky_f.append(None)
                else:
                    cholesky_f.append(np.linalg.cholesky(cov_matrix[i]))
                    # cholesky_f.append(cov_matrix[i]) #test
            return cholesky_f
    
    def _nodes(self):
        self.nodes_in = []
        self.node_out = self.params.shape[1]
        for i in range(self.branch_n):
            self.nodes_in.append(self.spectra[i].shape[1])
        self.fc_hidden = self.branch_hiddenLayer*2 + 1
        # self.fc_hidden = self.branch_hiddenLayer + self.trunk_hiddenLayer + 1 #also works, but not necessary
    
    def _net(self):
        if self.fix_initialize:
            torch.manual_seed(1000) #Fixed parameter initialization
        self._nodes()
        for i in range(self.branch_n):
            exec('self.branch_net%s=fcnet.FcNet(node_in=self.nodes_in[i], node_out=self.node_out,\
            hidden_layer=self.fc_hidden, activation_func=self.activation_func)'%(i+1))
        self.net = fcnet.MultiBranchFcNet(nodes_in=self.nodes_in, node_out=self.node_out, branch_hiddenLayer=self.branch_hiddenLayer,
                                          trunk_hiddenLayer=self.trunk_hiddenLayer, nodes_all=None, activation_func=self.activation_func)
        if self.print_info:
            print(self.net)
    
    #change the branch net & trunk net (contain training) to use multiple GPUs ???
    def transfer_branchNet(self, device=None):
        if self.use_GPU:
            for i in range(1, self.branch_n+1):
                exec('self.branch_net%s = self.branch_net%s.cuda(device)'%(i,i))
    
    def _copyLayer_fromBranch(self, branch_index=None):
        if branch_index is None:
            print('\nCopying hyperparameters of the branch networks to the multibranch network')
            for i in range(1, self.branch_n+1):
                for j in range(self.branch_hiddenLayer+1):
                    eval('self.net.branch%s[j*3].load_state_dict(self.branch_net%s.fc[j*3].state_dict())'%(i, i))#copy Linear
                    eval('self.net.branch%s[j*3+1].load_state_dict(self.branch_net%s.fc[j*3+1].state_dict())'%(i, i))#copy BN
        else:
            print('Copying hyperparameters of the branch network {} to the multibranch network\n'.format(branch_index))
            for j in range(self.branch_hiddenLayer+1):
                eval('self.net.branch%s[j*3].load_state_dict(self.branch_net%s.fc[j*3].state_dict())'%(branch_index, branch_index))#copy Linear
                eval('self.net.branch%s[j*3+1].load_state_dict(self.branch_net%s.fc[j*3+1].state_dict())'%(branch_index, branch_index))#copy BN
    
    def statistic(self):
        if self.scale_spectra:
            self.spectra_statistic = [dp.Statistic(self.spectra[i]/self.spectra_base[i]).statistic() for i in range(self.branch_n)]
            
            #test
            # sp_comb = np.concatenate(self.spectra, axis=1)/np.concatenate(self.spectra_base)
            # self.spectra_statistic = [dp.Statistic(sp_comb).statistic() for i in range(self.branch_n)]
        else:
            self.spectra_statistic = [dp.Statistic(self.spectra[i]).statistic() for i in range(self.branch_n)]
            
            #test
            # sp_comb = np.concatenate(self.spectra, axis=1)
            # self.spectra_statistic = [dp.Statistic(sp_comb).statistic() for i in range(self.branch_n)]
            
        if self.scale_params:
            self.params = self.scaling(self.params)
            if self.spectra_vali is not None:
                self.params_vali = self.scaling(self.params_vali)
        self.params_tot = self._get_params_tot() #updated
        if self.independent_norm:
            # self.params_statistic = [dp.Statistic(self.params[:,i]).statistic() for i in range(self.params_n)]
            #updated
            self.params_statistic = [dp.Statistic(self.params_tot[:,i]).statistic() for i in range(self.params_n)]
        else:
            # self.params_statistic = dp.Statistic(self.params).statistic()
            #updated
            self.params_statistic = dp.Statistic(self.params_tot).statistic()
        
    def transfer_data(self):
        if self.use_GPU:
            self.spectra = [dp.numpy2cuda(self.spectra[i]) for i in range(self.branch_n)]
            self.params = dp.numpy2cuda(self.params)
            for i in range(self.branch_n):
                if self.cholesky_factor[i] is None:
                    self.obs_errors[i] = dp.numpy2cuda(self.obs_errors[i])
                else:
                    self.cholesky_factor[i] = dp.numpy2cuda(self.cholesky_factor[i])
            self.spectra_base_torch = [dp.numpy2cuda(self.spectra_base[i]) for i in range(self.branch_n)]
            # self.spectra_base_torch = self.spectra_base #test
            #vali_set
            if self.spectra_vali is not None:
                self.spectra_vali = [dp.numpy2cuda(self.spectra_vali[i]) for i in range(self.branch_n)]
                self.params_vali = dp.numpy2cuda(self.params_vali)
        else:
            self.spectra = [dp.numpy2torch(self.spectra[i]) for i in range(self.branch_n)]
            self.params = dp.numpy2torch(self.params)
            for i in range(self.branch_n):
                if self.cholesky_factor[i] is None:
                    self.obs_errors[i] = dp.numpy2torch(self.obs_errors[i])
                else:
                    self.cholesky_factor[i] = dp.numpy2torch(self.cholesky_factor[i])
            self.spectra_base_torch = [dp.numpy2torch(self.spectra_base[i]) for i in range(self.branch_n)]
            #vali_set
            if self.spectra_vali is not None:
                self.spectra_vali = [dp.numpy2torch(self.spectra_vali[i]) for i in range(self.branch_n)]
                self.params_vali = dp.numpy2torch(self.params_vali)
            
#    def transfer_subData(self, device=None):
#        if self.use_GPU:
#            self.inputs = dp.numpy2cuda(self.inputs, device=device)
#            self.target = dp.numpy2cuda(self.target, device=device)
#            self.error = dp.numpy2cuda(self.error, device=device)
#        else:
#            self.inputs = dp.numpy2torch(self.inputs)
#            self.target = dp.numpy2torch(self.target)
#            self.error = dp.numpy2torch(self.error)
    
    def _train_branch(self, rank, repeat_n, showIter_n, device):
        
        optimizer = torch.optim.Adam(eval('self.branch_net%s.parameters()'%(rank+1)), lr=self.lr_branch)
        iteration = self.multi_noise*len(self.spectra[0])//self.batch_size * repeat_n
        
        self.inputs = self.spectra[rank]
        self.target = self.params
        self.error = self.obs_errors[rank]
        self.cholesky_f = self.cholesky_factor[rank]
#        self.transfer_subData(device=device)
        
        print('Training the branch network %s'%(rank+1))
        for subsample_num in range(1, self.epoch_branch+1):
            _inputs, _target = ds.AddGaussianNoise(self.inputs,params=self.target,obs_errors=self.error,cholesky_factor=self.cholesky_f,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            
            if self.scale_spectra:
                _inputs = _inputs / self.spectra_base_torch[rank] #to be tested !!!
            if self.norm_inputs:
                _inputs = dp.Normalize(_inputs, self.spectra_statistic[rank], norm_type=self.norm_type).norm()
            if self.norm_target:
                if self.independent_norm:
                    for i in range(self.params_n):
                        _target[:,i] = dp.Normalize(_target[:,i], self.params_statistic[i], norm_type=self.norm_type).norm()
                else:
                    _target = dp.Normalize(_target, self.params_statistic, norm_type=self.norm_type).norm()
            
            for iter_mid in range(1, iteration+1):
                batch_index = np.random.choice(len(_inputs), self.batch_size, replace=False)
                xx = _inputs[batch_index]
                yy = _target[batch_index]
                xx = Variable(xx)
                yy = Variable(yy, requires_grad=False)
                
                _predicted = eval('self.branch_net%s(xx)'%(rank+1))
                _loss = self.loss_func(_predicted, yy)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                
            if subsample_num%showIter_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch_branch, _loss.item(), optimizer.param_groups[0]['lr']))
            lrdc = optimize.LrDecay(subsample_num,iteration=self.epoch_branch,lr=self.lr_branch,lr_min=self.lr_min)
            optimizer.param_groups[0]['lr'] = lrdc.exp()
        
        #############################################################################
        # Note: hyperparameters must be transferred in the subprocess.
        #
        # Variables defined in the subprocess can not be called by the main process,
        # but, the hyperparameters of "self.branch_net%s"%i can be copied to "self.net",
        # the reason may be that hyperparameters of the network shared the memory.
        #############################################################################
        #print(eval("self.branch_net%s.fc[3].state_dict()['bias'][:5]"%(rank+1)))
        self._copyLayer_fromBranch(branch_index=rank+1)
    
    def _train_branchNet(self, repeat_n=3, showIter_n=10):
        #############################################################################
        # Note: variables used in the subprocess (in the function self._train_branch)
        #       should be defined before using "mp.spawn", and variables defined in the
        #       subprocess can not be called by the main process.
        #
        # # the following lines have the same function as "mp.spawn"
        # mp.set_start_method('spawn') #this is important
        # processes = []
        # for rank in range(self.branch_n):
        #     p = mp.Process(target=self._train_branch, args=(rank, repeat_n, showIter_n, device))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        #############################################################################
        
        #this means that the branch networks can only be trained on 1 GPU, how to train them on muliple GPUs?
        device = None
        
        #Note: all networks should be transfered to GPU when using "mp.spawn" to train the branch networks
        self.transfer_branchNet(device=device)
        mp.spawn(self._train_branch, nprocs=self.branch_n, args=(repeat_n, showIter_n, device), join=True)
    
    def _train_trunk(self, repeat_n=3, showIter_n=100, fix_lr=1e-4, reduce_fix_lr=False):
        branch_p = []
        for i in range(1, self.branch_n+1):
            branch_p.append(eval("{'params':self.net.branch%s.parameters(), 'lr':fix_lr}"%i)) #lr=fix_lr
        optimizer = torch.optim.Adam(branch_p + [{'params':self.net.trunk.parameters()}], lr=self.lr)

        print('Training the trunk network')
        for subsample_num in range(1, self.epoch_branch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.spectra,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            self.inputs = self.preprocessing_multiBranch_input(self.inputs, self.spectra_base_torch)
            self.target = self.preprocessing_multiBranch_target(self.target)
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.inputs[0]), self.batch_size, replace=False)
                xx = [self.inputs[i][batch_index] for i in range(self.branch_n)]
                yy = self.target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                
            if subsample_num%showIter_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch_branch, _loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']))
            lrdc_t = optimize.LrDecay(subsample_num,iteration=self.epoch_branch,lr=self.lr,lr_min=self.lr_min)
            optimizer.param_groups[-1]['lr'] = lrdc_t.exp()
            #test
            if reduce_fix_lr:
                lrdc_b = optimize.LrDecay(subsample_num,iteration=self.epoch_branch,lr=fix_lr,lr_min=self.lr_min)#change to lr=self.lr ?
                for i in range(self.branch_n):
                    optimizer.param_groups[i]['lr'] = lrdc_b.exp()
    
    def preprocessing_multiBranch_input(self, spectra, spectra_base, a=0, b=1):
        if self.scale_spectra:
            spectra = [spectra[i]/spectra_base[i] for i in range(self.branch_n)]
        if self.norm_inputs:
            spectra = [dp.Normalize(spectra[i], self.spectra_statistic[i], norm_type=self.norm_type, a=a, b=b).norm() for i in range(self.branch_n)]
        return spectra
    
    def preprocessing_multiBranch_target(self, params, a=0, b=1):
        if self.norm_target:
            if self.independent_norm:
                for i in range(self.params_n):
                    params[:,i] = dp.Normalize(params[:,i], self.params_statistic[i], norm_type=self.norm_type, a=a, b=b).norm()
            else:
                params = dp.Normalize(params, self.params_statistic, norm_type=self.norm_type, a=a, b=b).norm()
        return params
    
    def train(self, repeat_n=3, showIter_n=100, train_branch=True, parallel=True, train_trunk=False, fix_lr=1e-4, reduce_fix_lr=False):
        self._net()
        if self.transfer_learning==True and train_branch==False:
            self.copyLayers_fromTrainedNet()
        self.transfer_net(prints=self.print_info)
        
        #branch_p = [eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i) for i in range(1,self.branch_n+1)] #this will raise an error in python3.X
        #however, the following lines run well for both python2.X and python3.X, why?
        branch_p = []
        for i in range(1, self.branch_n+1):
            branch_p.append(eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i))
        self.optimizer = torch.optim.Adam(branch_p + [{'params':self.net.trunk.parameters()}], lr=self.lr)
        
        #added
        if self.auto_batchSize:
            self._auto_batchSize()
        self._check_batchSize()
        # print('batch size: %s'%self.batch_size)
        if self.auto_epoch:
            self._auto_epoch()
        if self.auto_repeat_n:
            repeat_n = self._auto_repeat_n(repeat_n)
        self.iteration = self.multi_noise*len(self.spectra[0])//self.batch_size * repeat_n
        # print('repeat_n:%s'%repeat_n)

        self.statistic()
        self.transfer_data()
        
        # np.random.seed(1000)#
        print('randn_num: {}'.format(self.randn_num))
        if train_branch:
            if parallel:
                self._train_branchNet(repeat_n=repeat_n, showIter_n=showIter_n)
            else:
                self.transfer_branchNet()
                for rank in range(self.branch_n):
                    self._train_branch(rank, repeat_n, showIter_n, None)
        
        if train_trunk:
            self._train_trunk(repeat_n=repeat_n, showIter_n=showIter_n, fix_lr=fix_lr, reduce_fix_lr=reduce_fix_lr)
        
        self.train_loss = [] #loss
        self.vali_loss = []
        print('\nTraining the multibranch network')
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.spectra,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            self.inputs = self.preprocessing_multiBranch_input(self.inputs, self.spectra_base_torch)
            self.target = self.preprocessing_multiBranch_target(self.target)
            running_loss = []
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.inputs[0]), self.batch_size, replace=False)
                xx = [self.inputs[i][batch_index] for i in range(self.branch_n)]
                yy = self.target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                running_loss.append(_loss.item())
            loss_mean = np.mean(running_loss)
            self.train_loss.append(loss_mean)
            
            #vali_set
            if self.spectra_vali is not None:
                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.spectra_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
                self.inputs_vali = self.preprocessing_multiBranch_input(self.inputs_vali, self.spectra_base_torch)
                self.target_vali = self.preprocessing_multiBranch_target(self.target_vali)
                self.net.eval()
                pred_vali = self.net([Variable(self.inputs_vali[i]) for i in range(self.branch_n)])
                _loss_vali = self.loss_func(pred_vali, Variable(self.target_vali, requires_grad=False))
                self.vali_loss.append(_loss_vali.item())
                self.net.train()
            
            if subsample_num%showIter_n==0:
                if self.lr==self.lr_branch:
                    if self.spectra_vali is None:
                        print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.optimizer.param_groups[0]['lr']))
                    else:
                        print('(epoch:%s/%s; train_loss/vali_loss:%.5f/%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.vali_loss[-1], self.optimizer.param_groups[0]['lr']))
                else:
                    if self.spectra_vali is None:
                        print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch, loss_mean, self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[-1]['lr']))
                    else:
                        print('(epoch:%s/%s; train_loss/vali_loss:%.5f/%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch, loss_mean, self.vali_loss[-1], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[-1]['lr']))
            lrdc_t = optimize.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[-1]['lr'] = lrdc_t.exp()
            lrdc_b = optimize.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr_branch,lr_min=self.lr_min)#change to lr=self.lr ?
            for i in range(self.branch_n):
                self.optimizer.param_groups[i]['lr'] = lrdc_b.exp()
        
        if self.use_multiGPU:
            self.net = self.net.module.cpu()
        else:
            self.net = self.net.cpu()
        self.train_loss = np.array(self.train_loss)
        self.vali_loss = np.array(self.vali_loss)
        return self.net, self.train_loss, self.vali_loss
    
    
    #test, 
    def _train_netBranch(self, repeat_n=3, showIter_n=100):
        #branch_p = [eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i) for i in range(1,self.branch_n+1)] #this will raise an error in python3.X
        #however, the following lines run well for both python2.X and python3.X, why?
        branch_p = []
        for i in range(1, self.branch_n+1):
            branch_p.append(eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i))
        trunk_p = [{'params':self.net.trunk.parameters(), 'lr':0}]
        optimizer = torch.optim.Adam(branch_p + trunk_p)
        
        print('\nTraining the branch parts of the network')
        for subsample_num in range(1, self.epoch_branch+1):
            _inputs, _target = ds.AddGaussianNoise(self.spectra,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            
            if self.scale_spectra:
                _inputs = [_inputs[i]/self.spectra_base_torch[i] for i in range(self.branch_n)]
            if self.norm_inputs:
                _inputs = [dp.Normalize(_inputs[i], self.spectra_statistic[i], norm_type=self.norm_type).norm() for i in range(self.branch_n)]
            if self.norm_target:
                if self.independent_norm:
                    for i in range(self.params_n):
                        _target[:,i] = dp.Normalize(_target[:,i], self.params_statistic[i], norm_type=self.norm_type).norm()
                else:
                    _target = dp.Normalize(_target, self.params_statistic, norm_type=self.norm_type).norm()
            
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(_inputs[0]), self.batch_size, replace=False)
                xx = [_inputs[i][batch_index] for i in range(self.branch_n)]
                yy = _target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                
            if subsample_num%showIter_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch_branch, _loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']))
            # lrdc_t = optimize.LrDecay(subsample_num,iteration=epoch,lr=self.lr,lr_min=self.lr_min)
            # self.optimizer.param_groups[-1]['lr'] = lrdc_t.exp()
            lrdc_b = optimize.LrDecay(subsample_num,iteration=self.epoch_branch,lr=self.lr_branch,lr_min=self.lr_min)#change to lr=self.lr ?
            for i in range(self.branch_n):
                optimizer.param_groups[i]['lr'] = lrdc_b.exp()
        
    #test, 
    def _train_netTrunk(self, repeat_n=3, showIter_n=100):
        #branch_p = [eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i) for i in range(1,self.branch_n+1)] #this will raise an error in python3.X
        #however, the following lines run well for both python2.X and python3.X, why?
        branch_p = []
        for i in range(1, self.branch_n+1):
            branch_p.append(eval("{'params':self.net.branch%s.parameters(), 'lr':0}"%i))
        trunk_p = [{'params':self.net.trunk.parameters(), 'lr':self.lr_branch}]
        optimizer = torch.optim.Adam(branch_p + trunk_p)
        
        print('\nTraining the trunk part of the network')
        for subsample_num in range(1, self.epoch_branch+1):
            _inputs, _target = ds.AddGaussianNoise(self.spectra,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            
            if self.scale_spectra:
                _inputs = [_inputs[i]/self.spectra_base_torch[i] for i in range(self.branch_n)]
            if self.norm_inputs:
                _inputs = [dp.Normalize(_inputs[i], self.spectra_statistic[i], norm_type=self.norm_type).norm() for i in range(self.branch_n)]
            if self.norm_target:
                if self.independent_norm:
                    for i in range(self.params_n):
                        _target[:,i] = dp.Normalize(_target[:,i], self.params_statistic[i], norm_type=self.norm_type).norm()
                else:
                    _target = dp.Normalize(_target, self.params_statistic, norm_type=self.norm_type).norm()
            
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(_inputs[0]), self.batch_size, replace=False)
                xx = [_inputs[i][batch_index] for i in range(self.branch_n)]
                yy = _target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                
            if subsample_num%showIter_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch_branch, _loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']))
            lrdc_t = optimize.LrDecay(subsample_num,iteration=self.epoch_branch,lr=self.lr_branch,lr_min=self.lr_min)#change to lr=self.lr ?
            optimizer.param_groups[-1]['lr'] = lrdc_t.exp()
            # lrdc_b = optimize.LrDecay(subsample_num,iteration=epoch,lr=self.lr_branch,lr_min=self.lr_min)#change to lr=self.lr ?
            # for i in range(self.branch_n):
            #     optimizer.param_groups[i]['lr'] = lrdc_b.exp()
    
    
    #test, 
    def train_branch_trunk(self, repeat_n=3, showIter_n=100, train_branch_trunk=True):
        self._net()
        if self.transfer_learning==True:
            self.copyLayers_fromTrainedNet()
        self.transfer_net(prints=self.print_info)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        #added
        if self.auto_batchSize:
            self._auto_batchSize()
        self._check_batchSize()
        # print('batch size: %s'%self.batch_size)
        if self.auto_epoch:
            self._auto_epoch()
        if self.auto_repeat_n:
            repeat_n = self._auto_repeat_n(repeat_n)
        self.iteration = self.multi_noise*len(self.spectra[0])//self.batch_size * repeat_n
        # print('repeat_n:%s'%repeat_n)

        self.statistic()
        self.transfer_data()
        
        # np.random.seed(1000)#
        print('randn_num: {}'.format(self.randn_num))
        if train_branch_trunk:
            self._train_netBranch(repeat_n=repeat_n, showIter_n=showIter_n)
            self._train_netTrunk(repeat_n=repeat_n, showIter_n=showIter_n)
        
        
        self.loss = []
        print('\nTraining the multibranch network')
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.spectra,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            
            if self.scale_spectra:
                self.inputs = [self.inputs[i]/self.spectra_base_torch[i] for i in range(self.branch_n)]
            if self.norm_inputs:
                self.inputs = [dp.Normalize(self.inputs[i], self.spectra_statistic[i], norm_type=self.norm_type).norm() for i in range(self.branch_n)]
            if self.norm_target:
                if self.independent_norm:
                    for i in range(self.params_n):
                        self.target[:,i] = dp.Normalize(self.target[:,i], self.params_statistic[i], norm_type=self.norm_type).norm()
                else:
                    self.target = dp.Normalize(self.target, self.params_statistic, norm_type=self.norm_type).norm()
            
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.inputs[0]), self.batch_size, replace=False)
                xx = [self.inputs[i][batch_index] for i in range(self.branch_n)]
                yy = self.target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                self.loss.append(_loss.item())
                
            if subsample_num%showIter_n==0:
                if self.lr==self.lr_branch:
                    print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, self.loss[-1], self.optimizer.param_groups[0]['lr']))
                else:
                    print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch, self.loss[-1], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[-1]['lr']))
            lrdc = optimize.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[0]['lr'] = lrdc.exp()
            
        if self.use_multiGPU:
            self.net = self.net.module.cpu()
        else:
            self.net = self.net.cpu()
        self.loss = np.array(self.loss)
        return self.net, self.loss
    
    
    
    #to be updated, to remove params net
    def _predict(self, inputs, in_type='numpy'):
        # in_type: 'numpy' or 'torch'
        self.net.eval() #this works for the batch normalization layers
        if in_type=='numpy':
            inputs = [dp.numpy2torch(inputs[i]) for i in range(len(inputs))]
        inputs = [Variable(inputs[i]) for i in range(len(inputs))]
        pred = self.net(inputs)
        pred = dp.torch2numpy(pred.data)
        #reshape chain
        if len(pred.shape)==1:
            pred = pred.reshape(-1, 1)
        return pred
    
    def predict(self, spectra, in_type='torch'):
        # spectra: [spectra1, spectra2, ...]
        if len(spectra[0].shape)==1:
            spectra = [spectra[i].reshape(1, -1) for i in range(len(spectra))] #for one spectrum
        spectra = self.preprocessing_multiBranch_input(spectra, [dp.numpy2torch(self.spectra_base[i]) for i in range(len(spectra))])
        self.pred_params = self._predict(spectra, in_type=in_type)
        self.pred_params = self.preprocessing_target_inv(self.pred_params)
        return self.pred_params
    
    def predict_chain(self, obs_data, cov_matrix=None, chain_leng=10000):
        # obs_data: observational spectrum in a list [spectra1, spectra2, ...], each element has shape (N, 3)
        # torch.manual_seed(1000)#
        if cov_matrix is None:
            cov_matrix = [None for i in range(len(obs_data))]
        obs_data = [dp.numpy2torch(obs_data[i]) for i in range(len(obs_data))]
        obs_best = [obs_data[i][:,1] for i in range(len(obs_data))]
        obs_errors = [obs_data[i][:,2] for i in range(len(obs_data))]
        obs_best_multi = [torch.ones((chain_leng, len(obs_best[i]))) * obs_best[i] for i in range(len(obs_data))]
        cholesky_factor = []
        for i in range(len(obs_data)):
            if cov_matrix[i] is None:
                cholesky_factor.append(None)
            else:
                cholesky_factor.append(dp.numpy2torch(np.linalg.cholesky(cov_matrix[i])))
        obs_best_multi = ds.AddGaussianNoise(obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisySpectra()
        self.chain = self.predict(obs_best_multi, in_type='torch')
        self.chain = self.cut_params(self.chain) #remove non-physical parameters
        return self.chain
    
    def save_net(self, path='ann', sample='TT'):
        if sample is None:
            fileName = 'net_branch%s_train%s_batch%s_epoch%s_epochBranch%s_%s.pt'%(self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.randn_num)
        else:
            fileName = 'net-%s_branch%s_train%s_batch%s_epoch%s_epochBranch%s_%s.pt'%(sample,self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.randn_num)
        utils.saveTorchPt(path+'/net', fileName, self.net)
        
    def save_loss(self, path='ann', sample='TT'):
        if sample is None:
            fileName = 'loss_branch%s_train%s_batch%s_epoch%s_epochBranch%s_%s'%(self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.randn_num)
        else:
            fileName = 'loss-%s_branch%s_train%s_batch%s_epoch%s_epochBranch%s_%s'%(sample,self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.randn_num)
        utils.savenpy(path+'/loss', fileName, [self.train_loss, self.vali_loss])
    
    def save_chain(self, path='ann', sample='TT'):
        if sample is None:
            fileName = 'chain_branch%s_train%s_batch%s_epoch%s_epochBranch%s_%s'%(self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.randn_num)
        else:
            fileName = 'chain-%s_branch%s_train%s_batch%s_epoch%s_epochBranch%s_%s'%(sample,self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.randn_num)
        utils.savenpy(path+'/chains', fileName, self.chain)
    
    def save_hparams(self, path='ann', sample='TT'):
        if sample is None:
            fileName = 'hparams_branch%s_train%s_batch%s_epoch%s_epochBranch%s_%s'%(self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.randn_num)
        else:
            fileName = 'hparams-%s_branch%s_train%s_batch%s_epoch%s_epochBranch%s_%s'%(sample,self.branch_n,len(self.params),self.batch_size,self.epoch,self.epoch_branch,self.randn_num)
        utils.savenpy(path+'/hparams', fileName, [self.spectra_statistic, self.params_statistic, self.spectra_base, self.params_base, self.param_names, self.params_dict,
                                                  self.scale_spectra, self.scale_params, self.norm_inputs, self.norm_target, self.independent_norm, self.norm_type, 
                                                  self.burnIn_step])

class LoadHparams_MB(object):
    def __init__(self, path='ann', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)

    def load_hparams(self):
        file_path = evaluate.FilePath(filedir=self.path+'/hparams', randn_num=self.randn_num, suffix='.npy').filePath()
        self.spectra_statistic, self.params_statistic, self.spectra_base, self.params_base, self.param_names, self.params_dict, self.scale_spectra, self.scale_params, self.norm_inputs, self.norm_target, self.independent_norm, self.norm_type, self.burnIn_step = np.load(file_path, allow_pickle=True)
        self.params_n = len(self.param_names)
        p_property = cosmic_params.ParamsProperty(self.param_names, params_dict=self.params_dict)
        self.params_limit = p_property.params_limit
        self.branch_n = len(self.spectra_statistic)

class RePredictMBMLP(MultiBranchMLP, LoadNet, LoadLoss, LoadChain, LoadHparams_MB):
    """Repredict cosmological parameters using the saved networks.
    
    Parameters
    ----------
    path : str
        The path of the results saved. Default: 'ann'
    randn_num : str or int
        A random number that identifies the saved results.
    
    """
    def __init__(self, path='ann', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)

#!!!
# 1. check how to reduce training time, when using multi_noise
# 2. whether to use transfer learning?


#%% test a new model for multiple data sets
class MultiDataSetMLP(OneBranchMLP):
    def __init__(self, spectra, parameters, param_names, obs_data, cov_matrix=None, params_dict=None,
                 hidden_layer=3, activation_func='rrelu', loss_func='L1',
                 noise_type='multiNormal', factor_sigma=0.5, multi_noise=5):
        #data
        self.spectra = spectra
        self.branch_n = len(spectra)
        self.spectra_base = [np.mean(spectra[i], axis=0) for i in range(self.branch_n)]
        self.params = parameters
        self.params_base = np.mean(parameters, axis=0)
        self.param_names = param_names
        self.params_n = len(param_names)
        self.obs_data = obs_data #updated, update for multi-branch network !!!
        self.obs_errors = [obs_data[i][:,2] for i in range(len(obs_data))]
        self.cov_matrix = cov_matrix
        self.cholesky_factor = self._cholesky_factor(cov_matrix)
        self.params_dict = params_dict
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict)
        self.params_limit = p_property.params_limit
        #ANN model
        self.hidden_layer = hidden_layer
        self.activation_func = activation_func
        self.loss_func = train.loss_funcs(name=loss_func) #train.Train should be updated
        self.lr = 1e-2
        self.lr_min = 1e-8
        self.batch_size = 750
        self.auto_batchSize = True
        self.epoch = 2000
        self.base_epoch = 1000
        self.auto_epoch = True
        self.fix_initialize = False
        self.print_info = False
        #data preprocessing
        self.noise_type = noise_type
        self.factor_sigma = factor_sigma
        self.multi_noise = multi_noise
        self.scale_spectra = True
        self.scale_params = True
        self.norm_inputs = True
        self.norm_target = True
        self.independent_norm = False
        self.norm_type = 'z_score'
        #training
        self.spaceSigma_min = 5
        self.auto_repeat_n = False
        self.burn_in = False
        self.burnIn_step = None
        self.transfer_learning = False
        self.randn_num = round(abs(np.random.randn()/5.), 5)

    def _cholesky_factor(self, cov_matrix):
        if cov_matrix is None:
            return [None for i in range(self.branch_n)]
        else:
            cholesky_f = []
            for i in range(self.branch_n):
                if cov_matrix[i] is None:
                    cholesky_f.append(None)
                else:
                    cholesky_f.append(np.linalg.cholesky(cov_matrix[i]))
            return cholesky_f
    
    def _nodes(self):
        self.node_in = sum([self.spectra[i].shape[1] for i in range(self.branch_n)])
        self.node_out = self.params.shape[1]
        return nodeframe.decreasingNode(node_in=self.node_in,node_out=self.node_out,hidden_layer=self.hidden_layer)
    
    def statistic(self):
        if self.scale_spectra:
            self.spectra_statistic = [dp.Statistic(self.spectra[i]/self.spectra_base[i]).statistic() for i in range(self.branch_n)]
        else:
            self.spectra_statistic = [dp.Statistic(self.spectra[i]).statistic() for i in range(self.branch_n)]
        if self.scale_params:
            self.params = self.scaling(self.params)
        if self.independent_norm:
            self.params_statistic = [dp.Statistic(self.params[:,i]).statistic() for i in range(self.params_n)]
        else:
            self.params_statistic = dp.Statistic(self.params).statistic()
        
    def transfer_data(self):
        if self.use_GPU:
            self.spectra = [dp.numpy2cuda(self.spectra[i]) for i in range(self.branch_n)]
            self.params = dp.numpy2cuda(self.params)
            for i in range(self.branch_n):
                if self.cholesky_factor[i] is None:
                    self.obs_errors[i] = dp.numpy2cuda(self.obs_errors[i])
                else:
                    self.cholesky_factor[i] = dp.numpy2cuda(self.cholesky_factor[i])
            self.spectra_base_torch = [dp.numpy2cuda(self.spectra_base[i]) for i in range(self.branch_n)]
        else:
            self.spectra = [dp.numpy2torch(self.spectra[i]) for i in range(self.branch_n)]
            self.params = dp.numpy2torch(self.params)
            for i in range(self.branch_n):
                if self.cholesky_factor[i] is None:
                    self.obs_errors[i] = dp.numpy2torch(self.obs_errors[i])
                else:
                    self.cholesky_factor[i] = dp.numpy2torch(self.cholesky_factor[i])
            self.spectra_base_torch = [dp.numpy2torch(self.spectra_base[i]) for i in range(self.branch_n)]
    
    def train(self, repeat_n=3, showIter_n=100):
        self._net()
        if self.transfer_learning:
            self.copyLayers_fromTrainedNet()
        self.transfer_net(prints=self.print_info)
        
        self.optimizer = self._optimizer(name='Adam')
        
        if self.auto_batchSize:
            self._auto_batchSize()
        self._check_batchSize()
        # print('batch size: %s'%self.batch_size)
        if self.auto_epoch:
            self._auto_epoch()
        if self.auto_repeat_n:
            repeat_n = self._auto_repeat_n(repeat_n)
        self.iteration = self.multi_noise * len(self.spectra[0])//self.batch_size * repeat_n

        self.statistic()
        self.transfer_data()
        
        self.loss = []
        print('randn_num: %s'%self.randn_num)
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.spectra,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            
            if self.scale_spectra:
                self.inputs = [self.inputs[i]/self.spectra_base_torch[i] for i in range(self.branch_n)]
            if self.norm_inputs:
                self.inputs = [dp.Normalize(self.inputs[i], self.spectra_statistic[i], norm_type=self.norm_type).norm() for i in range(self.branch_n)]
            if self.norm_target:
                if self.independent_norm:
                    for i in range(self.params_n):
                        self.target[:,i] = dp.Normalize(self.target[:,i], self.params_statistic[i], norm_type=self.norm_type).norm()
                else:
                    self.target = dp.Normalize(self.target, self.params_statistic, norm_type=self.norm_type).norm()
            
            self.inputs = torch.cat(self.inputs, dim=1)
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.inputs), self.batch_size, replace=False)
                xx = self.inputs[batch_index]
                yy = self.target[batch_index]
                xx = Variable(xx)
                yy = Variable(yy, requires_grad=False)
                
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                self.loss.append(_loss.item())
            
            if subsample_num%showIter_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, self.loss[-1], self.optimizer.param_groups[0]['lr']))
            lrdc = optimize.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[0]['lr'] = lrdc.exp()
            
        if self.use_multiGPU:
            self.net = self.net.module.cpu()
        else:
            self.net = self.net.cpu()
        self.loss = np.array(self.loss)
        return self.net, self.loss

    def predict(self, spectra, in_type='torch'):
        # spectra: [spectra1, spectra2, ...]
        if len(spectra[0].shape)==1:
            spectra = [spectra[i].reshape(1, -1) for i in range(len(spectra))] #for one spectrum
        if self.scale_spectra:
            spectra = [spectra[i]/dp.numpy2torch(self.spectra_base[i]) for i in range(len(spectra))]
        if self.norm_inputs:
            spectra = [dp.Normalize(spectra[i], self.spectra_statistic[i], norm_type=self.norm_type).norm() for i in range(len(spectra))]
        spectra = torch.cat(spectra, dim=1) #
        self.pred_params = self._predict(spectra, in_type=in_type)
        if self.norm_target:
            if self.independent_norm:
                for i in range(self.params_n):
                    self.pred_params[:,i] = dp.InverseNormalize(self.pred_params[:,i], self.params_statistic[i], norm_type=self.norm_type).inverseNorm()
            else:
                self.pred_params = dp.InverseNormalize(self.pred_params, self.params_statistic, norm_type=self.norm_type).inverseNorm()
        if self.scale_params:
            self.pred_params = self.inverseScaling(self.pred_params)
        return self.pred_params
    
    def predict_chain(self, obs_data, cov_matrix=None, chain_leng=10000):
        # obs_data: observational spectrum in a list [spectra1, spectra2, ...], each element has shape (N, 3)
        # torch.manual_seed(1000)#
        if cov_matrix is None:
            cov_matrix = [None for i in range(len(obs_data))]
        obs_data = [dp.numpy2torch(obs_data[i]) for i in range(len(obs_data))]
        obs_best = [obs_data[i][:,1] for i in range(len(obs_data))]
        obs_errors = [obs_data[i][:,2] for i in range(len(obs_data))]
        obs_best_multi = [torch.ones((chain_leng, len(obs_best[i]))) * obs_best[i] for i in range(len(obs_data))]
        cholesky_factor = []
        for i in range(len(obs_data)):
            if cov_matrix[i] is None:
                cholesky_factor.append(None)
            else:
                cholesky_factor.append(dp.numpy2torch(np.linalg.cholesky(cov_matrix[i])))
        obs_best_multi = ds.AddGaussianNoise(obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisySpectra()
        self.chain = self.predict(obs_best_multi, in_type='torch')
        self.chain = self.cut_params(self.chain) #remove non-physical parameters
        return self.chain

