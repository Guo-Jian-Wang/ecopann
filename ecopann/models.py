# -*- coding: utf-8 -*-

from . import data_processor as dp
from . import data_simulator as ds
from . import train, evaluate, optimize, fcnet, hpmodel, nodeframe, utils
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
import numpy as np


#%% multilayer perceptron (MLP)
class MLP(train.Train):
    """Multilayer perceptron (MLP) that is used to predict cosmological parameters with one set of datasets.
        
    Parameters
    ----------
    spectra : array-like
        The simulated observational spectra (data) with shape (N, spectra_length).
    parameters : array-like
        The simulated parameters of a specific cosmological (or theoretical) model.
    param_names : list
        A list which contains the parameter names, e.g. ['H0','Omega_m','ombh2','omch2','tau','As','ns'].
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the base values, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    obs_errors : array-like, optional
        Observational errors with shape (spectra_length,). Default: None
    cov_matrix : array-like or None, optional
        Covariance matrix of the observational data. Default: None
    hidden_layer : int, optional
        The number of the hidden layer of the network. Default: 3
    hp_model : str, optional
        Hyperparameter model that contains hyperparameters (such as activation function, batch normalization, dropout, etc.) used in the network.
        It can be 'eco_1', 'eco_2', 'eco_3', 'eco_4', 'eco_5', 'eco_6', or 'eco_7' (see :func:`~.hpmodel.models`). Default: 'eco_3'
    loss_func : str, optional
        The loss function used in the network. Default: 'L1'
    noise_type : str, optional
        The type of Gaussian noise added to the training set, 'singleSigma' or 'multiSigma'. Default: 'multiSigma'
    noise_sigma : float, optional
        For the case of 'singleSigma', it is the standard deviation of the Gaussian noise, while for the case of 'multiSigma' it is 
        a coefficient of the standard deviation. Default: 0.5
    multi_noise : int, optional
        The number of realization of noise added to a spectrum. Default: 5
    
    Attributes
    ----------
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
    norm_inputs : bool, optional
        If True, the input data of the network will be normalized. Default: True
    norm_target : bool, optional
        If True, the target data (cosmological parameters) will be normalized. Default: True
    norm_type : str, optional
        The method of normalization, 'z_score', 'minmax', or 'mean' (see :class:`~.data_processor.Normalize`). Default: 'z_score'
    spaceSigma_min : int, optional
        The minimum parameter space to be learned, e.g. for spaceSigma_min=5, the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    transfer_learning : bool, optional
        If True, the network will be initialized using the well-trained network of the previous step. Default: False
    """
    def __init__(self, spectra, parameters, param_names, params_dict=None, obs_errors=None, cov_matrix=None,
                 hidden_layer=3, hp_model='eco_3', loss_func='L1',
                 noise_type='multiSigma', noise_sigma=0.5, multi_noise=5):
        #data
        self.spectra = spectra
        self.params = dp.ParamsScaling(parameters, param_names, params_dict=params_dict).scaling()
        self.param_names = param_names
        self.params_dict = params_dict
        self.obs_errors = obs_errors
        self.cov_matrix = cov_matrix
        #ANN model
        self.hidden_layer = hidden_layer
        self.hp_model = hp_model
        self.loss_func = train.loss_funcs(name=loss_func)
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
        self.noise_sigma = noise_sigma
        self.multi_noise = multi_noise
        self.norm_inputs = True
        self.norm_target = True
        self.norm_type = 'z_score'
        #training
        self.spaceSigma_min = 5
        self.auto_repeat_n = False
        self.burn_in = False
        self.burnIn_step = None
        self.transfer_learning = False
        self.randn_num = round(abs(np.random.randn()/5.), 5)
        
    def _nodes(self):
        self.node_in = self.spectra.shape[1]
        self.node_out = self.params.shape[1]
        return nodeframe.decreasingNode(node_in=self.node_in,node_out=self.node_out,hidden_layer=self.hidden_layer)
    
    def _net(self):
        if self.fix_initialize:
            torch.manual_seed(1000) #Fixed parameter initialization
        self.nodes = self._nodes()
        self.hparams = hpmodel.models(self.hp_model)
        self.net = fcnet.get_FcNet(nodes=self.nodes, hparams=self.hparams)
        if self.print_info:
            print(self.net)
    
    def _check_batchSize(self):
        if self.batch_size > len(self.params):
            self.batch_size = len(self.params)
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
        self.spectra_statistic = dp.Statistic(self.spectra).statistic()
        self.params_statistic = dp.Statistic(self.params).statistic()

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
            if self.cov_matrix is None:
                self.obs_errors = dp.numpy2cuda(self.obs_errors)
            else:
                self.cov_matrix = dp.numpy2cuda(self.cov_matrix)
        else:
            self.spectra = dp.numpy2torch(self.spectra)
            self.params = dp.numpy2torch(self.params)
            if self.cov_matrix is None:
                self.obs_errors = dp.numpy2torch(self.obs_errors)
            else:
                self.cov_matrix = dp.numpy2torch(self.cov_matrix)
    
    def train(self, repeat_n=3, showIter_n=100):
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
        self.iteration = self.multi_noise*len(self.spectra)//self.batch_size * repeat_n
        
        self.statistic()
        self.transfer_data()
                
        self.loss = []
        # np.random.seed(1000)#
        print('randn_num: %s'%self.randn_num)
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddMultiGaussianNoise(self.spectra,self.params,obs_errors=self.obs_errors,cov_matrix=self.cov_matrix,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(noise_type=self.noise_type,sigma=self.noise_sigma,reorder=True)
            
            if self.norm_inputs:
                self.inputs = dp.Normalize(self.inputs, self.spectra_statistic, norm_type=self.norm_type).norm()
            if self.norm_target:
                self.target = dp.Normalize(self.target, self.params_statistic, norm_type=self.norm_type).norm()

            _, loss_ = self.train_1(self.inputs, self.target, repeat_n=1, set_seed=False, lr_decay=False, print_info=False)            
            self.loss += loss_
            if subsample_num%showIter_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_[-1], self.optimizer.param_groups[0]['lr']))
            lrdc = optimize.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[0]['lr'] = lrdc.exp()
                    
        self.net = self.net.cpu()
        self.loss = np.array(self.loss)
        return self.net, self.loss
    
    def predict(self, spectra, in_type='numpy'):
        if len(spectra.shape)==1:
            spectra = spectra.reshape(1, -1) #for one spectrum
        if self.norm_inputs:
            spectra = dp.Normalize(spectra, self.spectra_statistic, norm_type=self.norm_type).norm()
        self.pred_params = evaluate.predict(self.net, spectra, in_type=in_type)
        if self.norm_target:
            self.pred_params = dp.InverseNormalize(self.pred_params, self.params_statistic, norm_type=self.norm_type).inverseNorm()
        self.pred_params = dp.ParamsScaling(self.pred_params, self.param_names, params_dict=self.params_dict).inverseScaling()
        return self.pred_params
    
    def predict_chain(self, obs_spectra, cov_matrix=None, chain_leng=10000):
        # torch.manual_seed(1000)#
        obs_spectra = dp.numpy2torch(obs_spectra)
        obs_best, obs_errors = obs_spectra[:,1], obs_spectra[:,2]
        obs_best_multi = torch.ones((chain_leng, len(obs_best))) * obs_best
        if cov_matrix is not None:
            cov_matrix = dp.numpy2torch(cov_matrix)
        obs_best_multi = ds.AddGaussianNoise(obs_best_multi, obs_errors=obs_errors, cov_matrix=cov_matrix, use_GPU=False).singleSigma(sigma=1)
        self.chain = self.predict(obs_best_multi, in_type='torch')
        return self.chain

    def save_net(self, path='ann', sample=None):
        if sample is None:
            fileName = 'net_nodes%s_train%s_batch%s_epoch%s_%s.pt'%(str(self.nodes),len(self.params),self.batch_size,self.epoch,self.randn_num)
        else:
            fileName = 'net-%s_nodes%s_train%s_batch%s_epoch%s_%s.pt'%(sample,str(self.nodes),len(self.params),self.batch_size,self.epoch,self.randn_num)
        utils.saveTorchPt(path+'/net', fileName, self.net)
    
    def save_loss(self, path='ann', sample=None):
        if sample is None:
            fileName = 'loss_nodes%s_train%s_batch%s_epoch%s_%s'%(str(self.nodes),len(self.params),self.batch_size,self.epoch,self.randn_num)
        else:
            fileName = 'loss-%s_nodes%s_train%s_batch%s_epoch%s_%s'%(sample,str(self.nodes),len(self.params),self.batch_size,self.epoch,self.randn_num)
        utils.savenpy(path+'/net', fileName, self.loss)
    
    def save_chain(self, path='ann', sample=None):
        if sample is None:
            fileName = 'chain_nodes%s_train%s_batch%s_epoch%s_%s'%(str(self.nodes),len(self.params),self.batch_size,self.epoch,self.randn_num)
        else:
            fileName = 'chain-%s_nodes%s_train%s_batch%s_epoch%s_%s'%(sample,str(self.nodes),len(self.params),self.batch_size,self.epoch,self.randn_num)
        utils.savenpy(path+'/chains', fileName, self.chain)
    
    def save_hparams(self, path='ann', sample=None):
        if sample is None:
            fileName = 'hparams_nodes%s_train%s_batch%s_epoch%s_%s'%(str(self.nodes),len(self.params),self.batch_size,self.epoch,self.randn_num)
        else:
            fileName = 'hparams-%s_nodes%s_train%s_batch%s_epoch%s_%s'%(sample,str(self.nodes),len(self.params),self.batch_size,self.epoch,self.randn_num)
        utils.savenpy(path+'/hparams', fileName, [self.spectra_statistic, self.params_statistic, self.param_names, self.burnIn_step])
    
    def plot_loss(self):
        evaluate.plot_loss(self.loss)

class RePredictMLP(MLP):
    """Repredict cosmological parameters using the saved networks.
    
    Parameters
    ----------
    path : str
        The path of the results saved. Default: 'ann'
    randn_num : str or int
        A random number that identifies the saved results.
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the base values, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None

    Attributes
    ----------
    norm_inputs : bool, optional
        If True, the input data of the network will be normalized. Default: True
    norm_target : bool, optional
        If True, the target data (cosmological parameters) will be normalized. Default: True
    norm_type : str, optional
        The method of normalization, 'z_score', 'minmax', or 'mean' (see :class:`~.data_processor.Normalize`). Default: 'z_score'
    """
    def __init__(self, path='ann', randn_num='0.123', params_dict=None):
        self.path = path
        self.randn_num = str(randn_num)
        self.params_dict = params_dict
        self.norm_inputs = True
        self.norm_target = True
        self.norm_type = 'z_score'
    
    def load_net(self):
        file_path = evaluate.FilePath(filedir=self.path+'/net', randn_num=self.randn_num).filePath()
        self.net = torch.load(file_path)
    
    def load_loss(self):
        file_path = evaluate.FilePath(filedir=self.path+'/net', randn_num=self.randn_num, suffix='.npy').filePath()
        self.loss = np.load(file_path)

    def load_chain(self):
        file_path = evaluate.FilePath(filedir=self.path+'/chains', randn_num=self.randn_num, suffix='.npy').filePath()
        self.chain = np.load(file_path)
    
    def load_hparams(self):
        file_path = evaluate.FilePath(filedir=self.path+'/hparams', randn_num=self.randn_num, suffix='.npy').filePath()
        self.spectra_statistic, self.params_statistic, self.param_names, self.burnIn_step = np.load(file_path, allow_pickle=True)


#%% multibranch network
class MultiBranchNet(MLP):
    """Multibranch network that is used to predict cosmological parameters with multiple sets of datasets.
        
    Parameters
    ----------
    spectra : list
        Simulated observational spectra (data), it is a list spectra with shape [(N,spectra_length_1), (N,spectra_length_2), ...].
    parameters : array-like
        The simulated parameters of a specific cosmological (or theoretical) model.
    param_names : list
        A list which contains the parameter names, e.g. ['H0','Omega_m','ombh2','omch2','tau','As','ns'].
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the base values, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    obs_errors : list, optional
        Observational errors, it is a list of errors with shape [(spectra_length_1,), (spectra_length_2,), ...]. Default: None
    cov_matrix : list or None, optional
        A list of covariance matrix with shape [(spectra_length_1, spectra_length_1), (spectra_length_2, spectra_length_2), ...].
        If there is no covariance for some observations, the covariance matrix should be set to None. e.g. [cov_matrix_1, None, cov_matrix_3]. Default: None
    branch_hiddenLayer : int, optional
        The number of the hidden layer for the branch part of the network. Default: 2
    trunk_hiddenLayer : int, optional
        The number of the hidden layer for the trunk part of the network. Default: 1
    hp_model : str, optional
        Hyperparameter model that contains hyperparameters (such as activation function, batch normalization, dropout, etc.) used in the network.
        It can be 'eco_1', 'eco_2', 'eco_3', 'eco_4', 'eco_5', 'eco_6', or 'eco_7' (see :func:`~.hpmodel.models`). Default: 'eco_3'
    loss_func : str, optional
        The loss function used in the network. Default: 'L1'
    noise_type : str, optional
        The type of Gaussian noise added to the training set, 'singleSigma' or 'multiSigma'. Default: 'multiSigma'
    noise_sigma : float, optional
        For the case of 'singleSigma', it is the standard deviation of the Gaussian noise, while for the case of 'multiSigma' it is 
        a coefficient of the standard deviation. Default: 0.5
    multi_noise : int, optional
        The number of realization of noise added to a spectrum. Default: 5
    
    Attributes
    ----------
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
    norm_inputs : bool, optional
        If True, the input data of the network will be normalized. Default: True
    norm_target : bool, optional
        If True, the target data (cosmological parameters) will be normalized. Default: True
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
    def __init__(self, spectra, parameters, param_names, params_dict=None, obs_errors=None, cov_matrix=None,
                 branch_hiddenLayer=2, trunk_hiddenLayer=1, hp_model='eco_3', loss_func='L1',
                 noise_type='multiSigma', noise_sigma=0.5, multi_noise=5):
        #data
        self.spectra = spectra
        self.params = dp.ParamsScaling(parameters, param_names, params_dict=params_dict).scaling()
        self.param_names = param_names
        self.params_dict = params_dict
        self.obs_errors = self._obs_errors(obs_errors)
        self.cov_matrix = self._cov_matrix(cov_matrix)
        #ANN model
        self.branch_n = len(spectra)
        self.branch_hiddenLayer = branch_hiddenLayer
        self.trunk_hiddenLayer = trunk_hiddenLayer
        self.hp_model = hp_model
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
        self.noise_sigma = noise_sigma
        self.multi_noise = multi_noise
        self.norm_inputs = True
        self.norm_target = True
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
            return [None for i in range(len(self.spectra))]
        else:
            return errors
    
    def _cov_matrix(self, matrix):
        if matrix is None:
            return [None for i in range(len(self.spectra))]
        else:
            return matrix
    
    def _nodes(self):
        self.nodes_in = []
        self.node_out = self.params.shape[1]
        for i in range(self.branch_n):
            self.nodes_in.append(self.spectra[i].shape[1])
    
    def _net(self):
        if self.fix_initialize:
            torch.manual_seed(1000) #Fixed parameter initialization
        self._nodes()
        self.hparams = hpmodel.models(self.hp_model)
        for i in range(self.branch_n):
            exec('self.branch_net%s=fcnet.get_FcNet(node_in=self.nodes_in[i], node_out=self.node_out,\
            hidden_layer=2*self.branch_hiddenLayer+1, hparams=self.hparams)'%(i+1))
        self.net = fcnet.get_MultiBranchFcNet(nodes_in=self.nodes_in, node_out=self.node_out, branch_hiddenLayer=self.branch_hiddenLayer,
                                              trunk_hiddenLayer=self.trunk_hiddenLayer, nodes_all=None, hparams=self.hparams)
        if self.print_info:
            print(self.net)
    
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
        self.spectra_statistic = [dp.Statistic(self.spectra[i]).statistic() for i in range(self.branch_n)]
        self.params_statistic = dp.Statistic(self.params).statistic()
    
    def transfer_data(self):
        if self.use_GPU:
            self.spectra = [dp.numpy2cuda(self.spectra[i]) for i in range(self.branch_n)]
            self.params = dp.numpy2cuda(self.params)
            for i in range(self.branch_n):
                if self.cov_matrix[i] is None:
                    self.obs_errors[i] = dp.numpy2cuda(self.obs_errors[i])
                else:
                    self.cov_matrix[i] = dp.numpy2cuda(self.cov_matrix[i])
        else:
            self.spectra = [dp.numpy2torch(self.spectra[i]) for i in range(self.branch_n)]
            self.params = dp.numpy2torch(self.params)
            for i in range(self.branch_n):
                if self.cov_matrix[i] is None:
                    self.obs_errors[i] = dp.numpy2torch(self.obs_errors[i])
                else:
                    self.cov_matrix[i] = dp.numpy2torch(self.cov_matrix[i])

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
        self.cov_m = self.cov_matrix[rank]
#        self.transfer_subData(device=device)
        
        print('Training the branch network %s'%(rank+1))
        for subsample_num in range(1, self.epoch_branch+1):
            _inputs, _target = ds.AddMultiGaussianNoise(self.inputs,self.target,obs_errors=self.error,cov_matrix=self.cov_m,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(noise_type=self.noise_type,sigma=self.noise_sigma,reorder=True)
            
            if self.norm_inputs:
                _inputs = dp.Normalize(_inputs, self.spectra_statistic[rank], norm_type=self.norm_type).norm()
            if self.norm_target:
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
            self.inputs, self.target = ds.AddMultiGaussianNoise(self.spectra,self.params,obs_errors=self.obs_errors,cov_matrix=self.cov_matrix,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(noise_type=self.noise_type,sigma=self.noise_sigma,reorder=True)
            
            if self.norm_inputs:
                self.inputs = [dp.Normalize(self.inputs[i], self.spectra_statistic[i], norm_type=self.norm_type).norm() for i in range(self.branch_n)]
            if self.norm_target:
                self.target = dp.Normalize(self.target, self.params_statistic, norm_type=self.norm_type).norm()
            
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
        
        self.loss = []
        print('\nTraining the multibranch network')
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddMultiGaussianNoise(self.spectra,self.params,obs_errors=self.obs_errors,cov_matrix=self.cov_matrix,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(noise_type=self.noise_type,sigma=self.noise_sigma,reorder=True)
            
            if self.norm_inputs:
                self.inputs = [dp.Normalize(self.inputs[i], self.spectra_statistic[i], norm_type=self.norm_type).norm() for i in range(self.branch_n)]
            if self.norm_target:
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
            lrdc_t = optimize.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[-1]['lr'] = lrdc_t.exp()
            lrdc_b = optimize.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr_branch,lr_min=self.lr_min)#change to lr=self.lr ?
            for i in range(self.branch_n):
                self.optimizer.param_groups[i]['lr'] = lrdc_b.exp()
        
        self.net = self.net.cpu()
        self.loss = np.array(self.loss)
        return self.net, self.loss
    
    def _predict(self, net, inputs, in_type='numpy'):
        # in_type: 'numpy' or 'torch'
        net = net.eval() #this works for the batch normalization layers
        if in_type=='numpy':
            inputs = [dp.numpy2torch(inputs[i]) for i in range(len(inputs))]
        inputs = [Variable(inputs[i]) for i in range(len(inputs))]
        pred = net(inputs)
        pred = dp.torch2numpy(pred.data)
        return pred
    
    def predict(self, spectra, in_type='numpy'):
        # spectra: [spectra1, spectra2, ...]
        if len(spectra[0].shape)==1:
            spectra = [spectra[i].reshape(1, -1) for i in range(len(spectra))] #for one spectrum
        if self.norm_inputs:
            spectra = [dp.Normalize(spectra[i], self.spectra_statistic[i], norm_type=self.norm_type).norm() for i in range(len(spectra))]
        pred_params = self._predict(self.net, spectra, in_type=in_type)
        if self.norm_target:
            self.pred_params = dp.InverseNormalize(pred_params, self.params_statistic, norm_type=self.norm_type).inverseNorm()
        self.pred_params = dp.ParamsScaling(self.pred_params, self.param_names, params_dict=self.params_dict).inverseScaling()
        return self.pred_params
    
    def predict_chain(self, obs_spectra, cov_matrix=None, chain_leng=10000):
        # obs_spectra: observational spectrum in a list [spectra1, spectra2, ...], each element has shape (N, 3)
        # torch.manual_seed(1000)#
        if cov_matrix is None:
            cov_matrix = [None for i in range(len(obs_spectra))]
        obs_spectra = [dp.numpy2torch(obs_spectra[i]) for i in range(len(obs_spectra))]
        obs_best = [obs_spectra[i][:,1] for i in range(len(obs_spectra))]
        obs_errors = [obs_spectra[i][:,2] for i in range(len(obs_spectra))]
        obs_best_multi = [torch.ones((chain_leng, len(obs_best[i]))) * obs_best[i] for i in range(len(obs_spectra))]
        for i in range(len(obs_spectra)):
            if cov_matrix[i] is not None:
                cov_matrix[i] = dp.numpy2torch(cov_matrix[i])
        obs_best_multi = ds.AddGaussianNoise(obs_best_multi, obs_errors=obs_errors, cov_matrix=cov_matrix, use_GPU=False).singleSigma(sigma=1)
        self.chain = self.predict(obs_best_multi, in_type='torch')
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
        utils.savenpy(path+'/net', fileName, self.loss)
    
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
        utils.savenpy(path+'/hparams', fileName, [self.spectra_statistic, self.params_statistic, self.param_names, self.burnIn_step])

class RePredictMBNet(MultiBranchNet):
    """Repredict cosmological parameters using the saved networks.
    
    Parameters
    ----------
    path : str
        The path of the results saved. Default: 'ann'
    randn_num : str or int
        A random number that identifies the saved results.
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the base values, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    
    Attributes
    ----------
    norm_inputs : bool, optional
        If True, the input data of the network will be normalized. Default: True
    norm_target : bool, optional
        If True, the target data (cosmological parameters) will be normalized. Default: True
    norm_type : str, optional
        The method of normalization, 'z_score', 'minmax', or 'mean' (see :class:`~.data_processor.Normalize`). Default: 'z_score'
    """
    def __init__(self, path='ann', randn_num='0.123', params_dict=None):
        self.path = path
        self.randn_num = str(randn_num)
        self.params_dict = params_dict
        self.norm_inputs = True
        self.norm_target = True
        self.norm_type = 'z_score'
    
    def load_net(self):
        file_path = evaluate.FilePath(filedir=self.path+'/net', randn_num=self.randn_num).filePath()
        self.net = torch.load(file_path)
    
    def load_loss(self):
        file_path = evaluate.FilePath(filedir=self.path+'/net', randn_num=self.randn_num, suffix='.npy').filePath()
        self.loss = np.load(file_path)
    
    def load_chain(self):
        file_path = evaluate.FilePath(filedir=self.path+'/chains', randn_num=self.randn_num, suffix='.npy').filePath()
        self.chain = np.load(file_path)

    def load_hparams(self):
        file_path = evaluate.FilePath(filedir=self.path+'/hparams', randn_num=self.randn_num, suffix='.npy').filePath()
        self.spectra_statistic, self.params_statistic, self.param_names, self.burnIn_step = np.load(file_path, allow_pickle=True)

#!!!
# 1. check how to reduce training time, when using multi_noise
# 2. for unilateral distribution, the result is not good, try to solve this
# 3. whether to use transfer learning?
