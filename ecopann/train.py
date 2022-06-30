# -*- coding: utf-8 -*-

from . import optimize
from . import data_processor as dp
import torch
from torch.autograd import Variable
import numpy as np
import itertools


def loss_funcs(name='L1'):
    """Some loss functions.
    
    Parameters
    ----------
    name : str
        Abbreviation of loss function name. 'L1', 'MSE', or 'SmoothL1'. Default: 'L1'.

    Returns
    -------
    object
        The corresponding loss function.
    """
    if name=='L1':
        lf = torch.nn.L1Loss()
    elif name=='MSE':
        lf = torch.nn.MSELoss()
    elif name=='SmoothL1':
        lf = torch.nn.SmoothL1Loss()
    return lf

#update
class SimilarityIdx(object):
    def __init__(self, param_nums):
        self.param_nums = param_nums
            
    def simiIdx(self):
        """ calculate the parameter index surrounding a specific parameter """
        self.indexes = []
        for i in range(self.param_nums):
            idx_remain = [p for p in range(self.param_nums)]
            idx_remain.remove(i)
            self.indexes.append(idx_remain)
        return self.indexes
    
    def simiIdx_2(self):
        """ calculate the parameter index surrounding a specific parameter (using combination) """
        comb = np.array([[c[0], c[1]] for c in itertools.combinations(range(self.param_nums), 2)])
        return comb

#update
def additional_loss(predicted, yy, simiIdx, reduction='mean'):
    param_nums = yy.shape[1]
    # print(param_nums, 'pppp', len(simiIdx[1]),predicted[:, [0 for i in range(param_nums)]].shape, predicted[:, simiIdx[0]].shape)
    diff_simi_all = torch.abs( torch.abs(predicted[:, [0 for i in range(param_nums-1)]] - predicted[:, simiIdx[0]]) - torch.abs(yy[:, [0 for i in range(param_nums-1)]] - yy[:, simiIdx[0]]) )
    for p in range(1, param_nums):
        diff_simi_i = torch.abs( torch.abs(predicted[:, [p for i in range(param_nums-1)]] - predicted[:, simiIdx[p]]) - torch.abs(yy[:, [p for i in range(param_nums-1)]] - yy[:, simiIdx[p]]) )
        diff_simi_all = diff_simi_all + diff_simi_i
    
    # print(diff_simi_all.shape, 'shape')
    diff_simi_all = diff_simi_all / param_nums
    if reduction=='mean':
        diff_simi_all = torch.mean(diff_simi_all)
    return diff_simi_all

#update
def additional_loss_2(predicted, yy, simiIdx, reduction='mean'):
    param_nums = yy.shape[1]
    # print(param_nums, 'pppp', len(simiIdx[1]),predicted[:, [0 for i in range(param_nums)]].shape, predicted[:, simiIdx[0]].shape)
    diff_simi_all = torch.abs( torch.abs(predicted[:, simiIdx[:,0]] - predicted[:, simiIdx[:,1]]) - torch.abs(yy[:, simiIdx[:,0]] - yy[:, simiIdx[:,1]]) )
    
    # print(diff_simi_all.shape, 'shape')
    diff_simi_all = diff_simi_all / param_nums
    if reduction=='mean':
        diff_simi_all = torch.mean(diff_simi_all)
    return diff_simi_all

class Train(object):
    """Train the network.
    """
    def __init__(self,net,loss_func='L1',iteration=10000,optimizer='Adam'):
        self.net = net
        self.loss_func = loss_funcs(name=loss_func)
        self.iteration = iteration
        self.lr = 1e-1
        self.lr_min = 1e-6
        self.batch_size = 128
        self.optimizer = self._optimizer(name=optimizer)
    
    def _prints(self, items, prints=True):
        if prints:
            print(items)
    
    def check_GPU(self):
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            device = device_ids[0]
        else:
            device_ids = None
            device = None
        return device_ids, device
    
    def call_GPU(self, prints=True):
        if torch.cuda.is_available():
            self.use_GPU = True
            gpu_num = torch.cuda.device_count()
            if gpu_num > 1:
                self.use_multiGPU = True
                self._prints('\nTraining the network using {} GPUs'.format(gpu_num), prints=prints)
            else:
                self.use_multiGPU = False
                self._prints('\nTraining the network using 1 GPU', prints=prints)
        else:
            self.use_GPU = False
            self._prints('\nTraining the network using CPU', prints=prints)

    #improve to use multiple GPUs?
    # def call_GPU(self, prints=True):
    #     if torch.cuda.is_available():
    #         self.use_GPU = True
    #         self.use_multiGPU = False
    #         self._prints('\nTraining the network using 1 GPU', prints=prints)
    #     else:
    #         self.use_GPU = False
    #         self._prints('\nTraining the network using CPU', prints=prints)
    
    def transfer_net(self, use_DDP=False, device_ids=None, prints=True):
        if device_ids is None:
            device = None
        else:
            device = device_ids[0]
        self.call_GPU(prints=prints)
        if self.use_GPU:
            self.net = self.net.cuda(device=device)
            if self.use_multiGPU:
                if use_DDP:
                    self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=device_ids)
                else:
                    self.net = torch.nn.DataParallel(self.net, device_ids=device_ids)
    
    def transfer_data(self, device=None):
        if self.use_GPU:
            self.inputs = dp.numpy2cuda(self.inputs, device=device)
            self.target = dp.numpy2cuda(self.target, device=device)
        else:
            self.inputs = dp.numpy2torch(self.inputs)
            self.target = dp.numpy2torch(self.target)
    
    def _optimizer(self, name='Adam'):
        if name=='Adam':
            _optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return _optim
    
    def train_0(self, xx, yy, iter_mid, repeat_n=3, lr_decay=True):
        """Training batch samples.

        Parameters
        ----------
        xx : torch tensor
            The input of the network.
        yy : torch tensor
            The target of the network.
        iter_mid : int
            The i-th iteration.
        repeat_n : int, optional
            The number of iterations using the same batch of data during network training,
            which is usually set to 1 or 3. Default: 3
        lr_decay : bool, optional
            If True, the learning rate will decrease with the iteration, otherwise, the learning rate will not change.
        
        Returns
        -------
        float
            The loss.
        Tensor
            The predicted values.
        """
        xx = Variable(xx)
        yy = Variable(yy, requires_grad=False)
        for t in range(repeat_n):
            self._predicted = self.net(xx)
            _loss = self.loss_func(self._predicted, yy) #
            
            # simiIdx = SimilarityIdx(yy.shape[1]).simiIdx() #update - new loss
            # _loss = self.loss_func(_predicted, yy) + additional_loss(_predicted, yy, simiIdx) #update - new loss
            
            # simiIdx = SimilarityIdx(yy.shape[1]).simiIdx_2() #update - new loss 2
            # _loss = self.loss_func(_predicted, yy) + additional_loss_2(_predicted, yy, simiIdx) #update - new loss 2
            
            self.optimizer.zero_grad()
            _loss.backward()
            self.optimizer.step()
        
        if lr_decay:
            #reduce the learning rate
            lrdc = optimize.LrDecay(iter_mid,iteration=self.iteration,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[0]['lr'] = lrdc.exp()
        return _loss.item(), self._predicted.data

    def train_1(self, inputs, target, repeat_n=1, set_seed=False, lr_decay=True,
                print_info=True, showIter_n=200):
        """Training the training set (or a subsample of the training set).
        """        
        if self.batch_size > len(inputs):
            raise ValueError('The batch size should be smaller than the number of the training set')
            
        if set_seed:
            np.random.seed(1000)#
        loss_all = []
        for iter_mid in range(1, self.iteration+1):
            batch_index = np.random.choice(len(inputs), self.batch_size, replace=False)#Note: replace=False
            # batch_index = np.random.choice(len(inputs), self.batch_size, replace=True)#test
            xx = inputs[batch_index]
            yy = target[batch_index]
            
            _loss, _ = self.train_0(xx, yy, iter_mid, repeat_n=repeat_n, lr_decay=lr_decay)
            loss_all.append(_loss)
            
            if print_info:
                if iter_mid%showIter_n==0:
                    print('(iteration:%s/%s; loss:%.5f; lr:%.8f)'%(iter_mid, self.iteration, _loss, self.optimizer.param_groups[0]['lr']))
        return self.net, loss_all
