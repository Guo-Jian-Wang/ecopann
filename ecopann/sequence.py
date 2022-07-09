# -*- coding: utf-8 -*-

from . import element
import torch.nn as nn
import collections
from itertools import repeat


class SeqName(object):
    """The name of sequence, to be used by class :class:`~LinearSeq`.
    """
    def __init__(self, module_name):
        self.moduleName = module_name
    
    def seq_name(self):
        self.moduleName = str(eval(self.moduleName)+1)
        return self.moduleName

class BatchNorm(object):
    """
    Batch Normalization, to be used by class :class:`~LinearSeq`.
    """
    def _batchnorm1d(self, name, n_output):
        self.seq.add_module(name, nn.BatchNorm1d(n_output, eps=self.eps, momentum=self.momentum))
    
    def _batchnorm2d(self, name, out_channel):
        self.seq.add_module(name, nn.BatchNorm2d(out_channel, eps=self.eps, momentum=self.momentum))
    
    def _batchnorm3d(self, name, out_channel):
        self.seq.add_module(name, nn.BatchNorm3d(out_channel, eps=self.eps, momentum=self.momentum))

class Activation(object):
    """Activation functions, to be used by class :class:`~LinearSeq`.
    """
    def _activation(self, module_name, activation_name):
        self.seq.add_module(module_name, element.activation(activation_name=activation_name))

class Pooling(object):
    """ Pooling, to be used by classes LinearSeq & Conv2dSeq """
    def _pooling(self, module_name, pool_name, kernel_size=2):
        self.seq.add_module(module_name, element.pooling(pool_name=pool_name, kernel_size=kernel_size))

class Dropout(object):
    """Dropout, to be used by class :class:`~LinearSeq`.
    """
    def _dropout(self, module_name, dropout_name):
        self.seq.add_module(module_name, element.get_dropout(dropout_name))

class LinearSeq(SeqName,BatchNorm,Activation,Dropout):
    """Sequence of Linear.
    
    Parameters
    ----------
    nodes : list
        A list that contains the number of nodes in each layer.
    mainBN : bool, optional
        If True, the network will contain batch normalization layer in its main part. Default: True
    finalBN : bool, optional
        If True, the network will contain batch normalization layer in its last layer. Default: False
    mainActive : str, optional
        The activation function used in the main part of the network. See :func:`~.element.activation` for the available activation functions. Default: 'rrelu'
    finalActive : str, optional
        The activation function used in the last layer of the network. See :func:`~.element.activation` for the available activation functions. Default: 'None'
    mainDropout : str, optional
        The dropout used in the main part of the network, 'None' or 'dropout'. Default: 'None'
    finalDropout : str, optional
        The dropout used in the final layer of the network, 'None' or 'dropout'. Default: 'None'    
    """
    def __init__(self, nodes, mainBN=True, finalBN=False, mainActive='rrelu',
                 finalActive='None', mainDropout='None', finalDropout='None'):
        SeqName.__init__(self, '-1') #or super(LinearSeq, self).__init__('-1')
        self.nodes = nodes
        self.layers = len(nodes) - 1
        self.mainBN = mainBN
        self.finalBN = finalBN
        self.mainActive = mainActive
        self.finalActive = finalActive
        self.mainDropout = mainDropout
        self.finalDropout = finalDropout
        self.eps = 1e-05
        self.momentum = 0.1
        self.seq = nn.Sequential()

    def __linear(self, name, n_input, n_output):
        self.seq.add_module(name, nn.Linear(n_input, n_output))
    
    def get_seq(self):
        for i in range(self.layers-1):
            self.__linear(self.seq_name(), self.nodes[i], self.nodes[i+1])
            if self.mainBN:
                self._batchnorm1d(self.seq_name(), self.nodes[i+1])
            if self.mainActive!='None':
                self._activation(self.seq_name(), self.mainActive)
            if self.mainDropout!='None':
                self._dropout(self.seq_name(), self.mainDropout)
        
        self.__linear(self.seq_name(), self.nodes[-2], self.nodes[-1])
        if self.finalBN:
            self._batchnorm1d(self.seq_name(), self.nodes[-1])
        if self.finalActive!='None':
            self._activation(self.seq_name(), self.finalActive)
        if self.finalDropout!='None':
            self._dropout(self.seq_name(), self.finalDropout)
        return self.seq

