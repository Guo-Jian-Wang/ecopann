# -*- coding: utf-8 -*-

from . import sequence as seq
from . import nodeframe, hpmodel
import torch
import torch.nn as nn


#%% fully connected single network
class FcNet(torch.nn.Module):
    def __init__(self, nodes, mainActive='rrelu', finalActive='None', mainBN=False,
                 finalBN=False, mainDropout='None', finalDropout='None'):
        super(FcNet, self).__init__()
        self.fc = seq.LinearSeq(nodes,mainActive=mainActive,finalActive=finalActive,mainBN=mainBN,
                                finalBN=finalBN,mainDropout=mainDropout,finalDropout=finalDropout).get_seq()
        
    def forward(self, x):
        x = self.fc(x)
        return x

def get_FcNet(node_in=2000, node_out=6, hidden_layer=3, nodes=None, hparams={}):
    """Get a fully connected network.
    
    Parameters
    ----------
    node_in : int
        The number of the input nodes.
    node_out : int
        The number of the output nodes.
    hidden_layer : int
        The number of the hidden layers.
    nodes : None or list
            If list, it should be a collection of nodes of the network, e.g. [node_in, node_hidden1, node_hidden2, ..., node_out]
    hparams : dict
        A dictionary of hyperparameters (or hidden parameters, such as the activation function,
        the batch normalization, and the dropout) of the netwowrk. See :func:`~.hpmodel.models`.
    """
    if nodes is None:
        nodes = nodeframe.decreasingNode(node_in=node_in, node_out=node_out, hidden_layer=hidden_layer, get_allNode=True)
    
    if hparams:
        hp = hparams
    else:
        hp = hpmodel.models('default')
        print ('Using the default hyperparameters, you can also select another set of hyperparameters \n')
    
    hp = hpmodel.nuisance_hp(hp)
    net = FcNet(nodes,mainActive=hp['active'],finalActive=hp['finalActive'],mainBN=hp['BN'],
                finalBN=hp['finalBN'],mainDropout=hp['dropout'],finalDropout=hp['finalDropout'])
#    print ('Nodes: %s'%nodes)
    return net

#%% multibranch network
class MultiBranchFcNet(nn.Module):
    def __init__(self, nodes_all, mainActive='rrelu', finalActive='None', mainBN=False, finalBN=False,
                 mainDropout='None', finalDropout='None'):
        super(MultiBranchFcNet, self).__init__()
        self.branch_n = len(nodes_all) - 1
        for i in range(self.branch_n):
            exec('self.branch%s = seq.LinearSeq(nodes_all[i],mainActive=mainActive,finalActive=mainActive,mainBN=mainBN,finalBN=mainBN,mainDropout=mainDropout,finalDropout=mainDropout).get_seq()'%(i+1))
        self.trunk = seq.LinearSeq(nodes_all[-1],mainActive=mainActive,finalActive=finalActive,mainBN=mainBN,finalBN=finalBN,mainDropout=mainDropout,finalDropout=finalDropout).get_seq()
        
    def forward(self, x_all):
        x1 = self.branch1(x_all[0])
        x_comb = x1
        for i in range(1, self.branch_n-1+1):
            x_n = eval('self.branch%s(x_all[i])'%(i+1))#Note:i & i+1
            x_comb = torch.cat((x_comb, x_n),1)
        x = self.trunk(x_comb)
        return x

def get_MultiBranchFcNet(nodes_in=[100,100,20], node_out=6, branch_hiddenLayer=1, trunk_hiddenLayer=3, nodes_all=None, hparams={}):
    """Get a multibranch network.

    Parameters
    ----------
    nodes_in : list
        The number of the input nodes for each branch. e.g. [node_in_branch1, node_in_branch2, ...]
    node_out : int
        The number of the output nodes.
    branch_hiddenLayer : int
        The number of the hidden layers for the branch part.
    trunk_hiddenLayer : int
        The number of the hidden layers for the trunk part.
    nodes_all : list
        The number of nodes of the multibranch network. e.g. [nodes_branch1, nodes_branch2, ..., nodes_trunk]
    hparams : dict
        A dictionary of hyperparameters (or hidden parameters, such as the activation function,
        the batch normalization, and the dropout) of the netwowrk. See :func:`~.hpmodel.models`.
    """
    if nodes_all is None:
        nodes_all = []
        branch_outs = []
        fc_hidden = 2 * branch_hiddenLayer + 1
        for i in range(len(nodes_in)):
            fc_node = nodeframe.decreasingNode(node_in=nodes_in[i], node_out=node_out, hidden_layer=fc_hidden, get_allNode=True)
            branch_node = fc_node[:branch_hiddenLayer+2]
            nodes_all.append(branch_node)
            branch_outs.append(branch_node[-1])
        nodes_all.append(nodeframe.decreasingNode(node_in=sum(branch_outs), node_out=node_out, hidden_layer=trunk_hiddenLayer, get_allNode=True))
    
    if hparams:
        hp = hparams
    else:
        hp = hpmodel.models('default')
        print ('Using the default hyperparameters, you can also select another set of hyperparameters \n')
    
    hp = hpmodel.nuisance_hp(hp)
    net = MultiBranchFcNet(nodes_all,mainActive=hp['active'],finalActive=hp['finalActive'],mainBN=hp['BN'],
                           finalBN=hp['finalBN'],mainDropout=hp['dropout'],finalDropout=hp['finalDropout'])
    # print ('Nodes: %s'%nodes_all)
    return net
