# -*- coding: utf-8 -*-

from . import sequence as seq
from . import nodeframe
import torch
import torch.nn as nn
import numpy as np


#%% fully connected single network
class FcNet(torch.nn.Module):
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
    activation_func : str
        Activation function.
    """
    def __init__(self, node_in=2000, node_out=6, hidden_layer=3, nodes=None, activation_func='rrelu'):
        super(FcNet, self).__init__()
        if nodes is None:
            nodes = nodeframe.decreasingNode(node_in=node_in, node_out=node_out, hidden_layer=hidden_layer, get_allNode=True)
        self.fc = seq.LinearSeq(nodes, mainActive=activation_func, finalActive='None', mainBN=True, finalBN=False, mainDropout='None', finalDropout='None').get_seq()
        
    def forward(self, x):
        x = self.fc(x)
        return x

#%% multibranch network
def split_nodes(nodes, weight=[]):
    nodes_new = [[] for i in range(len(weight))]
    for i in range(len(weight)):
        for j in range(len(nodes)):
            nodes_new[i].append(round(nodes[j]*weight[i]))
    return nodes_new


class MultiBranchFcNet(nn.Module):
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
    """
    def __init__(self, nodes_in=[100,100,20], node_out=6, branch_hiddenLayer=1, trunk_hiddenLayer=3, 
                 nodes_all=None, activation_func='rrelu'):
        super(MultiBranchFcNet, self).__init__()
        if nodes_all is None:
            
            # method 1
            nodes_all = []
            branch_outs = []
            fc_hidden = branch_hiddenLayer*2 + 1
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1 #also works, but not necessary
            for i in range(len(nodes_in)):
                fc_node = nodeframe.decreasingNode(node_in=nodes_in[i], node_out=node_out, hidden_layer=fc_hidden, get_allNode=True)
                branch_node = fc_node[:branch_hiddenLayer+2]
                nodes_all.append(branch_node)
                branch_outs.append(branch_node[-1])
            nodes_all.append(nodeframe.decreasingNode(node_in=sum(branch_outs), node_out=node_out, hidden_layer=trunk_hiddenLayer, get_allNode=True))
            
            
            # #method 2
            # nodes_all = []
            # branch_outs = []
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1
            # fc_hidd_node = nodeframe.decreasingNode(node_in=sum(nodes_in), node_out=node_out, hidden_layer=fc_hidden, get_allNode=False)
            # fc_hidd_node_split = split_nodes(fc_hidd_node[:branch_hiddenLayer+1], weight=[nodes_in[i]/sum(nodes_in) for i in range(len(nodes_in))])
            # for i in range(len(nodes_in)):
            #     branch_node = [nodes_in[i]] + fc_hidd_node_split[i]
            #     nodes_all.append(branch_node)
            #     branch_outs.append(branch_node[-1])
            # trunk_node = [sum(branch_outs)] + list(fc_hidd_node[branch_hiddenLayer+1:]) + [node_out]
            # nodes_all.append(trunk_node)


            # #method 3
            # nodes_all = []
            # nodes_comb = []
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1
            # for i in range(len(nodes_in)):
            #     fc_node = nodeframe.decreasingNode(node_in=nodes_in[i], node_out=node_out, hidden_layer=fc_hidden, get_allNode=True)
            #     branch_node = fc_node[:branch_hiddenLayer+2]
            #     nodes_all.append(branch_node)
            #     nodes_comb.append(fc_node[branch_hiddenLayer+1:-1])
            # trunk_node = list(np.sum(np.array(nodes_comb), axis=0)) + [node_out]
            # nodes_all.append(trunk_node)
            
            
        self.branch_n = len(nodes_all) - 1
        for i in range(self.branch_n):
            exec("self.branch%s = seq.LinearSeq(nodes_all[i],mainActive=activation_func,finalActive=activation_func,mainBN=True,finalBN=True,mainDropout='None',finalDropout='None').get_seq()"%(i+1))
        self.trunk = seq.LinearSeq(nodes_all[-1],mainActive=activation_func,finalActive='None',mainBN=True,finalBN=False,mainDropout='None',finalDropout='None').get_seq()
        
    def forward(self, x_all):
        x1 = self.branch1(x_all[0])
        x_comb = x1
        for i in range(1, self.branch_n-1+1):
            x_n = eval('self.branch%s(x_all[i])'%(i+1))#Note:i & i+1
            x_comb = torch.cat((x_comb, x_n),1)
        x = self.trunk(x_comb)
        return x
