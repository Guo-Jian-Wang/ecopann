# -*- coding: utf-8 -*-

import torch.nn as nn


#%% activation functions
def relu():
    #here 'inplace=True' is used to save GPU memory
    return nn.ReLU(inplace=True)

def leakyrelu():
    return nn.LeakyReLU(inplace=True)

def prelu():
    return nn.PReLU()

def rrelu():
    return nn.RReLU(inplace=True)

def relu6():
    return nn.ReLU6(inplace=True)

#
def elu():
    return nn.ELU(inplace=True)

def selu():
    return nn.SELU(inplace=True)

#
def sigmoid():
    return nn.Sigmoid()

def tanh():
    return nn.Tanh()

def softsign():
    return nn.Softsign()

def activation(active_name='rrelu'):
    """Activation functions.
    
    Parameters
    ----------
    active_name : str, optional
        The name of activation function, which can be 'relu', 'leakyrelu', 'prelu', 'rrelu', 
        'relu6', 'elu', 'selu', 'sigmoid', 'tanh', or 'softsign'. Default: 'rrelu'

    Returns
    -------
    object
        Activation functions.
    
    Note
    ----
    Although many activation functions are available, the recommended activation function is 'rrelu'.
    """
    return eval('%s()'%active_name)

#%% Dropout
def dropout():
    return nn.Dropout(inplace=False)

def get_dropout(drouput_name='dropout'):
    """Get the dropout."""
    return eval('%s()'%drouput_name)
