# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

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

def elu():
    return nn.ELU(inplace=True)

class ELU_1(nn.Module):
    def __init__(self):
        super(ELU_1, self).__init__()

    def forward(self, x):
        x = F.elu(x)
        x = x + 1
        return x
def elu_1():
    return ELU_1()

def celu():
    return nn.CELU(inplace=True)

def selu():
    return nn.SELU(inplace=True)

def silu():
    return nn.SiLU(inplace=True)

def sigmoid():
    return nn.Sigmoid()

def logsigmoid():
    return nn.LogSigmoid()

def tanh():
    return nn.Tanh()

def tanhshrink():
    return nn.Tanhshrink()

def softsign():
    return nn.Softsign()

def softplus():
    return nn.Softplus()

class Softplus_1(nn.Module):
    def __init__(self):
        super(Softplus_1, self).__init__()

    def forward(self, x):
        x = F.softplus(x)
        x = x - 1
        return x
def softplus_1():
    return Softplus_1()

class Softplus_2(nn.Module):
    def __init__(self):
        super(Softplus_2, self).__init__()

    def forward(self, x):
        x = F.softplus(x)
        x = x - 2
        return x
def softplus_2():
    return Softplus_2()

class Sigmoid_1(nn.Module):
    def __init__(self):
        super(Sigmoid_1, self).__init__()

    def forward(self, x):
        x = F.sigmoid(x)
        x = x - 0.5
        return x
def sigmoid_1():
    return Sigmoid_1()
    
def activation(activation_name='rrelu'):
    """Activation functions.
    
    Parameters
    ----------
    activation_name : str, optional
        The name of activation function, which can be 'relu', 'leakyrelu', 'prelu', 'rrelu', 
        'relu6', 'elu', 'celu', 'selu', 'silu', 'sigmoid', 'logsigmoid', 'tanh', 'tanhshrink', 'softsign', or 'softplus'. Default: 'rrelu'

    Returns
    -------
    object
        Activation functions.
    
    Note
    ----
    Although many activation functions are available, the recommended activation function is 'rrelu'.
    """
    return eval('%s()'%activation_name)

#%% Pooling
def maxPool1d(kernel_size):
    return nn.MaxPool1d(kernel_size)

def maxPool2d(kernel_size):
    return nn.MaxPool2d(kernel_size)

def maxPool3d(kernel_size):
    return nn.MaxPool3d(kernel_size)

def avgPool1d(kernel_size):
    return nn.AvgPool1d(kernel_size)

def avgPool2d(kernel_size):
    return nn.AvgPool2d(kernel_size)

def avgPool3d(kernel_size):
    return nn.AvgPool3d(kernel_size)

def pooling(pool_name='maxPool2d', kernel_size=2):
    return eval('%s(kernel_size)'%pool_name)

#%% Dropout
def dropout():
    return nn.Dropout(inplace=False)

def dropout2d():
    return nn.Dropout2d(inplace=False)

def dropout3d():
    return nn.Dropout3d(inplace=False)

def get_dropout(drouput_name='dropout'):
    """Get the dropout."""
    return eval('%s()'%drouput_name)
