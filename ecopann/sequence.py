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


#%% further plans !!!???
def _ntuple(x, n=2):
    '''
    return a tuple
    
    :param x: an integer or a tuple with more than two elements
    :param n: the number to be repeated for an integer, it only works for an integer
    '''
    if isinstance(x, collections.Iterable):
        return x
    else:
        return tuple(repeat(x, n))

def multi_ntuple(x, n=2):
    '''
    return a tuple or a list that contain tuples
    
    :param x: an integer, a tuple or a list whose element is tuple (with more than two elements)
    :param n: the number to be repeated for an integer, it only works for an integer
    '''
    if type(x) is list:
        for i in range(len(x)):
            x[i] = _ntuple(x[i], n=n)
    else:
        x = _ntuple(x, n=n)
    return x

class Conv1dSeq(SeqName,BatchNorm,Activation,Pooling,Dropout):
    def __init__(self, channels, kernels_size=None, strides=None, extra_pads=None, 
                 mainBN=True, finalBN=True, mainActive='relu', finalActive='relu', 
                 mainPool='None', finalPool='None', pool_ks=2, mainDropout='None', finalDropout='None',
                 eps=1e-05, momentum=0.1, transConv1d=False, in_side=512):
        ''' sequence of Conv2d or ConvTranspose2d '''
        
        super(Conv1dSeq, self).__init__('-1') #or SeqName.__init__(self, '-1')
        self.channels = channels
        self.layers = len(channels) - 1
        if kernels_size is None:
            self.kernels_size = [3 for i in range(self.layers)]
        else:
            self.kernels_size = kernels_size
        if strides is None:
            self.strides = [2 for i in range(self.layers)]
        else:
            self.strides = strides
        if extra_pads is None:
            self.extra_pads = [0 for i in range(self.layers)]
        else:
            self.extra_pads = extra_pads
        self.dilation = 1
        self.mainBN = mainBN
        self.finalBN = finalBN
        self.mainActive = mainActive
        self.finalActive = finalActive
        self.mainPool = mainPool
        self.finalPool = finalPool
        self.pool_ks = pool_ks
        self.mainDropout = mainDropout
        self.finalDropout = finalDropout
        self.eps = eps
        self.momentum = momentum
        self.bias = True
        self.transConv1d = transConv1d
        self.sides = [in_side] #[_ntuple(in_side)]
        self.seq = nn.Sequential()
    
    def getPadding(self, extra_pad, kernel_size, stride, dilation):
        '''
        obtain the padding or output_padding for Conv1d and ConvTranspose1d when giving kernel size, stride, and extra_pad
        '''
        # pads = [0, 1, 2, 3, 4, 5]
        pads = [i for i in range(kernel_size)]
        #note that output padding must be smaller than either stride or dilation
        out_pads = [i for i in range(min(stride, dilation)+1)]
        if self.transConv1d:
            for out_pad in out_pads:
                for pad in pads:
                    if kernel_size-stride-2*pad+out_pad == extra_pad:
                        padding = pad
                        output_padding = out_pad
                        return padding, output_padding
        else:
            for pad in pads:
                if (2*pad - dilation*(kernel_size-1) -1)//stride + 1 == extra_pad:
                    padding = pad
                    return padding
            print ('Warning: No padding matching !!!')

    def __conv1d(self, name, in_channel, out_channel, kernel_size, stride, extra_pad):
        '''
        The default settings are 'extra_pad=0' and 'dilation=1', so, the size of channel will be reduced by half when using 'Conv1d',
        and will be increase to two times of the original size when using 'ConvTranspose1d'.
        
        For Conv1d, the output size is: H_out = H_in/stride + (2*padding - dilation*(kernel_size-1) -1)/stride + 1,
        for transConv1d, the output size is: H_out = H_in*stride + kernel_size-stride-2*padding + output_padding
        
        :param extra_pad: extra_pad is defined as "(2*padding - dilation*(kernel_size-1) -1)/stride + 1" for Conv1d,
                          and "kernel_size-stride-2*padding + output_padding" for transConv1d
        '''
        if self.transConv1d:
            padding, output_panding = self.getPadding(extra_pad, kernel_size, stride, self.dilation)
            side_H = self.sides[-1]*stride + kernel_size-stride-2*padding + output_panding
            self.sides.append(side_H)
            self.seq.add_module(name, nn.ConvTranspose1d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=output_panding,bias=self.bias,dilation=self.dilation))
        else:
            padding = self.getPadding(extra_pad, kernel_size, stride, self.dilation)
            side_H = self.sides[-1]//stride + (2*padding - self.dilation*(kernel_size-1)-1)//stride + 1
            self.sides.append(side_H)
            self.seq.add_module(name, nn.Conv1d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,bias=self.bias,dilation=self.dilation))
    
    def _stride(self):
        for i in range(len(self.kernels_size)):
            if self.kernels_size[i]==1 and self.strides[i]!=1:
                print ('The stride "%s" that not match kernel size was enforced to be "1" !!!'%self.strides[i])
                self.strides[i] = 1
    
    def get_seq(self):
        self._stride()
        
        for i in range(self.layers-1):
            self.__conv1d(self.seq_name(), self.channels[i], self.channels[i+1], self.kernels_size[i], self.strides[i], self.extra_pads[i])
            if self.mainBN:
                self._batchnorm1d(self.seq_name(), self.channels[i+1])
            if self.mainActive!='None':
                self._activation(self.seq_name(), self.mainActive)
            if self.mainPool!='None':
                self._pooling(self.seq_name(), self.mainPool, kernel_size=self.pool_ks)
                self.sides[-1] = self.sides[-1]//self.pool_ks #ks
            if self.mainDropout!='None':
                self._dropout(self.seq_name(), self.mainDropout) #added later
        
        self.__conv1d(self.seq_name(), self.channels[-2], self.channels[-1], self.kernels_size[-1], self.strides[-1], self.extra_pads[-1])
        if self.finalBN:
            self._batchnorm1d(self.seq_name(), self.channels[-1])
        if self.finalActive!='None':
            self._activation(self.seq_name(), self.finalActive)
        if self.finalPool!='None':
            self._pooling(self.seq_name(), self.finalPool, kernel_size=self.pool_ks)
            self.sides[-1] = self.sides[-1]//self.pool_ks #ks
        if self.finalDropout!='None':
            self._dropout(self.seq_name(), self.finalDropout) #added later
        print ('sides: %s'%self.sides)
        return self.seq

class Conv2dSeq(SeqName,BatchNorm,Activation,Pooling,Dropout):
    def __init__(self, channels, kernels_size=None, strides=None, extra_pads=None, 
                 mainBN=True, finalBN=True, mainActive='relu', finalActive='relu', 
                 mainPool='None', finalPool='None', mainDropout='None', finalDropout='None',
                 eps=1e-05, momentum=0.1, transConv2d=False, in_side=512):
        ''' sequence of Conv2d or ConvTranspose2d '''
        
        super(Conv2dSeq, self).__init__('-1') #or SeqName.__init__(self, '-1')
        self.channels = channels
        self.layers = len(channels) - 1
        if kernels_size is None:
            self.kernels_size = [(3,3) for i in range(self.layers)]
        else:
            self.kernels_size = multi_ntuple(kernels_size)
        if strides is None:
            self.strides = [(2,2) for i in range(self.layers)]
        else:
            self.strides = multi_ntuple(strides)
        if extra_pads is None:
            self.extra_pads = [(0,0) for i in range(self.layers)]
        else:
            self.extra_pads = multi_ntuple(extra_pads)
        self.dilation = 1
        self.mainBN = mainBN
        self.finalBN = finalBN
        self.mainActive = mainActive
        self.finalActive = finalActive
        self.mainPool = mainPool
        self.finalPool = finalPool
        self.mainDropout = mainDropout
        self.finalDropout = finalDropout
        self.eps = eps
        self.momentum = momentum
        self.bias = True
        self.transConv2d = transConv2d
        self.sides = [_ntuple(in_side)]
        self.seq = nn.Sequential()
    
    def getPadding(self, extra_pad, kernel_size, stride, dilation):
        '''
        obtain the padding or output_padding for Conv2d and ConvTranspose2d when giving kernel size, stride, and extra_pad
        '''
        # pads = [0, 1, 2, 3, 4, 5]
        pads = [i for i in range(kernel_size)]
        #note that output padding must be smaller than either stride or dilation
        out_pads = [i for i in range(min(stride, dilation)+1)]
        if self.transConv2d:
            for out_pad in out_pads:
                for pad in pads:
                    if kernel_size-stride-2*pad+out_pad == extra_pad:
                        padding = pad
                        output_padding = out_pad
                        return padding, output_padding
        else:
            for pad in pads:
                if (2*pad - dilation*(kernel_size-1) -1)//stride + 1 == extra_pad:
                    padding = pad
                    return padding
            print ('Warning: No padding matching !!!')
    
    def __conv2d(self, name, in_channel, out_channel, kernel_size, stride, extra_pad):
        '''
        The default settings are 'extra_pad=0' and 'dilation=1', so, the size of channel will be reduced by half when using 'Conv2d',
        and will be increase to two times of the original size when using 'ConvTranspose2d'.
        
        For Conv2d, the output size is: H_out = H_in/stride + (2*padding - dilation*(kernel_size-1) -1)/stride + 1,
        for transConv2d, the output size is: H_out = H_in*stride + kernel_size-stride-2*padding + output_padding
        
        :param extra_pad: extra_pad is defined as "(2*padding - dilation*(kernel_size-1) -1)/stride + 1" for Conv2d,
                          and "kernel_size-stride-2*padding + output_padding" for transConv2d
        '''
        kernel_size = _ntuple(kernel_size)
        stride = _ntuple(stride)
        extra_pad = _ntuple(extra_pad)
        if self.transConv2d:
            padding_0, output_panding_0 = self.getPadding(extra_pad[0], kernel_size[0], stride[0], _ntuple(self.dilation)[0])
            padding_1, output_panding_1 = self.getPadding(extra_pad[1], kernel_size[1], stride[1], _ntuple(self.dilation)[1])
            padding = (padding_0, padding_1)
            output_panding = (output_panding_0, output_panding_1)
            
            side_H = self.sides[-1][0]*stride[0] + kernel_size[0]-stride[0]-2*padding[0] + output_panding[0]
            side_W = self.sides[-1][1]*stride[1] + kernel_size[1]-stride[1]-2*padding[1] + output_panding[1]
            self.sides.append((side_H, side_W))
            self.seq.add_module(name, nn.ConvTranspose2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=output_panding,bias=self.bias,dilation=self.dilation))
        else:
            padding_0 = self.getPadding(extra_pad[0], kernel_size[0], stride[0], _ntuple(self.dilation)[0])
            padding_1 = self.getPadding(extra_pad[1], kernel_size[1], stride[1], _ntuple(self.dilation)[1])
            padding = (padding_0, padding_1)
            
            side_H = self.sides[-1][0]//stride[0] + (2*padding[0] - _ntuple(self.dilation)[0]*(kernel_size[0]-1)-1)//stride[0] + 1
            side_W = self.sides[-1][1]//stride[1] + (2*padding[1] - _ntuple(self.dilation)[1]*(kernel_size[1]-1)-1)//stride[1] + 1
            self.sides.append((side_H, side_W))
            self.seq.add_module(name, nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,bias=self.bias,dilation=self.dilation))
    
    def _stride(self):
        self.kernels_size = multi_ntuple(self.kernels_size)
        self.strides = multi_ntuple(self.strides)
        stride_list = [list(self.strides[i]) for i in range(len(self.strides))]
        for i in range(len(self.kernels_size)):
            for j in range(2):
                if self.kernels_size[i][j]==1 and self.strides[i][j]!=1:
                    print ('The stride "%s" that not match kernel size was enforced to be "1" !!!'%self.strides[i][j])
                    stride_list[i][j] = 1
        self.strides = [tuple(stride_list[i]) for i in range(len(self.strides))]
    
    def get_seq(self):
        self._stride()
        
        for i in range(self.layers-1):
            self.__conv2d(self.seq_name(), self.channels[i], self.channels[i+1], self.kernels_size[i], self.strides[i], self.extra_pads[i])
            if self.mainBN:
                self._batchnorm2d(self.seq_name(), self.channels[i+1])
            if self.mainActive!='None':
                self._activation(self.seq_name(), self.mainActive)
            if self.mainPool!='None':
                self._pooling(self.seq_name(), self.mainPool, kernel_size=2)
                self.sides[-1] = (self.sides[-1][0]//2, self.sides[-1][1]//2)
            if self.mainDropout!='None':
                self._dropout(self.seq_name(), self.mainDropout) #added later
        
        self.__conv2d(self.seq_name(), self.channels[-2], self.channels[-1], self.kernels_size[-1], self.strides[-1], self.extra_pads[-1])
        if self.finalBN:
            self._batchnorm2d(self.seq_name(), self.channels[-1])
        if self.finalActive!='None':
            self._activation(self.seq_name(), self.finalActive)
        if self.finalPool!='None':
            self._pooling(self.seq_name(), self.finalPool, kernel_size=2)
            self.sides[-1] = (self.sides[-1][0]//2, self.sides[-1][1]//2)
        if self.finalDropout!='None':
            self._dropout(self.seq_name(), self.finalDropout) #added later
        print ('sides: %s'%self.sides)
        return self.seq

class Conv3dSeq(SeqName,BatchNorm,Activation,Pooling,Dropout):
    def __init__(self, channels, kernels_size=None, strides=None, extra_pads=None, 
                 mainBN=True, finalBN=True, mainActive='relu', finalActive='relu', 
                 mainPool='None', finalPool='None', mainDropout='None', finalDropout='None',
                 eps=1e-05, momentum=0.1, transConv3d=False, in_side=512):
        ''' sequence of Conv3d or ConvTranspose3d '''
        
        super(Conv3dSeq, self).__init__('-1') #or SeqName.__init__(self, '-1')
        self.channels = channels
        self.layers = len(channels) - 1
        if kernels_size is None:
            self.kernels_size = [(3,3,3) for i in range(self.layers)]
        else:
            self.kernels_size = multi_ntuple(kernels_size, n=3)
        if strides is None:
            self.strides = [(2,2,2) for i in range(self.layers)]
        else:
            self.strides = multi_ntuple(strides, n=3)
        if extra_pads is None:
            self.extra_pads = [(0,0,0) for i in range(self.layers)]
        else:
            self.extra_pads = multi_ntuple(extra_pads, n=3)
        self.dilation = 1
        self.mainBN = mainBN
        self.finalBN = finalBN
        self.mainActive = mainActive
        self.finalActive = finalActive
        self.mainPool = mainPool
        self.finalPool = finalPool
        self.mainDropout = mainDropout
        self.finalDropout = finalDropout
        self.eps = eps
        self.momentum = momentum
        self.bias = True
        self.transConv3d = transConv3d
        self.sides = [_ntuple(in_side, n=3)]
        self.seq = nn.Sequential()
    
    def getPadding(self, extra_pad, kernel_size, stride, dilation):
        '''
        obtain the padding or output_padding for Conv3d and ConvTranspose3d when giving kernel size, stride, and extra_pad
        '''
        # pads = [0, 1, 2, 3, 4, 5]
        pads = [i for i in range(kernel_size)]
        #note that output padding must be smaller than either stride or dilation
        out_pads = [i for i in range(min(stride, dilation)+1)]
        if self.transConv3d:
            for out_pad in out_pads:
                for pad in pads:
                    if kernel_size-stride-2*pad+out_pad == extra_pad:
                        padding = pad
                        output_padding = out_pad
                        return padding, output_padding
        else:
            for pad in pads:
                if (2*pad - dilation*(kernel_size-1) -1)//stride + 1 == extra_pad:
                    padding = pad
                    return padding
            print ('Warning: No padding matching !!!')
    
    def __conv3d(self, name, in_channel, out_channel, kernel_size, stride, extra_pad):
        '''
        The default settings are 'extra_pad=0' and 'dilation=1', so, the size of channel will be reduced by half when using 'Conv2d',
        and will be increase to two times of the original size when using 'ConvTranspose2d'.
        
        For Conv3d, the output size is: H_out = H_in/stride + (2*padding - dilation*(kernel_size-1) -1)/stride + 1,
        for transConv3d, the output size is: H_out = H_in*stride + kernel_size-stride-2*padding + output_padding
        
        :param extra_pad: extra_pad is defined as "(2*padding - dilation*(kernel_size-1) -1)/stride + 1" for Conv3d,
                          and "kernel_size-stride-2*padding + output_padding" for transConv3d
        '''
        kernel_size = _ntuple(kernel_size, n=3)
        stride = _ntuple(stride, n=3)
        extra_pad = _ntuple(extra_pad, n=3)
        if self.transConv3d:
            padding_0, output_panding_0 = self.getPadding(extra_pad[0], kernel_size[0], stride[0], _ntuple(self.dilation, n=3)[0])
            padding_1, output_panding_1 = self.getPadding(extra_pad[1], kernel_size[1], stride[1], _ntuple(self.dilation, n=3)[1])
            padding_2, output_panding_2 = self.getPadding(extra_pad[2], kernel_size[2], stride[2], _ntuple(self.dilation, n=3)[2])
            padding = (padding_0, padding_1, padding_2)
            output_panding = (output_panding_0, output_panding_1, output_panding_2)
            
            side_D = self.sides[-1][0]*stride[0] + kernel_size[0]-stride[0]-2*padding[0] + output_panding[0]
            side_H = self.sides[-1][1]*stride[1] + kernel_size[1]-stride[1]-2*padding[1] + output_panding[1]
            side_W = self.sides[-1][2]*stride[2] + kernel_size[2]-stride[2]-2*padding[2] + output_panding[2]
            self.sides.append((side_D, side_H, side_W))
            self.seq.add_module(name, nn.ConvTranspose3d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=output_panding,bias=self.bias,dilation=self.dilation))
        else:
            padding_0 = self.getPadding(extra_pad[0], kernel_size[0], stride[0], _ntuple(self.dilation, n=3)[0])
            padding_1 = self.getPadding(extra_pad[1], kernel_size[1], stride[1], _ntuple(self.dilation, n=3)[1])
            padding_2 = self.getPadding(extra_pad[2], kernel_size[2], stride[2], _ntuple(self.dilation, n=3)[2])
            padding = (padding_0, padding_1, padding_2)
            
            side_D = self.sides[-1][0]//stride[0] + (2*padding[0] - _ntuple(self.dilation, n=3)[0]*(kernel_size[0]-1)-1)//stride[0] + 1
            side_H = self.sides[-1][1]//stride[1] + (2*padding[1] - _ntuple(self.dilation, n=3)[1]*(kernel_size[1]-1)-1)//stride[1] + 1
            side_W = self.sides[-1][2]//stride[2] + (2*padding[2] - _ntuple(self.dilation, n=3)[2]*(kernel_size[2]-1)-1)//stride[2] + 1
            self.sides.append((side_D, side_H, side_W))
            self.seq.add_module(name, nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,bias=self.bias,dilation=self.dilation))
    
    def _stride(self):
        self.kernels_size = multi_ntuple(self.kernels_size, n=3)
        self.strides = multi_ntuple(self.strides, n=3)
        stride_list = [list(self.strides[i]) for i in range(len(self.strides))]
        for i in range(len(self.kernels_size)):
            for j in range(3):
                if self.kernels_size[i][j]==1 and self.strides[i][j]!=1:
                    print ('The stride "%s" that not match kernel size was enforced to be "1" !!!'%self.strides[i][j])
                    stride_list[i][j] = 1
        self.strides = [tuple(stride_list[i]) for i in range(len(self.strides))]
    
    def get_seq(self):
        self._stride()
        
        for i in range(self.layers-1):
            self.__conv3d(self.seq_name(), self.channels[i], self.channels[i+1], self.kernels_size[i], self.strides[i], self.extra_pads[i])
            if self.mainBN:
                self._batchnorm3d(self.seq_name(), self.channels[i+1])
            if self.mainActive!='None':
                self._activation(self.seq_name(), self.mainActive)
            if self.mainPool!='None':
                self._pooling(self.seq_name(), self.mainPool, kernel_size=2)
                self.sides[-1] = (self.sides[-1][0]//2, self.sides[-1][1]//2, self.sides[-1][2]//2)
            if self.mainDropout!='None':
                self._dropout(self.seq_name(), self.mainDropout) #added later
        
        self.__conv3d(self.seq_name(), self.channels[-2], self.channels[-1], self.kernels_size[-1], self.strides[-1], self.extra_pads[-1])
        if self.finalBN:
            self._batchnorm3d(self.seq_name(), self.channels[-1])
        if self.finalActive!='None':
            self._activation(self.seq_name(), self.finalActive)
        if self.finalPool!='None':
            self._pooling(self.seq_name(), self.finalPool, kernel_size=2)
            self.sides[-1] = (self.sides[-1][0]//2, self.sides[-1][1]//2, self.sides[-1][2]//2)
        if self.finalDropout!='None':
            self._dropout(self.seq_name(), self.finalDropout) #added later
        print ('sides: %s'%self.sides)
        return self.seq
