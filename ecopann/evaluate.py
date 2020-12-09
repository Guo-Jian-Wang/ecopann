# -*- coding: utf-8 -*-

from . import data_processor as dp
import os
from torch.autograd import Variable
import numpy  as np
import matplotlib.pyplot as plt


#%% get file path
class FilePath:
    def __init__(self, filedir='cnn', randn_num='', suffix='.pt', separator='_'):
        """Obtain the path of a specific file.
        
        Parameters
        ----------
        filedir : str
            The relative path of a file.
        randn_num : str
            A random number that owned by a file name.
        suffix : str
            The suffix of the file, e.g. '.npy', '.pt'
        separator : str
            Symbol for splitting the random number in the file name.
        """
        self.filedir = filedir
        self.randn_num = randn_num
        self.separator = separator
        self.file_suffix = suffix
    
    def filePath(self):
        listdir = os.listdir(self.filedir)
        for File in listdir:
            if File.endswith(self.file_suffix):
                fileName = os.path.splitext(File)[0]
                randn = fileName.split(self.separator)[-1]
                if randn == self.randn_num:
                    target_file = self.filedir + '/' + File
        if 'target_file' not in dir():
            raise IOError('No eligible files!')
        return target_file

#%% predict
def predict(net, inputs, use_GPU=False, in_type='numpy'):
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
        net = net.cuda()
        if in_type=='numpy':
            inputs = dp.numpy2cuda(inputs)
        elif in_type=='torch':
            inputs = dp.torch2cuda(inputs)
    else:
        if in_type=='numpy':
            inputs = dp.numpy2torch(inputs)
    net = net.eval() #this works for the batch normalization layers
    pred = net(Variable(inputs))
    if use_GPU:
        pred = dp.cuda2numpy(pred.data)
    else:
        pred = dp.torch2numpy(pred.data)
    return pred

#%% plot loss
def plot_loss(loss):
    # print ('The last 5 losses: ', np.array(loss[-5:]))
    print ('The average of last 100 losses: %.8f\n'%(np.mean(loss[-100:])))
    plt.figure(figsize=(6*2., 4.5*1.))
    plt.subplot(1,2,1)
    plt.semilogx(loss)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    
    plt.subplot(1,2,2)
    plt.loglog(loss)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    # plt.show()
    return
