# -*- coding: utf-8 -*-

import coplot.plot_settings as pls

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
        randn_num : str or float
            A random number that owned by a file name.
        suffix : str
            The suffix of the file, e.g. '.npy', '.pt'
        separator : str
            Symbol for splitting the random number in the file name.
        """
        self.filedir = filedir
        self.randn_num = str(randn_num)
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
    #reshape chain
    if len(pred.shape)==1:
        pred = pred.reshape(-1, 1)
    return pred

#%% plot loss
# def plot_loss(train_loss, vali_loss=[]):
#     vali_loss_size = len(vali_loss)
#     if vali_loss_size==0:
#         print ('The average of last 100 training set losses: %.5f\n'%(np.mean(train_loss[-100:])))
#     else:
#         print ('The aveage of last 100 training/validation set losses: %.5f/%.5f\n'%(np.mean(train_loss[-100:]), np.mean(vali_loss[-100:])))
#     x = np.linspace(1, len(train_loss), len(train_loss))
#     plt.figure(figsize=(6*2., 4.5*1.))
#     pls.PlotSettings().setting(location=[1,2,1])
#     plt.plot(x, train_loss, label='Training set')
#     if vali_loss_size!=0:
#         plt.plot(x, vali_loss, label='Validation set', alpha=0.6)
#         ylim_min = min(train_loss.min(), vali_loss.min())
#     else:
#         ylim_min = train_loss.min()
#     plt.xlabel('Epochs', fontsize=16)
#     plt.ylabel('Loss', fontsize=16)
#     plt.legend(fontsize=16)
#     plt.xlim(1, len(train_loss))
#     plt.ylim(ylim_min, train_loss.max()*1.0618)
    
#     pls.PlotSettings().setting(location=[1,2,2])
#     plt.loglog(x, train_loss, label='Training set')
#     if vali_loss_size!=0:
#         plt.loglog(x, vali_loss, label='Validation set', alpha=0.6)
#     plt.xlabel('Epochs', fontsize=16)
#     plt.ylabel('Loss', fontsize=16)
#     plt.legend(fontsize=16)
#     plt.xlim(1, len(train_loss))

