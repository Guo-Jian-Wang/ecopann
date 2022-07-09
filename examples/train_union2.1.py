# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('..')
import coplot.plot_contours as plc
import ecopann.cosmic_params as cosmic_params
import ecopann.ann as ann
import simulator
import matplotlib.pyplot as plt
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"


#%% obs data
union = np.loadtxt('data/Union2.1.txt')[:,:3]
obs_z = union[:,0]

#%% cosmic model & initial parameters
param_names = ['w', 'omm']
init_params = np.array([[-1.665, -0.35], [0, 0.5959]])
model = simulator.Simulate_mu(obs_z)


#%% estimate parameters using ECoPANN
stepStop_n = 3 #3
num_train = 1000 #3000
epoch = 1000 #2000


predictor = ann.ANN(union, model, param_names, params_dict=simulator.params_dict,
                    cov_matrix=None, init_params=init_params, epoch=epoch,
                    num_train=num_train, local_samples=None, stepStop_n=stepStop_n)

predictor.train(path='union2.1_fwCDM')
chain_ann = predictor.chain_ann

predictor.plot_steps()
predictor.plot_contours(fill_contours=False, show_titles=True)
predictor.save_steps()
predictor.save_contours()

predictor.eco.plot_loss()

# %% MCMC chain
chain_mcmc = np.load('data/MCMC_chains/chain_fwCDM_2params.npy')
chain_all = [chain_ann, chain_mcmc]

plc.Contours(chain_all).plot(labels=cosmic_params.ParamsProperty(param_names).labels,smooth=5,
                              fill_contours=False,show_titles=True,line_width=2,layout_adjust=[0.0,0.0],
                              lims=None,legend=True,legend_labels=['ANN', 'MCMC'])


plt.show()

