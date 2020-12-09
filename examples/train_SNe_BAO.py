# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('..')
import ecopann.coplot.plot_contours as plc
import ecopann.cosmic_params as cosmic_params
import ecopann.ann as ann
import simulator
import matplotlib.pyplot as plt
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"


#%% observational data
fid_params = [-1, 0.3]

sim_mu = simulator.sim_SNe(fid_params=fid_params)
sim_Hz, sim_DA = simulator.sim_BAO(fid_params=fid_params)
z_SNe = sim_mu[:,0]
z_BAO = sim_Hz[:,0]


#%% cosmic model & initial parameters
param_names = ['w', 'omm']
model = simulator.Simulate_SNe_BAO(z_SNe, z_BAO)

init_params = np.array([[-2, 0], [0, 0.6]])
params_dict = {'omm'     : [r'$\Omega_m$', 0.3, 0.0, 1.0],
               'w'       : [r'$w$', -1, np.nan, np.nan]}


#%% estimate parameters using ECoPANN
steps_n = 8
num_train = 1000 #3000
epoch = 1000 #2000


predictor = ann.ANN([sim_mu, sim_Hz, sim_DA], model, param_names, params_dict=params_dict,
                    cov_matrix=None, init_params=init_params, epoch=epoch,
                    num_train=num_train, local_samples=None, steps_n=steps_n)

predictor.train(path='SNe_BAO')
chain_ann = predictor.chain_ann

predictor.plot_steps()
predictor.plot_contours(fill_contours=False, show_titles=True)
predictor.save_steps()
predictor.save_contours()


#%%
labels = cosmic_params.ParamsProperty(param_names, params_dict=params_dict).labels
plc.Contours(chain_ann).plot(bins=100,smooth=5,labels=labels,fill_contours=False,show_titles=True,
                             best_values=fid_params,show_best_value_lines=True)


#%%
plt.show()

