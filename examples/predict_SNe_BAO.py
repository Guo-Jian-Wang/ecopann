# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
import ecopann.ann as ann
import coplot.plot_contours as plc
import ecopann.cosmic_params as cosmic_params
import simulator
import matplotlib.pyplot as plt
import numpy as np


#%% observational data
fid_params = [-1, 0.3]

sim_mu = simulator.sim_SNe(fid_params=fid_params)
sim_Hz, sim_DA = simulator.sim_BAO(fid_params=fid_params)
z_SNe = sim_mu[:,0]
z_BAO = sim_Hz[:,0]

param_names = ['w', 'omm']
params_dict = {'omm'     : [r'$\Omega_m$', 0.3, 0.0, 1.0],
               'w'       : [r'$w$', -1, np.nan, np.nan]}


# %% estimate parameters using ECoPANN
randn_num = ; steps_n = 8


predictor = ann.RePredict([sim_mu, sim_Hz, sim_DA], cov_matrix=None, path='SNe_BAO',
                         randn_num=randn_num, steps_n=steps_n,
                         params_dict=simulator.params_dict)
predictor.from_chain()
# predictor.from_net()
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

