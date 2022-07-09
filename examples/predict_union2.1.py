# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
import ecopann.ann as ann
import coplot.plot_contours as plc
import ecopann.cosmic_params as cosmic_params
import simulator
import matplotlib.pyplot as plt
import numpy as np


#%% obs data
union = np.loadtxt('data/Union2.1.txt')[:,:3]


# %% estimate parameters using ECoPANN
randn_num = ; steps_n = 4


predictor = ann.RePredict(union, cov_matrix=None, path='union2.1_fwCDM',
                         randn_num=randn_num, steps_n=steps_n,
                         params_dict=simulator.params_dict)
predictor.from_chain()
# predictor.from_net()
chain_ann = predictor.chain_ann

predictor.plot_steps()
predictor.plot_contours(fill_contours=False, show_titles=True)
predictor.save_steps()
predictor.save_contours()

predictor.eco.plot_loss()

#%%
chain_mcmc = np.load('data/MCMC_chains/chain_fwCDM_2params.npy')
chain_all = [chain_ann, chain_mcmc]

param_names = ['w', 'omm']
plc.Contours(chain_all).plot(labels=cosmic_params.ParamsProperty(param_names).labels,smooth=5,
                              fill_contours=False,show_titles=True,line_width=2,layout_adjust=[0.0,0.0],
                              lims=None,legend=True,legend_labels=['ANN', 'MCMC'])

#%%
plt.show()

