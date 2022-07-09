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
a_fid, b_fid = 1.5, 2.5
x = np.linspace(10, 20, 501)
sim_data, y_th = simulator.get_data(x, a_fid, b_fid)

params_dict = {'a' : [r'$a$', 1.5, np.nan, np.nan],
               'b': [r'$b$', 2.5, np.nan, np.nan]}
param_names = [key for key in params_dict.keys()]

# %% estimate parameters using ECoPANN
randn_num = ; steps_n = 8

predictor = ann.RePredict(sim_data, cov_matrix=None, path='linear',
                          randn_num=randn_num, steps_n=steps_n,
                          params_dict=params_dict)
predictor.from_chain()
# predictor.from_net()
chain_ann = predictor.chain_ann
predictor.plot_steps()
predictor.plot_contours(bins=50, fill_contours=True, show_titles=True)
predictor.save_steps()
predictor.save_contours()

#%%
labels = cosmic_params.ParamsProperty(param_names, params_dict=params_dict).labels
plc.Contours(chain_ann).plot(bins=50,labels=labels,fill_contours=False,show_titles=True,
                             best_values=[a_fid, b_fid],show_best_value_lines=True)


#%%
plt.show()

