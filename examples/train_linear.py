# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
import simulator
import coplot.plot_contours as plc
import ecopann.cosmic_params as cosmic_params
import ecopann.ann as ann
import numpy as np
import matplotlib.pyplot as plt


#%% simulate data
a_fid, b_fid = 1.5, 2.5
x = np.linspace(10, 20, 501)
sim_data, y_th = simulator.get_data(x, a_fid, b_fid, random=False)

#%% plot
plt.figure(figsize=(8, 6))
plt.errorbar(x, sim_data[:,1], yerr=sim_data[:,2], fmt='.', color='gray', alpha=0.5, label='Simulated data')
plt.plot(x, y_th, 'r-', label='Fiducial', lw=3)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.legend(fontsize=16)


#%%
model = simulator.SimLinear(x)
params_dict = {'a' : [r'$a$', np.nan, np.nan],
               'b' : [r'$b$', 0, 10]}
param_names = [key for key in params_dict.keys()]
init_params = np.array([[0, 5], [1, 3]])


#%%
stepStop_n = 3 #3
num_train = 1000 #3000
epoch = 1000 #2000

predictor = ann.ANN(sim_data, model, param_names, params_dict=params_dict,
                    cov_matrix=None, init_params=init_params, epoch=epoch,
                    num_train=num_train, local_samples=None, stepStop_n=stepStop_n)

predictor.train(path='linear')
chain_ann = predictor.chain_ann

predictor.plot_steps()
predictor.plot_contours(bins=50, fill_contours=False, show_titles=True)
predictor.save_steps()
predictor.save_contours()

predictor.eco.plot_loss()

#%%
labels=cosmic_params.ParamsProperty(param_names, params_dict=params_dict).labels
plc.Contours(chain_ann).plot(bins=50, labels=labels,fill_contours=False,show_titles=True,
                              best_values=[a_fid, b_fid],show_best_value_lines=True)


plt.show()

