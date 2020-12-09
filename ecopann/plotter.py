# -*- coding: utf-8 -*-

from .coplot import plots as pl
from .coplot import plot_contours as plc
from . import space_updater as su
import numpy as np
import matplotlib.pyplot as plt


class BestParamsData(object):
    def __init__(self, chain_all, chain_ann, param_labels='', burnIn_step=None):
        self.chain_all = chain_all
        self.steps_n = len(self.chain_all)
        self.chain_ann = chain_ann
        self.param_labels = param_labels
        self.burnIn_step = burnIn_step
    
    @property
    def bestFits_all(self):
        best_ann = []
        for i in range(self.steps_n):
            best_ann.append(su.Chains.bestFit(self.chain_all[i], symmetry_error=False))
        return np.array(best_ann)
    
    @property
    def best_fit(self):
        return su.Chains.bestFit(self.chain_ann, symmetry_error=False)
    
    def panel_data(self, p_index):
        data = {'labels' : ['Steps', self.param_labels[p_index]]}
        best_mean = self.best_fit[p_index][0]#
        data['best_mean'] = best_mean
        data['err_mean'] = [self.best_fit[p_index][1], self.best_fit[p_index][2]]
        best_fits = self.bestFits_all[:, p_index, :]
        data['bestFits'] = best_fits
        y_max = max(best_fits[:,0] + best_fits[:,2])
        y_min = min(best_fits[:,0] - best_fits[:,1])
        dy = y_max - y_min
        ylim_min = y_min - dy*0.1
        if y_max-best_mean > 0.7*dy or y_max-best_mean < 0.3*dy:
            ylim_max = y_max + dy*0.1
        else:
            ylim_max = y_max + dy*0.7
        
        lims = [0, self.steps_n+1, ylim_min, ylim_max]
        data['ylim_max'], data['ylim_min'] = ylim_max, ylim_min
        data['lims'] = lims
        data['steps_n'] = self.steps_n
        return data
    
    def panels_data(self):
        datasets = []
        for index in range(len(self.param_labels)):
            data = self.panel_data(index)
            data['burnIn_step'] = self.burnIn_step
            datasets.append(data)
        return datasets
    
    def panel(self, data):
        #ann
        plt.errorbar(range(1, data['steps_n']+1), data['bestFits'][:,0], yerr=[data['bestFits'][:,1],data['bestFits'][:,2]], fmt='o', color='r', label='ANN')
        #burn-in
        if data['burnIn_step'] is not None:
            plt.plot([data['burnIn_step']+0.5, data['burnIn_step']+0.5], [data['ylim_min'], data['ylim_max']], '--', color='grey', label='Burn-in', lw=2)
        #mean values
        plt.plot([0, data['steps_n']+1], [data['best_mean'], data['best_mean']], 'k-', label='Best-fit', lw=2)
        plt.fill_between([0, data['steps_n']+1], data['best_mean']-data['err_mean'][0], data['best_mean']+data['err_mean'][1], color='grey', alpha=0.3)
        plt.legend(fontsize=12)

class Plotters(object):
    def __init__(self, chain_all, chain_ann, param_labels='', burnIn_step=None, randn_num='', path='ann'):
        self.chain_all = chain_all
        self.steps_n = len(self.chain_all)
        self.chain_ann = chain_ann
        self.param_labels = param_labels
        self.burnIn_step = burnIn_step
        self.randn_num = randn_num
        self.path = path
        
    def plot_steps(self):
        panel_model = BestParamsData(self.chain_all, self.chain_ann, param_labels=self.param_labels, burnIn_step=self.burnIn_step)
        self.fig_steps = pl.MultiplePanels(panel_model).plot(layout_adjust=[0.3, 0.25], ticks_size=10)
        return self.fig_steps
    
    def plot_contours(self, bins=100, smooth=5, show_index=None, fill_contours=True, sigma=2, show_titles=True, line_width=2, lims=None):
        if show_index is None:
            chain_show = self.chain_ann
            labels_show = self.param_labels
        else:
            index = [idx - 1 for idx in show_index]
            chain_show = self.chain_ann[:, index]
            labels_show = []
            for idx in index:
                labels_show.append(self.param_labels[idx])
            
        if len(self.chain_ann.shape)==1:
            self.fig_contours = plc.Plot_1d(chain_show).plot(bins=bins,labels=labels_show,smooth=smooth,
                                                             show_title=show_titles,line_width=line_width,
                                                             legend=False,legend_labels=['ANN'])
        else:
            self.fig_contours = plc.Contours(chain_show).plot(bins=bins,labels=labels_show,smooth=smooth,fill_contours=fill_contours,
                                                              show_titles=show_titles,line_width=line_width,layout_adjust=[0.0,0.0],
                                                              sigma=sigma,lims=lims,legend=False,legend_labels=['ANN'],best_values=None)
        return self.fig_contours
    
    @property
    def randn_suffix(self):
        return str(self.randn_num).split('.')[-1]
    
    def save_steps(self):
        pl.savefig(self.path+'/figures', 'steps_%s.pdf'%self.randn_suffix, self.fig_steps)
    
    def save_contours(self):
        pl.savefig(self.path+'/figures', 'contours_%s.pdf'%self.randn_suffix, self.fig_contours)
