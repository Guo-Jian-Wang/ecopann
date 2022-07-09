# -*- coding: utf-8 -*-
import coplot.plots as pl
import coplot.plot_contours as plc

from . import space_updater as su
from . import cosmic_params
import numpy as np
import matplotlib.pyplot as plt


class BestFitsData(object):
    #best fit values of each steps, used to plot steps in PlotPosterior
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

#to be updated based on ann.ANN???
class PlotPosterior(object):
    def __init__(self, chain_all, chain_ann, param_names, params_dict=None, 
                 burnIn_step=None, randn_num='', path='ann'):
        self.chain_all = chain_all
        self.steps_n = len(self.chain_all)
        self.chain_ann = chain_ann
        self.param_labels = cosmic_params.ParamsProperty(param_names,params_dict=params_dict).labels
        self.burnIn_step = burnIn_step
        self.randn_num = randn_num
        self.path = path
        
    def plot_steps(self, layout_adjust=[0.3, 0.25], suptitle=''):
        panel_model = BestFitsData(self.chain_all, self.chain_ann, param_labels=self.param_labels, burnIn_step=self.burnIn_step)
        self.fig_steps = pl.MultiplePanels(panel_model).plot(layout_adjust=layout_adjust, ticks_size=10)
        plt.suptitle(suptitle, fontsize=16)
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


#%%
def pcc(x, y):
    '''Pearson correlation coefficient
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    '''
    data = np.c_[x, y].T
    cov = np.cov(data)
    rho = cov[0,1] / np.sqrt(cov[0,0]) / np.sqrt(cov[1,1])
    return rho

class BestPredictedData(object):
    def __init__(self, params_testSet, predParams_testSet, params_trainingSet=None, 
                 predParams_trainingSet=None, param_labels='', show_reErr=True):
        self.sim_params = params_testSet
        self.pred_params = predParams_testSet
        self.params_trainingSet = params_trainingSet
        self.predParams_trainingSet = predParams_trainingSet
        self.param_labels = param_labels
        self.show_reErr = show_reErr
        
    def panel_data(self, p_index):
        data = {'labels' : [self.param_labels[p_index]+' (True)', self.param_labels[p_index]+' (Predicted)']}
        data['sim_param'] = self.sim_params[:, p_index]
        data['pred_param'] = self.pred_params[:, p_index]
        data['pcc'] = pcc(self.sim_params[:, p_index], self.pred_params[:, p_index])
        
        param_min_1, param_max_1 = self.sim_params[:, p_index].min(), self.sim_params[:, p_index].max()
        param_min_2, param_max_2 = self.pred_params[:, p_index].min(), self.pred_params[:, p_index].max()
        param_min, param_max = min([param_min_1, param_min_2]), max([param_max_1, param_max_2])
        lims = [param_min, param_max, param_min, param_max]
        data['lims'] = lims
        pp = np.linspace(param_min, param_max, 100)
        data['pp'] = pp
        
        data['xx'] = (param_max-param_min)*0.06 + param_min #0.1, 0.06
        data['yy'] = (param_max-param_min)*0.85 + param_min #0.8, 0.85
        data['yy_pcc'] = (param_max-param_min)*0.72 + param_min #0.68, 0.72
        return data
    
    def panels_data(self):
        datasets = []
        re_err = (self.pred_params - self.sim_params) / self.sim_params
        reErr_bestfit = su.Chains.bestFit(re_err, symmetry_error=False)#relative error of predicted parameters
        for index in range(len(self.param_labels)):
            data = self.panel_data(index)
            data['reErr_bestfit'] = reErr_bestfit[index]
            datasets.append(data)
        return datasets

    def panel(self, data):
        plt.plot(data['sim_param'], data['pred_param'], '.')
        plt.plot(data['pp'], data['pp'], 'r', lw=1.618)
        if self.show_reErr:
            plt.text(data['xx'], data['yy'], '$\delta: %.3f_{-%.3f}^{+%.3f}$'%(data['reErr_bestfit'][0],data['reErr_bestfit'][1],data['reErr_bestfit'][2]), fontsize=16)
            plt.text(data['xx'], data['yy_pcc'], '$r: %.3f$'%(data['pcc']), fontsize=16)

        # plt.legend(fontsize=12)

class PlotPrediction(object):
    def __init__(self, params_testSet, predParams_testSet, param_names, params_trainingSet=None, 
                 predParams_trainingSet=None, params_dict=None, show_reErr=True, randn_num='', path='ann'):
        self.params_testSet = params_testSet
        self.predParams_testSet = predParams_testSet
        self.param_labels = cosmic_params.ParamsProperty(param_names,params_dict=params_dict).labels
        self.params_trainingSet = params_trainingSet
        self.predParams_trainingSet = predParams_trainingSet
        self.show_reErr = show_reErr
        self.randn_num = randn_num
        self.path = path

    def plot(self, layout_adjust=[0.3, 0.25], suptitle=''):
        panel_model = BestPredictedData(self.params_testSet, self.predParams_testSet, param_labels=self.param_labels, show_reErr=self.show_reErr)
        self.fig_pred = pl.MultiplePanels(panel_model).plot(layout_adjust=layout_adjust, ticks_size=10)
        plt.suptitle(suptitle, fontsize=16)
        return self.fig_pred

    @property
    def randn_suffix(self):
        return str(self.randn_num).split('.')[-1]
    
    def save_fig(self):
        pl.savefig(self.path+'/figures', 'prediction_%s.pdf'%self.randn_suffix, self.fig_pred)

