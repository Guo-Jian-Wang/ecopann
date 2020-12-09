# -*- coding: utf-8 -*-

import numpy as np


#%%parameters zoo
def params_dict_zoo():
    """Information of cosmological parameters that include the labels, the base values, the minimum values, and the maximum values: [label, base value, minimum, maximum]
    
    The label is used to plot figures, the general value is used for data normalization when training the network,
    the extremum of parameters is used to update parameter space in the training process.
    
    Note
    ----
    If the extremum of parameters is unknown or there is no extremum, it should be set to np.nan.
    """
    return {'H0'      : [r'$H_0$', 70.0, np.nan, np.nan], #the Hubble constant
            'omm'     : [r'$\Omega_m$', 0.3, 0.0, 1.0], #the matter density parameter
            'ombh2'   : [r'$\Omega_b h^2$', 0.02222, np.nan, np.nan], #baryon density
            'omch2'   : [r'$\Omega_c h^2$', 0.1197, np.nan, np.nan], #cold dark matter density
            'tau'     : [r'$\tau$', 0.078, 0.003, 0.45], #the optical depth
            'As'      : [r'$A_s$', 2.195511e-9, np.nan, np.nan], #the amplitude of primordial inflationary perturbations
            'A'       : [r'$10^9A_s$', 2.195511, np.nan, np.nan], #As/10^-9
            'ns'      : [r'$n_s$', 0.9655, np.nan, np.nan], #the spectral index of primordial inflationary perturbations
            'mnu'     : [r'$\sum m_\nu$', 0.715, 0.0, 3.0], #the sum of neutrino masses, eV, #Note: here we set it to 1 to avoid be scaled when training the network
            'w'       : [r'$w$', -1, np.nan, np.nan], #parameter of wCDM model
            'oml'     : [r'$\Omega_\Lambda$', 0.7, 0.0, 1.0], #\Omega_\Lambda, 1-Omega_m-Omega_k
            
            'MB'      : [r'$M_B$', -19.3, np.nan, np.nan], #the absolute magnitude of SNe Ia (M_B)
            'mu0'     : [r'$\mu_0$', 23.8, np.nan, np.nan], #MB + 5*log10(c/H0) + 25
            }

#%%
class ParamsProperty:
    def __init__(self, param_names, params_dict=None):
        self.param_names = param_names
        if params_dict is None:
            self.params_dict = params_dict_zoo()
        else:
            self.params_dict = params_dict
    
    @property
    def labels(self):
        return [self.params_dict[p][0] for p in self.param_names]
    
    @property
    def param_fullNames(self):
        name = []
        for i in range(len(self.param_names)):
            latex_name = self.labels[i].split('$')[1]
            name.append([self.param_names[i], latex_name])
        name = np.array(name)
        return name
    
    @property
    def params_base(self):
        return np.array([self.params_dict[p][1] for p in self.param_names])
    
    @property
    def params_limit(self):
        p_limit = []
        for p in self.param_names:
            p_min = self.params_dict[p][2]
            p_max = self.params_dict[p][3]
            p_limit.append([p_min, p_max])
        return np.array(p_limit)
