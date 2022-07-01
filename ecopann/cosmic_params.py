# -*- coding: utf-8 -*-

import numpy as np


#%%parameters zoo
def params_dict_zoo():
    """Information of cosmological parameters that include the labels and physical limits: [label, limit_min, limit_max]
    
    The label is used to plot figures. 
    The physical limits are used to ensure that the simulated parameters have physical meaning.
    
    Note
    ----
    If the physical limits of parameters is unknown or there is no physical limits, it should be set to np.nan.
    """
    return {'H0'      : [r'$H_0$', np.nan, np.nan], #the Hubble constant
            'omm'     : [r'$\Omega_{\rm m}$', 0.0, 1.0], #the matter density parameter
            'ombh2'   : [r'$\Omega_{\rm b} h^2$', np.nan, np.nan], #baryon density
            'omch2'   : [r'$\Omega_{\rm c} h^2$', np.nan, np.nan], #cold dark matter density
            'tau'     : [r'$\tau$', 0.003, np.nan], #the optical depth
            'As'      : [r'$A_{\rm s}$', np.nan, np.nan], #the amplitude of primordial inflationary perturbations
            'A'       : [r'$10^9A_{\rm s}$', np.nan, np.nan], #As/10^-9
            'ns'      : [r'$n_{\rm s}$', np.nan, np.nan], #the spectral index of primordial inflationary perturbations
            'mnu'     : [r'$\sum m_\nu$', 0.0, np.nan], #the sum of neutrino masses, eV, #Note: here we set it to 1 to avoid be scaled when training the network
            'w'       : [r'$w$', np.nan, np.nan], #parameter of wCDM model
            'oml'     : [r'$\Omega_\Lambda$', 0.0, 1.0], #\Omega_\Lambda, 1-Omega_m-Omega_k
            
            'MB'      : [r'$M_B$', np.nan, np.nan], #the absolute magnitude of SNe Ia (M_B)
            'muc'     : [r'$\mu_c$', np.nan, np.nan], #5*log10(c/H0/Mpc) + MB + 25
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
    def params_limit(self):
        p_limit = []
        for p in self.param_names:
            p_min = self.params_dict[p][1]
            p_max = self.params_dict[p][2]
            p_limit.append([p_min, p_max])
        return np.array(p_limit)
