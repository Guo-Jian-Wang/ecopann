# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate


#%% linear model
class SimLinear(object):
    def __init__(self, x):
        self.x = x
        
    def model(self, x, a, b):
        return a + b * x
    
    def sim_y(self, params):
        a, b = params
        return self.model(self.x, a, b)
    
    def simulate(self, sim_params):
        return self.x, self.sim_y(sim_params)

def get_data(x, a_fid, b_fid, random=True):
    y_th = SimLinear(x).sim_y([a_fid, b_fid])
    err_y = y_th * 0.05
    if random:
        y = y_th + np.random.randn(len(x))*err_y
    else:
        y = y_th
    sim_data = np.c_[x, y, err_y]
    return sim_data, y_th


#%%
# params_dict
"""
Information of cosmological parameters that include the labels and physical limits: [label, limit_min, limit_max]
    
The label is used to plot figures. 
The physical limits are used to ensure that the simulated parameters have physical meaning.
    
Note: If the physical limits of parameters is unknown or there is no physical limits, it should be set to np.nan.
"""
params_dict = {'H0'      : [r'$H_0$', np.nan, np.nan], #the Hubble constant
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


#%% simulate \mu of fwCDM
class Simulate_mu(object):
    '''
    parameters to be estimated: w, \Omega_m
    '''
    def __init__(self, z):
        self.z = z
        self.c = 2.99792458e5
        
    def fwCDM_dl(self, z_i, w, omm, H0=70):
        def fwCDM(x, omm):
            return 1./np.sqrt( omm*(1+x)**3 + (1-omm)*(1+x)**(3*(1+w)) )
        dll = integrate.quad(fwCDM, 0, z_i, args=(omm))[0]
        dl_i = (1+z_i)*self.c*dll/H0
        return dl_i
    
    def fwCDM_mu(self, params):
        w, omm = params
        dl = np.vectorize(self.fwCDM_dl)(self.z, w, omm)
        mu = 5*np.log10(dl) + 25
        return mu
    
    def simulate(self, sim_params):
        return self.z, self.fwCDM_mu(sim_params)

#%% simulated SNe & BAO
class Simulate_SNe_BAO(object):
    '''
    parameters to be estimated: w, \Omega_m
    '''
    def __init__(self, z_SNe, z_BAO):
        self.z_SNe = z_SNe
        self.z_BAO = z_BAO
        self.c = 2.99792458e5
    
    def fwCDM_E(self, x, w, omm):
        return 1./np.sqrt( omm*(1+x)**3 + (1-omm)*(1+x)**(3*(1+w)) )
    
    def fwCDM_dl(self, z, w, omm, H0=70):
        def dl_i(z_i, w, omm, H0):
            dll = integrate.quad(self.fwCDM_E, 0, z_i, args=(w, omm))[0]
            dl_i = (1+z_i)*self.c *dll/H0
            return dl_i
        dl = np.vectorize(dl_i)(z, w, omm, H0)
        return dl
    
    def fwCDM_mu(self, params):
        w, omm = params
        dl = self.fwCDM_dl(self.z_SNe, w, omm)
        mu = 5*np.log10(dl) + 25
        return mu
    
    def fwCDM_Hz(self, params):
        w, omm = params
        H0 = 70
        hz = H0 * np.sqrt(omm*(1+self.z_BAO)**3 + (1-omm)*(1+self.z_BAO)**(3*(1+w)) )
        return hz

    def fwCDM_DA(self, params):
        w, omm = params
        dl = self.fwCDM_dl(self.z_BAO, w, omm)
        da = dl/(1+self.z_BAO)**2
        return da

    def simulate(self, sim_params):
        zz = [self.z_SNe, self.z_BAO, self.z_BAO]
        yy = [self.fwCDM_mu(sim_params), self.fwCDM_Hz(sim_params), self.fwCDM_DA(sim_params)]
        return zz, yy

def sim_SNe(fid_params = [-1, 0.3]):
    z = np.arange(0.1+0.05, 1.7+0.05, 0.1)
    N_per_bin = np.array([69,208,402,223,327,136,136,136,136,136,136,136,136,136,136,136])
    err_stat = np.sqrt( 0.08**2+0.09**2+(0.07*z)**2 )/np.sqrt(N_per_bin)
    err_sys = 0.01*(1+z)/1.8
    err_tot = np.sqrt( err_stat**2+err_sys**2 )
    sim_mu = Simulate_SNe_BAO(z, None).fwCDM_mu(fid_params)
    sne = np.c_[z, sim_mu, err_tot]
    return sne

def sim_BAO(fid_params = [-1, 0.3]):
    z = np.array([0.2264208 , 0.32872246, 0.42808132, 0.53026194, 0.62958298,
                  0.72888132, 0.82817967, 0.93030733, 1.02958298, 1.12885863,
                  1.22811158, 1.33017872, 1.42938629, 1.53137778, 1.63045674,
                  1.72942222, 1.80803026])
    errOverHz = np.array([0.01824, 0.01216, 0.00992, 0.00816, 0.00704, 0.00656, 0.0064 ,
                          0.00624, 0.00656, 0.00704, 0.008  , 0.00944, 0.01168, 0.0152 ,
                          0.02096, 0.02992, 0.05248])
    errOverDA = np.array([0.0112 , 0.00752, 0.00608, 0.00496, 0.00432, 0.00416, 0.004  ,
                          0.004  , 0.00432, 0.00464, 0.00544, 0.00672, 0.00848, 0.01136,
                          0.01584, 0.02272, 0.04016])
    
    sim_Hz = Simulate_SNe_BAO(None, z).fwCDM_Hz(fid_params)
    sim_Hz_err = sim_Hz * errOverHz
    sim_DA = Simulate_SNe_BAO(None, z).fwCDM_DA(fid_params)
    sim_DA_err = sim_DA * errOverDA
    sim_Hz_all = np.c_[z, sim_Hz, sim_Hz_err]
    sim_DA_all = np.c_[z, sim_DA, sim_DA_err]
    return sim_Hz_all, sim_DA_all

