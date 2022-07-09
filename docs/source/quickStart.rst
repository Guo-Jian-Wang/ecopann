.. _quickStart:

Quick Start
***********

Here we show two examples to illustrate how ECoPANN works, and the readers can archieve their project quickly by modifing the examples. The code used to generate these examples can be downloaded `here <https://github.com/Guo-Jian-Wang/ecopann/tree/master/examples>`_.


Using one dataset
=================

The main process of using ECoPANN includes preparing observational data and theoretical model, training the network, estimating parameters using the ANN chains or the well-trained network.

Let's consider a general case, the simple linear model:

.. math::
    y = a + b\times x

where `a` and `b` are two free parameters to be estimated, and `y` is the measurement. We first build a class object for this model::
    
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

.. Note::
    The class object must contain a ``simulate`` method, which is used to simulate samples in the training process.

Then a data sample can be simulated as observational data, by using the function below::
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def get_data(x, a_fid, b_fid):
        y_th = SimLinear(x).sim_y([a_fid, b_fid])
        err_y = y_th * 0.05
        y = y_th + np.random.randn(len(x))*err_y
        sim_data = np.c_[x, y, err_y]
        return sim_data, y_th
    
    a_fid, b_fid = 1.5, 2.5
    x = np.linspace(10, 20, 501)
    sim_data, y_th = get_data(x, a_fid, b_fid)
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(x, sim_data[:,1], yerr=sim_data[:,2], fmt='.', color='gray', alpha=0.5, label='Simulated data')
    plt.plot(x, y_th, 'r-', label='Fiducial', lw=3)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.legend(fontsize=16)

.. figure:: figures/linear_samples.png
    :align: center
    :scale: 30 %

After that, we can build a model instance and make some settings for parameter initialization::

    model = SimLinear(x)
    params_dict = {'a' : [r'$a$', np.nan, np.nan],
                   'b' : [r'$b$', 0, 10]}
    param_names = [key for key in params_dict.keys()]
    init_params = np.array([[0, 5], [1, 3]])

where ``params_dict`` is a dictionary that contains information of the parameters, which include the labels and physical limits, and ``init_params`` is the initial settings of the parameter space.

.. Note::
    If the physical limits of parameters (the minimum and maximum values) is unknown or there is no physical limits, it should be set to
    ``np.nan``.

Finally, we can build a predictor and pass the data and model instance to it to train the network::

    import ecopann.ann as ann
    
    stepStop_n = 3
    num_train = 1000
    epoch = 1000

    predictor = ann.ANN(sim_data, model, param_names, params_dict=params_dict,
                        cov_matrix=None, init_params=init_params, epoch=epoch,
                        num_train=num_train, local_samples=None, stepStop_n=stepStop_n)

    predictor.train(path='linear')

In the training process, the results which include the network model, the predicted ANN chain, and some hyperparameters will be saved to the indicated folder. After the training process, we can plot and save the predicted parameters in each step by using the following commands::

    predictor.plot_steps()
    predictor.save_steps()

.. figure:: figures/linear_steps.png
    :align: center
    :scale: 35 %

and can also plot the contours of the estimated parameters::
    
    predictor.plot_contours(fill_contours=False, show_titles=True)
    predictor.save_contours()

.. figure:: figures/linear_contour.png
    :align: center
    :scale: 35 %

.. Note::
    The parameters are estimated using the chains after the burn-in phase, and the ``stepStop_n`` is the number of chains to be obtained.
    
    Also, the number of the training set (``num_train``) and the ``epoch`` should be set large enough to ensure the network
    learns a reliable mapping. See the ``ecopann.ann.ANN`` module in :ref:`parameter_estimation` for details.

In the training process, the results of each step will be saved, so it is possible to estimate parameters before the ending of the training process. To do this, one needs to build another predictor and pass the random number of the first step (``randn_num``) that identifies the saved results to the predictor::

    import ecopann.cosmic_params as cosmic_params
    import coplot.plot_contours as plc
    
    # randn_num and steps_n should be replaced by your values 
    randn_num = 1.06304; steps_n = 4
    
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
    
    labels = cosmic_params.ParamsProperty(param_names, params_dict=params_dict).labels
    plc.Contours(chain_ann).plot(bins=50,labels=labels,fill_contours=False,show_titles=True,
                                 best_values=[a_fid, b_fid],show_best_value_lines=True)

Note that ``steps_n`` should be set according to the steps completed.



Using multiple datasets
=======================

In practical scientific research, we may need to use multiple data sets to constrain the parameters, which is also possible for ECoPANN. To illustrate this, we constrain parameters of :math:`w`\CDM cosmological model using the observations of Type Ia supernovae (SNe Ia) and  baryon acoustic oscillations (BAO). We first build a class object for this model::

    import numpy as np
    from scipy import integrate
    
    class Simulate_SNe_BAO(object):
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

Note that the measurement of SNe Ia is the distance modulus :math:`\mu(z)` (``fwCDM_mu``), and the measurements of BAO are the Hubble parameter :math:`H(z)` (``fwCDM_Hz``) and the angular diameter distance :math:`D_A(z)` (``fwCDM_DA``). So, the outputs of the ``simulate`` method are :math:`\mu(z)`, :math:`H(z)`, and :math:`D_A(z)`. The parameters to be constrained are :math:`w` (``w``) and :math:`\Omega_m` (``omm``). Then we generate mock observational using the method below::
    
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

    fid_params = [-1, 0.3]
    sim_mu = sim_SNe(fid_params=fid_params)
    sim_Hz, sim_DA = sim_BAO(fid_params=fid_params)
    z_SNe = sim_mu[:,0]
    z_BAO = sim_Hz[:,0]

After that, we can build a model instance and make some settings for parameter initialization::

    model = Simulate_SNe_BAO(z_SNe, z_BAO)
    params_dict = {'w'      : [r'$w$', np.nan, np.nan],
                  'omm'     : [r'$\Omega_m$', 0.0, 1.0]}
    param_names = [key for key in params_dict.keys()]
    init_params = np.array([[-2, 0], [0, 0.6]])

Finally, we can build a predictor and pass the data and model instance to it to train the network::

    stepStop_n = 3
    num_train = 1000
    epoch = 1000
    
    predictor = ann.ANN([sim_mu, sim_Hz, sim_DA], model, param_names, params_dict=params_dict,
                        cov_matrix=None, init_params=init_params, epoch=epoch,
                        num_train=num_train, local_samples=None, stepStop_n=stepStop_n)
    
    predictor.train(path='SNe_BAO')
    chain_ann = predictor.chain_ann
    predictor.plot_steps()
    predictor.plot_contours(fill_contours=False, show_titles=True)
    predictor.save_steps()
    predictor.save_contours()

.. figure:: figures/SNe_BAO_steps.png
    :align: center
    :scale: 35 %

.. figure:: figures/SNe_BAO_contour.png
    :align: center
    :scale: 35 %

.. Note::
    The data used here have no covariance, so the covariance matrix (``cov_matrix``) is set to ``None``. If the data have
    covariance matrices, such as ``cov1``, ``cov2``, and ``cov3``, they should be passed to the predictor by setting
    ``cov_matrix=[cov1, cov2, cov3]``. Furthermore, if some data sets have no covariance, such as the first data set, the
    setting of the covariance matrix should be ``cov_matrix=[None, cov2, cov3]``.


.. predict future data =================== predict future data using the well-trained networks


