# -*- coding: utf-8 -*-

def models(key):
    """Hyperparameter models.
    
    Parameters
    ----------
    key : str
        Hyperparameter model that contains hyperparameters (such as activation function, batch normalization, dropout, etc.) used in the network.
        It can be 'eco_1', 'eco_2', 'eco_3', 'eco_4', 'eco_5', 'eco_6', or 'eco_7'.

    Returns
    -------
    object
        Hyperparameter model.
    
    Note
    ----
    The hyperparameters model 'eco_3' that contains the 'rrelu' activation function is recommended.
    """
    return eval(key)()

#%% this is used for estimating cosmological parameters
def eco_1():
    return {'active' : 'relu',
            'BN' : True,
            'dropout' : 'None'
            }

def eco_2():
    return {'active' : 'elu',
            'BN' : True,
            'dropout' : 'None'
            }

def eco_3():
    return {'active' : 'rrelu',
            'BN' : True,
            'dropout' : 'None'
            }

def eco_4():
    return {'active' : 'prelu',
            'BN' : True,
            'dropout' : 'None'
            }

def eco_5():
    return {'active' : 'leakyrelu',
            'BN' : True,
            'dropout' : 'None'
            }

def eco_6():
    return {'active' : 'sigmoid',
            'BN' : True,
            'dropout' : 'None'
            }

def eco_7():
    return {'active' : 'tanh',
            'BN' : True,
            'dropout' : 'None'
            }

#%% default
def default():
    return {'active' : 'rrelu',
            'BN' : True,
            'dropout' : 'None'
            }

def nuisance_hp(hparams):
    """ The hyperparameters that could be set to deterministic values. """
    nhp = {'finalActive' : 'None',
           'finalBN' : False,
           'finalDropout' : 'None'}
    for key in nhp.keys():
        if key not in hparams.keys():
            hparams[key] = nhp[key]
    return hparams
