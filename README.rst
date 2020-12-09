ECoPANN
=======

**ECoPANN (Estimating Cosmological Parameters with Artificial Neural Network)**

ECoPANN is a new method to estimate cosmological parameters accurately based on ANN.

It is an alternative to the traditional `Markov chain Monte Carlo (MCMC) <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ method, and can obtain almost the same results as MCMC.

ECoPANN can be applied to the research of cosmology and even other broader scientific fields.

It is proposed by `Guo-Jian Wang, Si-Yao Li, Jun-Qing Xia (2020) <https://doi.org/10.3847/1538-4365/aba190>`_.


Attribution
-----------

If you use this code in your research, please cite `Guo-Jian Wang, Si-Yao Li, Jun-Qing Xia (2020) <https://doi.org/10.3847/1538-4365/aba190>`_.


Dependencies
------------

The main dependencies of ecopann are:

* `PyTorch <https://pytorch.org/>`_
* `CUDA <https://developer.nvidia.com/cuda-downloads>`_ (optional, but suggested)

and some commonly used modules:

* os
* sys
* numpy
* scipy
* math
* decimal
* matplotlib
* logging
* warnings
* time


Package managers
----------------

You can install ecopann by using pip::

    $ sudo pip install ecopann

or from source::

    $ git clone https://github.com/Guo-Jian-Wang/ecopann.git    
    $ cd ecopann
    $ sudo python setup.py install


License
-------

Copyright 2020-2020 Guojian Wang

ecopann is free software made available under the MIT License. For details see the LICENSE file.
