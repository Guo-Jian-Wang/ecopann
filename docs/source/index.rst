.. ECoPANN documentation master file, created by
   sphinx-quickstart on Sat Dec  5 04:13:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. Welcome to ECoPANN's documentation! **********************************


ECoPANN
=======

**ECoPANN** (``E``\stimating ``Co``\smological ``P``\arameters with ``A``\rtificial ``N``\eural ``N``\etwork) is a new method to estimate cosmological parameters accurately based on Artificial Neural Network (ANN) proposed by `Guo-Jian Wang, Si-Yao Li, Jun-Qing Xia (2020) <https://doi.org/10.3847/1538-4365/aba190>`_. 

It is an alternative to the traditional `Markov chain Monte Carlo (MCMC) <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ method, and can obtain almost the same results as MCMC.

As a general method of parameter estimation, ECoPANN can be used for research in many scientific fields. The code ecopann is available for free from `GitHub <https://github.com/Guo-Jian-Wang/ecopann>`_. It can be executed on GPUs or CPUs.



.. image:: https://img.shields.io/badge/GitHub-ecopann-blue.svg?style=flat
    :target: https://github.com/Guo-Jian-Wang/ecopann
.. image:: https://img.shields.io/badge/License-MIT-green.svg?style=flat
    :target: https://github.com/Guo-Jian-Wang/ecopann/blob/master/LICENSE
.. image:: https://img.shields.io/badge/ApJS-ECoPANN-blue.svg?style=flat
    :target: https://iopscience.iop.org/article/10.3847/1538-4365/aba190
.. image:: https://img.shields.io/badge/arXiv-2005.07089-gold.svg?style=flat
    :target: https://arxiv.org/abs/2005.07089


How to use ECoPANN
==================

First, you are probably going to needs to see the :ref:`introduction` guide to learn the basic principles of ECoPANN. After that, you may need to install ecopann on your computer according to the :ref:`installation` guide, and then following the :ref:`quickStart` guide to learn how to use it. If you need more detailed information about a specific function, the :ref:`package_reference` below should have what you need.


Contents:
=========

.. toctree::
   :maxdepth: 2
   
   introduction
   installation
   quickStart
   package_reference
   release_history

.. potential_advantages


Attribution
===========

If you use this code in your research, please cite our paper
(`ApJS <https://doi.org/10.3847/1538-4365/aba190>`_,
`arXiv <https://arxiv.org/abs/2005.07089>`_,
`ADS <https://ui.adsabs.harvard.edu/abs/2020ApJS..249...25W/abstract>`_,
`BibTex <https://ui.adsabs.harvard.edu/abs/2020ApJS..249...25W/exportcitation>`_).


License
=======

Copyright 2020-2020 Guojian Wang

ecopann is free software made available under the MIT License. For details see the `LICENSE <https://github.com/Guo-Jian-Wang/ecopann/blob/master/LICENSE>`_.

