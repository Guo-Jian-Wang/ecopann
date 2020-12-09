# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:17:03 2019

@author: Guojian Wang
"""

import os
import re
from setuptools import setup, find_packages


def read(filename):
    f = open(filename)
    r = f.read()
    f.close()
    return r

ver = re.compile("__version__ = \"(.*?)\"")
#m = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "refann", "__init__.py"))
#m = read(os.path.join(os.path.dirname(__file__), "refann", "__init__.py"))
m = read(os.path.join(os.getcwd(), "ecopann", "__init__.py"))
version = ver.findall(m)[0]



setup(
    name = "ecopann",
    version = version,
    keywords = ("pip", "ANN"),
    description = "Estimating Cosmological Parameters with Artificial Neural Network",
    long_description = "",
    license = "MIT",

    url = "",
    author = "Guojian Wang",
    author_email = "gjwang@mail.bnu.edu.cn",

#    packages = find_packages(),
    packages = ["ecopann", "ecopann/coplot", "examples"],
    include_package_data = True,
    data_files = ["examples/data/Union2.1.txt",
                  # "examples/data/Union21_covariance_systematics.txt",
                  "examples/data/MCMC_chains/chain_fwCDM_2params.npy",
                  # "examples/data/MCMC_chains/chain_fwCDM_2params_cov.npy",
                  ],
    platforms = "any",
    install_requires = []
)

