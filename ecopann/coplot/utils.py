# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:14:25 2020

@author: Guojian Wang
"""

import os


def mkdir(path):
    """Make a directory in a particular location if it is not exists, otherwise, do nothing.
    
    Parameters
    ----------
    path : str
        The path of a file.
    
    Examples
    --------
    >>> mkdir('/home/UserName/test')
    >>> mkdir('test/one')
    >>> mkdir('../test/one')
    """    
    #remove the blank space in the before and after strings
    #path.strip() is used to remove the characters in the beginning and the end of the character string
#    path = path.strip()
    #remove all blank space in the strings, there is no need to use path.strip() when using this command
    path = path.replace(' ', '')
    #path.rstrip() is used to remove the characters in the right of the characters strings
    
    if path=='':
        raise ValueError('The path cannot be an empty string')
    path = path.rstrip("/")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('The directory "%s" is successfully created !'%path)
        return True
    else:
#        print('The directory "%s" is already exists!'%path)
#        return False
        pass
