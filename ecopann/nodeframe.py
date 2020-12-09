# -*- coding: utf-8 -*-

def decreasingNode(node_in=1970, node_out=5, hidden_layer=3, get_allNode=True):
    """A network structure that the number of neurons in each hidden layer is decreased proportionally.
    
    Parameters
    ----------
    node_in : int
        The number of nodes in the input layer.
    node_out : int
        The number of nodes in the output layer.
    hidden_layer : int
        The number of the hidden layers.
    get_allNode : bool
        If True, return the number of all nodes, otherwise, only return the number of nodes of hidden layers. Default: True
    
    Returns
    -------
    list
        A list that contains the number of nodes in each layer.
    """
    decreasing_factor = (node_in*1.0/node_out)**( 1.0/(hidden_layer+1) )
    nodes = []
    for i in range(hidden_layer):
        nodes.append(int(round(node_in/decreasing_factor**(i+1))))
    nodes = tuple(nodes)
    if get_allNode:
        nodes = tuple([node_in])+nodes+tuple([node_out])
    return list(nodes)
