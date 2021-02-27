"""
Implements an offline tree-weight estimation or a constrained search tree
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper import to_df, make_dfs_ordering, get_domain_size, EPSILON

from typing import List

###########################
#### WEIGHTING SCHEMES ####
###########################


def uniform_scheme(node_id: int, nodes_df: pd.DataFrame) -> List[float]:
    """ Assign uniform weights for a node's children """

    if nodes_df['Status'].isin([3]).sum() > 0:
        raise ValueError('Nodes_df must consist only of valid nodes')
    num_kids = nodes_df[nodes_df['ParentID'].isin([node_id])].shape[0]
    weights = [1 / num_kids] * num_kids
    return weights

def domain_weight_scheme(node_id: int, nodes_df: pd.DataFrame, info_df: pd.DataFrame=None,
                            no_good_no_domain: bool=False) -> List[float]:
    """ Assign weights based on domain's sizes for a node's children, ignoring restart nodes """

    if info_df is None:
        raise ValueError("No information on nodes given!")
    elif info_df.index.name != 'NodeID':
        raise ValueError("Info dataframe must be indexed by node id")
    elif nodes_df['Status'].isin([3]).sum() > 0:
        raise ValueError('Nodes_df must consist only of valid nodes')


    kids = nodes_df[nodes_df['ParentID'] == node_id]
    domains = info_df.loc[kids.index, 'Info'].apply(get_domain_size)
    return (domains / domains.sum()).to_list()

#############################
#### ASSIGNMENT FUNCTION ####
#############################

def assign_weights(nodes_df: pd.DataFrame, weight_scheme: 'function', **kwargs) -> None:
    """
    Assigns weight = 1 to root and propogates that weight down using a weighting scheme.
    Nodes which are backjumped over and pruned are ignored.
    
    Parameter:
        - nodes_df: dataframe of nodes, indexed by nodes_id
        - weight_scheme: function that takes in the current node_id
                         and outputs a list of weights for its children
        - kwargs: keyword arguments required for the weight scheme
    """
    
    nodes_df['NodeWeight'] = 0
    invalid_status = [3] # pruned nodes that are jumped over without being considered
    valid_df = pd.DataFrame.copy(nodes_df[~nodes_df['Status'].isin(invalid_status)])
    valid_df.loc[0, 'NodeWeight'] = 1 # root node has weight 1

    # propogate weights down
    for i in valid_df[valid_df['NKids'] > 0].index:

        par_weight = valid_df.loc[i, 'NodeWeight']
        children = valid_df[valid_df['ParentID'].isin([i])]
        
        if children.shape[0] == 0:
            continue

        weights = weight_scheme(i, valid_df, **kwargs)
        assert len(weights) == children.shape[0]
        assert abs(1 - sum(weights)) < EPSILON
        
        valid_df.loc[children.index, 'NodeWeight'] = par_weight * np.array(weights) 
        assert abs(valid_df.loc[valid_df['ParentID'] == i, 'NodeWeight'].sum() - par_weight) < EPSILON

    assert abs(valid_df.loc[valid_df['Status'].isin({0, 1, 3}), 'NodeWeight'].sum() - 1) < EPSILON

    # assign valid back to nodes, default invalid status nodes' weights to 0
    # this preserves the postconditions, as sum of children is unchanged
    nodes_df.loc[valid_df.index, 'NodeWeight'] = valid_df['NodeWeight']
    nodes_df['NodeWeight'].fillna(0, inplace=True)

if __name__ == '__main__':
    # ### do testing here
    # import time
    # import pathlib

    # start = time.time()
    # fig, axs = plt.subplots(nrows=3, ncols=9, figsize=(9, 27))

    # tree_files = [tree]

    # for i in range(3 * 9):
    #     ax = axs[i // 3][i % 3]
    #     nodes_df = to_df(tree.format(i), 'nodes').set_index('NodeID')
    #     assign_weights(nodes_df, uniform_scheme)
    #     ax.title.set_text(f'golomb_{i}_nodes={nodes_df.shape[0]}')
    #     plot_goodness(nodes_df, axis=ax)
    
    # print("Time taken: {}".format(time.time() - start))
    # plt.show()
    pass

