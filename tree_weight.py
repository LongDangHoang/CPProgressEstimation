"""
Implements an offline tree-weight estimation or a constrained search tree

4. Measure weight as we go along node_id and compare it with actual number of nodes we eventually visit
"""

import pandas as pd
import matplotlib.pyplot as plt
from helper import to_df, make_dfs_ordering, plot_goodness, EPSILON

from typing import List

###########################
#### WEIGHTING SCHEMES ####
###########################


def uniform_scheme(node_id: int, nodes_df: pd.DataFrame) -> List[float]:
    """ Assign uniform weights for a node's children """
    return [1 / nodes_df.loc[node_id, 'NKids'] for i in range(nodes_df.loc[node_id, 'NKids'])]

def domain_weight_scheme(node_id: int, nodes_df: pd.DataFrame, info_df: pd.DataFrame=None) -> List[float]:
    """ Assign weights based on domain's sizes for a node's children """

    if info_df is None:
        raise ValueError("No information on nodes given!")

    children_df = nodes_df[nodes_df['ParentID'] == node_id]
    children_df['DomainSize'] = info_df.loc[children_df.index, 'Info'].apply(get_domain_size)
    return (children_df['DomainSize'] / children_df['DomainSize'].sum()).to_list()

#############################
#### ASSIGNMENT FUNCTION ####
#############################

def assign_weights(nodes_df: pd.DataFrame, weight_scheme: 'function', **kwargs) -> None:
    """
    Assigns weight = 1 to root and propogates that weight down using a weighting scheme
    
    Parameter:
        - nodes_df: dataframe of nodes, indexed by nodes_id
        - weight_scheme: function that takes in the current node_id
                         and outputs a list of weights for its children
        - kwargs: keyword arguments required for the weight scheme
    """
    
    nodes_df['NodeWeight'] = 0
    nodes_df.loc[0, 'NodeWeight'] = 1 # root node has weight 1

    # propogate weights down
    for i in nodes_df[nodes_df['NKids'] > 0].index:

        par_weight = nodes_df.loc[i, 'NodeWeight']
        num_kids = nodes_df.loc[i, 'NKids']
        if num_kids > 0:
            weights = weight_scheme(i, nodes_df, **kwargs)
            assert len(weights) == num_kids
            assert abs(1 - sum(weights)) < EPSILON
            nodes_df.loc[nodes_df['ParentID'] == i, 'NodeWeight'] = par_weight * \
                                   nodes_df[nodes_df['ParentID'] == i]['Alternative'].apply(lambda x: weights[x]) 
            assert abs(nodes_df.loc[nodes_df['ParentID'] == i, 'NodeWeight'].sum() - par_weight) < EPSILON
    
    assert abs(nodes_df.loc[nodes_df['Status'].isin({0, 1, 3}), 'NodeWeight'].sum() - 1) < EPSILON

if __name__ == '__main__':
    ### do testing here
    import time

    start = time.time()
    tree = './trees/golomb_{}.sqlite'
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    for i in range(3, 8):
        ax = axs[(i - 3) // 3][(i - 3) % 3]
        nodes_df = to_df(tree.format(i), 'nodes').set_index('NodeID')
        assign_weights(nodes_df, uniform_scheme)
        ax.title.set_text(f'golomb_{i}_nodes={nodes_df.shape[0]}')
        plot_goodness(nodes_df, axis=ax)
    
    print("Time taken: {}".format(time.time() - start))
    plt.show()
    


