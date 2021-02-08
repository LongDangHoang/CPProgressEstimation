"""
Implements an offline tree-weight estimation or a constrained search tree

4. Measure weight as we go along node_id and compare it with actual number of nodes we eventually visit
"""

import pandas as pd
import matplotlib.pyplot as plt
from helper import to_df

def uniform_scheme(node_id: int, nodes_df: pd.DataFrame) -> float:
    """ Assign uniform weights """
    return [1 / nodes_df.loc[node_id, 'NKids'] for i in range(nodes_df.loc[node_id, 'NKids'])]

def assign_weights(nodes_df: pd.DataFrame, weight_scheme: 'function') -> None:
    """
    Assigns weight = 1 to root and propogates that weight down using a weighting scheme
    
    Parameter:
        - nodes_df: dataframe of nodes, indexed by nodes_id
        - weight_scheme: function that takes in the current node_id
                         and outputs a list of weights for its children
    """
    
    nodes_df['NodeWeight'] = 0
    nodes_df.loc[0, 'NodeWeight'] = 1 # root node has weight 1
    i = 0

    # propogate weights down
    while i < nodes_df.shape[0] - 1:

        par_weight = nodes_df.loc[i, 'NodeWeight']
        num_kids = nodes_df.loc[i, 'NKids']
        if num_kids > 0:
            weights = uniform_scheme(i, nodes_df)
            assert len(weights) == num_kids
            nodes_df.loc[nodes_df['ParentID'] == i, 'NodeWeight'] = par_weight * \
                                   nodes_df[nodes_df['ParentID'] == i]['Alternative'].apply(lambda x: weights[x]) 
        i += 1

def make_dfs_ordering(nodes_df: pd.DataFrame) -> list:
    """
    Return list of nodeids in the order they were completely solved by
    the depth-first search algorithm
    """
    dfs_ordering = [0]
    boundary = nodes_df[nodes_df['ParentID'] == 0].sort_values('Alternative', ascending=False).index.to_list()

    # run simulated dfs on tree
    while len(boundary) > 0:
        nxt = boundary.pop()
        dfs_ordering.append(nxt)
        boundary.extend(nodes_df[nodes_df['ParentID'] == nxt].sort_values('Alternative', ascending=False).index.to_list())

    return dfs_ordering

def plot_goodness(nodes_df: pd.DataFrame, axis) -> None:
    """
    Plot prediction of tree weight against ground truth
    nodes_df must have node weight already assigned
    """
    if 'NodeWeight' not in nodes_df.columns:
        raise ValueError
    if abs(1 - nodes_df[nodes_df['Status'].isin({1, 0, 3})]['NodeWeight'].sum()) >= 10e-6:
        print(abs(1 - nodes_df[nodes_df['Status'].isin({1, 0, 3})]['NodeWeight'].sum())) 

    dfs_ordering = make_dfs_ordering(nodes_df)

    cumulative = nodes_df.reindex(dfs_ordering).reset_index()
    cumulative = cumulative[cumulative['Status'].isin({0, 1, 3})]['NodeWeight']\
        .cumsum()\
        .reindex(list(range(1, nodes_df.shape[0]+1)), method='ffill')\
        .fillna(0)

    x = range(1, nodes_df.shape[0] + 1)
    axis.plot(pd.Series(x) / nodes_df.shape[0], label='Ground Truth')
    axis.plot(cumulative, label='Uniform weight')
    axis.legend()
    
if __name__ == '__main__':
    ### do testing here
    import time

    start = time.time()
    tree = './trees/golomb_{}.sqlite'
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    for i in range(3, 11):
        ax = axs[(i - 3) // 3][(i - 3) % 3]
        nodes_df = to_df(tree.format(i), 'nodes').set_index('NodeID')
        assign_weights(nodes_df, uniform_scheme)
        ax.title.set_text(f'golomb_{i}_nodes={nodes_df.shape[0]}')
        plot_goodness(nodes_df, axis=ax)
    
    print("Time taken: {}".format(time.time() - start))
    plt.show()

    # tree = './trees/golomb_11.sqlite'
    # nodes_df = to_df(tree, 'nodes').set_index('NodeID')
    # assign_weights(nodes_df, uniform_scheme)
    # fig, ax = plt.subplots(nrows=1, ncols=1, figisze=(6,6))
    # ax.title.set_text(f'golomb_11_nodes={nodes_df.shape[0]}')
    # plot_goodness(nodes_df, axis=ax)
    # plt.show()
    


