"""
Implements an offline tree-weight estimation or a constrained search tree

4. Measure weight as we go along node_id and compare it with actual number of nodes we eventually visit
"""

import pandas as pd
from helper import to_df

def uniform_scheme(node_id: int, nodes_df: pd.DataFrame) -> float:
    """ Assign uniform weight """
    return 1 / nodes_df.loc[nodes_df.loc[node_id, 'ParentID'], 'NKids']

def assign_weights(nodes_df: pd.DataFrame, weight_scheme: 'function') -> None:
    """
    Assigns weight = 1 to root and propogates that weight down using a weighting scheme
    
    Parameter:
        - nodes_df: dataframe of nodes, indexed by nodes_id
        - weight_scheme: function that takes in the current node_id
                         and outputs the weight it has relative to its parent node
    """
    
    nodes_df['NodeWeight'] = 0
    nodes_df.loc[0, 'NodeWeight'] = 1 # root node has weight 1
    next_ids = nodes_df[nodes_df['ParentID'] == 0].index

    # propogate weights down
    while len(next_ids) > 0:
        node = next_ids.pop(0)
        nodes_df.loc[node, 'NodeWeight'] = nodes_df.loc[nodes_df.loc[node, 'ParentID'], :]\
                                            ['NodeWeight'] * weight_scheme(node, nodes_df)
        next_ids = pd.concat(next_ids, nodes_df[nodes_df['ParentID'] == node].index.to_list())

    ## debug:
    print(nodes_df[['NodeID', 'NodeWeight']])
    print("Sum of all leaves weights: ", nodes_df[nodes_df['status'].isin({1, 0, 3})]['NodeWeight'].sum())

    # undo all alter table
    return None

if __name__ == '__main__':
    ### do testing here

    tree = './trees/golomb_3.sqlite'
    nodes_df = to_df(tree, 'nodes').set_index('NodeID')
    


