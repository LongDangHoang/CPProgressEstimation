"""
Implements an offline tree-weight estimation or a constrained search tree
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
# import logging

from multiprocessing import Pool, Array
from helper import get_domain_size, find_split_variable, calculate_subtree_size, to_sqlite, EPSILON
from pathlib import Path
from os import cpu_count

from typing import List

###########################
#### WEIGHTING SCHEMES ####
###########################

SUM_TO_1 = {'uniform_scheme', 'domain_scheme', 'subtreeSize_scheme', 'true_scheme'}
PARALLEL_SAFE = {'uniform_scheme', 'domain_scheme', 'searchSpace_scheme', 'true_scheme', 
    'zero_scheme', 'test_scheme', 'one_scheme', 'sumK_scheme'
}

def make_weight_scheme(scheme_name: str, **kwargs):

    if scheme_name.upper() == 'UNIFORM_SCHEME':
        return UniformScheme()
    elif scheme_name.upper() == 'DOMAIN_SCHEME':
        return DomainScheme(**kwargs)
    elif scheme_name.upper() == 'SEARCHSPACE_SCHEME':
        return SearchSpaceScheme(**kwargs)
    elif scheme_name.upper() == 'SUBTREESIZE_SCHEME':
        return SubtreeSizeScheme(**kwargs)
    elif scheme_name.upper() == 'TRUE_SCHEME':
        return TrueScheme(**kwargs)
    elif scheme_name.upper() == 'ZERO_SCHEME':
        return ZeroScheme()
    elif scheme_name.upper() == 'TEST_SCHEME':
        return TestScheme(**kwargs)
    elif scheme_name.upper() == 'ONE_SCHEME':
        return OneScheme()
    elif scheme_name.upper() == 'SUMK_SCHEME':
        return SumKScheme(**kwargs)

##################
## TEST SCHEMES ##
##################

class TrueScheme():
    """
    Mainly scheme to test correctness of methods
    Value computed should match true subtree size absolutely
    """

    def __init__(self, nodes_df: pd.DataFrame=None, tree: str=None, **kwargs):
        
        if nodes_df is None:
            raise ValueError('Cannot calculate true scheme without nodes_df')

        if 'SubtreeSize' not in nodes_df.columns:
            calculate_subtree_size(nodes_df)
            if tree:
                to_sqlite(nodes_df, tree)

    def get_weight(self, node_id: int, nodes_df: pd.DataFrame):

        kids = nodes_df[nodes_df['ParentID'] == node_id]
        weights = kids['SubtreeSize'] / (nodes_df.loc[node_id, 'SubtreeSize'] - 1) # subtree size include the root node
        
        assert abs(1 - sum(weights)) < EPSILON, 'Sum of weights not close to 1!'
        return weights

class ZeroScheme():
    """
    Mainly scheme to test correctness of methods
    Value computed should all be 0
    """
    def get_weight(self, node_id: int, nodes_df: pd.DataFrame):
        return [0] * (nodes_df['ParentID'] == node_id).sum()

class OneScheme():
    """
    All children's weights are identical to parent's
    """
    def get_weight(self, node_id: int, nodes_df: pd.DataFrame):
        return [1] * (nodes_df['ParentID'] == node_id).sum()

class TestScheme():
    """
    Reversed test case where weights are assigned to match randomly assigned node weights
    Value computed should match randomly assigned node weights
    """

    def __init__(self, nodes_df: pd.DataFrame=None, **kwargs):

        if 'RandomTrueNodeWeight' not in nodes_df.columns:
            nodes_df['RandomTrueNodeWeight'] = np.random.random(len(nodes_df))
            nodes_df.loc[0, 'RandomTrueNodeWeight'] = 1

    def get_weight(self, node_id: int, nodes_df: pd.DataFrame):
        kids = nodes_df[nodes_df['ParentID'] == node_id]
        par_w = nodes_df.loc[node_id, 'RandomTrueNodeWeight']
        return (kids['RandomTrueNodeWeight'] / par_w).values

class SumKScheme():
    """
    Sum of children is k times parent's weight, where
    k is random posi    d   tive float
    """
    def __init__(self, k: float=None, **kwargs):
        assert 1 >= k >= 0
        self.k = k
    
    def get_weight(self, node_id: int, nodes_df: pd.DataFrame):
        num_kids = (nodes_df['ParentID'] == node_id).sum()
        weights = np.random.random(num_kids)
        return weights + (self.k / num_kids) - np.mean(weights) # average to self.k / num_kids

####################
## WEIGHT SCHEMES ##
####################

class UniformScheme():
    def get_weight(self, node_id: int, nodes_df: pd.DataFrame, **kwargs) -> List[float]:
        """ Assign uniform weights for a node's children """
        
        num_kids = nodes_df[nodes_df['ParentID'].isin([node_id])].shape[0]
        weights = [1 / num_kids] * num_kids

        assert abs(1 - sum(weights)) < EPSILON, 'Sum of weights not close to 1!'
        return weights

class DomainScheme():

    def __init__(self, info_df: pd.DataFrame=None, **kwargs):
        if info_df is None:
            raise ValueError("No information on nodes given!")
        elif info_df.index.name != 'NodeID':
            raise ValueError("Info dataframe must be indexed by node id")

        self.info_df = info_df

    def get_weight(self, node_id: int, nodes_df: pd.DataFrame) -> List[float]:
        """ Assign weights based on domain's sizes for a node's children, ignoring restart nodes """

        kids = nodes_df[nodes_df['ParentID'] == node_id]
        domains = self.info_df.loc[kids.index, 'Info'].apply(get_domain_size)
        weights = (domains / domains.max()) / (domains / domains.max()).sum() # divide by max to avoid overflow
        
        assert abs(1 - sum(weights)) < EPSILON, 'Sum of weights not close to 1!'
        return weights

class SearchSpaceScheme():
    def __init__(self, tree: str=None, info_df: pd.DataFrame=None, **kwargs):
        # init mappings
        self.mappings = {}
        if tree:
            # if tree is provided, find mapping file
            tree_dir = Path(tree).parent 
            tree_name = str(Path(tree).resolve()).split('/')[-1].strip('.sqlite')
            mapping_file = [f for f in tree_dir.iterdir() if str(f.resolve()).split('/')[-1].strip('.csv') == tree_name]
            # if mapping file is found, use it
            if len(mapping_file) == 1:
                mappings = pd.read_csv(mapping_file[0], header=None, quotechar="'")
                self.mappings = mappings.set_index(0).to_dict()[1]
                self.mappings.update(mappings.set_index(1).to_dict()[0])

        # init info_df
        if info_df is None:
            raise ValueError("No information on nodes given!")
        elif info_df.index.name != 'NodeID':
            raise ValueError("Info dataframe must be indexed by node id")
        self.info_df = info_df

    def get_weight(self, node_id: int, nodes_df: pd.DataFrame) -> List[float]:
        """
        Assign weights as percentage of parent's domain
        """

        kids = nodes_df[nodes_df['ParentID'] == node_id]
        cands, mappings, par_domain, children_domain = find_split_variable(node_id, nodes_df, self.info_df, self.mappings)

        if len(cands) == 1:
            # use split_variable
            split_variable = cands[0]
            par_size = len(par_domain[split_variable])
            children_size = np.array([len(child_domain[split_variable]) for child_domain in children_domain])
            weights = children_size / par_size
        else:
            # use uniform
            weights = [1 / kids.shape[0]] * kids.shape[0]
        
        # assert abs(1 - sum(weights)) < EPSILON, 'Sum of weights not close to 1!'
        return weights

class SubtreeSizeScheme():

    def __init__(self, assign_in_dfs_order: bool=False, decay_rate: float=0.5, 
                nodes_df: pd.DataFrame=None, tree: str=None, **kwargs):
        
        if assign_in_dfs_order is False:
            raise ValueError('Cannot use subtree size scheme if not assigning weights in DFS ordering')
        elif nodes_df is None:
            raise ValueError('Cannot use substree size scheme without nodes_df initializer')

        if 'SubtreeSize' not in nodes_df.columns:
            calculate_subtree_size(nodes_df)
            if tree:
                to_sqlite(nodes_df, tree)

        self.characteristic_w = 1 # initialize the characteristic weight that describes distribution of subtree sizes
        self.previous_root = -1 # initlaize the previous root
        self.path = [] # track path to current nodeassign_in_dfs_order
        self.decay_rate = decay_rate

    def get_weight(self, node_id: int, nodes_df: pd.DataFrame) -> List[float]:
        """
        Assign weight based on characteristic weight of siblings' subtree sizes
        """

        if self.previous_root != nodes_df.loc[node_id, 'ParentID']:
            # we have moved to a different subtree than previous node,
            # i.e. there should be an update on characteristic w
            self.__update_characteristic_w(node_id, nodes_df)
        else:
            self.path.append(node_id)
            self.previous_root = node_id

        kids = nodes_df[nodes_df['ParentID'] == node_id]
        relative_sizes = np.array([1 * (self.characteristic_w ** -i) for i in range(kids.shape[0])])
        weights = relative_sizes / relative_sizes.sum()

        assert abs(1 - weights.sum()) < EPSILON
        return weights

    def __update_characteristic_w(self, current_node: int, nodes_df: pd.DataFrame):
        """
        Update the internal value for characteristic w
        """
        # find path from previous root
        current_node_parent = nodes_df.loc[current_node, 'ParentID']
        while self.path[-1] != current_node_parent:
            par = self.path.pop()
            par_subtree_w = self.__fit_characteristic(par, nodes_df)
            self.characteristic_w = self.characteristic_w * self.decay_rate + (1 - self.decay_rate) * par_subtree_w

        self.path.append(current_node) # go down current node now

    def __fit_characteristic(self, node_id: int, nodes_df: pd.DataFrame) -> float:
        """ Return w that best fit node_id's subtree """
        truth = nodes_df[nodes_df['ParentID'] == node_id]['SubtreeSize'].values
        a_s = [truth[i] / truth[i - 1] for i in range(1, len(truth))]
        return np.mean(a_s) if len(a_s) > 0 else 0

#############################
#### ASSIGNMENT FUNCTION ####
#############################

def make_weight_parallel(j: int, use_parallel: bool=False):
    """
    Use a global variable to store weights generated by this function in a sparse matrix
    """
    global parallel_matrix, wa_state
    try:
        assert parallel_matrix is not None
        assert wa_state is not None
        
        valid_df = wa_state['valid_df']
        ws = wa_state['ws']
        weight_scheme = wa_state['weight_scheme']
        weight_colname = wa_state['weight_colname']

    except NameError:
        raise ValueError("Cannot run parallel weights creation method as no global storage for them exist!")
    except KeyError:
        raise ValueError("Weight assignment state not populated for parallel")
        
    par_weight = valid_df.loc[j, weight_colname]
    children = valid_df[valid_df['ParentID'].isin([j])]

    if children.shape[0] == 0:
        return

    weights = np.array(ws.get_weight(j, valid_df))
    # print(f"At node {j}, weights retrieved are {list(weights)}")
    assert len(weights) == children.shape[0]

    # store retrieved weights
    for i in range(len(children.index)):
        child_id = children.index[i]
        parallel_matrix['data'][child_id] = weights[i]
        parallel_matrix['row'][child_id] = j
        
def assign_weight(nodes_df: pd.DataFrame, weight_scheme: str, 
        weight_colname: str='NodeWeight',
        assign_in_dfs_order: bool=True,
        use_parallel: bool=False,
        **kwargs) -> None:
    """
    Assigns weight = 1 to root and propogates that weight down using a weighting scheme.
    Nodes which are backjumped over and pruned are ignored.
    
    Parameter:
        - nodes_df: dataframe of nodes, indexed by nodes_id
        - weight_scheme: name of function defined in WeightScheme class
        - kwargs: keyword arguments required for the weight scheme
    """
    
    ws = make_weight_scheme(weight_scheme, assign_in_dfs_order=assign_in_dfs_order, nodes_df=nodes_df, **kwargs)
    nodes_df[weight_colname] = 0
    invalid_status = [3] # pruned nodes that are jumped over without being considered
    valid_df = pd.DataFrame.copy(nodes_df[~nodes_df['Status'].isin(invalid_status)])
    valid_df.loc[0, weight_colname] = 1 # root node has weight 1

    if assign_in_dfs_order:
        assert 'DFSOrdering' in valid_df.columns
        index_lst = valid_df[valid_df['NKids'] > 0].sort_values('DFSOrdering').index
    else:
        index_lst = valid_df[valid_df['NKids'] > 0].index
    
    # propogate weights down
    if weight_scheme not in PARALLEL_SAFE or not use_parallel:
        for j in index_lst:
            par_weight = valid_df.loc[j, weight_colname]
            children = valid_df[valid_df['ParentID'].isin([j])]

            if children.shape[0] == 0:
                continue

            weights = ws.get_weight(j, valid_df)
            # print(f"At node {j}, weights retrieved are {list(weights)}")
            assert len(weights) == children.shape[0]

            valid_df.loc[children.index, weight_colname] = par_weight * np.array(weights) 
            if weight_scheme in SUM_TO_1: # schemes requiring sum to 1
                assert abs(valid_df.loc[valid_df['ParentID'] == j, weight_colname].sum() - par_weight) < EPSILON
    
    else:
        # do the same things as above but stores weights in a global state array instead
        global parallel_matrix # declare variable in global scope
        parallel_matrix = {} # reset global state for new run
        parallel_matrix['data'] = Array('d', nodes_df.shape[0])
        parallel_matrix['row'] = Array('i', nodes_df.shape[0])
        
        global wa_state # state for weight assignment
        wa_state = {} # reset state
        wa_state['valid_df'] = valid_df
        wa_state['ws'] = ws
        wa_state['weight_colname'] = weight_colname
        wa_state['weight_scheme'] = weight_scheme

        try:
            pool = Pool(processes=cpu_count())
            pool.map(make_weight_parallel, index_lst)
        finally:
            pool.close()

        # rows are parent, columns are node ids, data is weight of child for parent
        weights = sparse.csr_matrix((parallel_matrix['data'], (parallel_matrix['row'], range(nodes_df.shape[0]))), 
                    shape=(nodes_df.shape[0], nodes_df.shape[0]))
        running = weights

        # > 0 works for normal weight, but test weighting schemes include negative weights
        while np.any((running[0] != 0).data):
            temp = running[0].tocoo()
            col = temp.col[temp.data != 0]
            valid_df.loc[col, weight_colname] = temp.data[temp.data != 0]
            running = running * weights

        valid_df.loc[0, weight_colname] = 1
        
        del parallel_matrix
        del wa_state # free up memory
        
    if weight_scheme in SUM_TO_1: # schemes requiring sum to 1
        assert abs(valid_df.loc[valid_df['Status'].isin({0, 1, 3}), weight_colname].sum() - 1) < EPSILON

    # assign valid back to nodes, default invalid status nodes' weights to 0
    # this preserves the postconditions, as sum of children is unchanged
    nodes_df.loc[valid_df.index, weight_colname] = valid_df[weight_colname]
    nodes_df[weight_colname].fillna(0, inplace=True)

