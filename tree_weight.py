"""
Implements an offline tree-weight estimation or a constrained search tree
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

from abc import ABC, abstractmethod 
from multiprocessing import Pool, Array
from pandarallel import pandarallel
from helper import get_domain_size, find_split_variable, calculate_subtree_size, to_sqlite, is_unequal_split, get_parent_column, parse_info_string, EPSILON
from pathlib import Path
from os import cpu_count

from typing import List

###########################
#### WEIGHTING SCHEMES ####
###########################

SUM_TO_1 = {'uniform_scheme', 'domain_scheme', 'subtreeSize_scheme', 'searchSpace_scheme', 'true_scheme'}
PARALLEL_SAFE = {'uniform_scheme', 'domain_scheme', 'searchSpace_scheme', 'true_scheme', 
    'zero_scheme', 'test_scheme', 'one_scheme', 'sumK_scheme'
}

def make_weight_scheme(state_dict: dict):

    scheme_name = state_dict['weight_scheme']

    if scheme_name.upper() == 'UNIFORM_SCHEME':
        return UniformScheme()
    elif scheme_name.upper() == 'DOMAIN_SCHEME':
        return DomainScheme(state_dict)
    elif scheme_name.upper() == 'SEARCHSPACE_SCHEME':
        return SearchSpaceScheme(state_dict)
    elif scheme_name.upper() == 'SUBTREESIZE_SCHEME':
        return SubtreeSizeScheme(state_dict)
    elif scheme_name.upper() == 'TRUE_SCHEME':
        return TrueScheme(state_dict)
    elif scheme_name.upper() == 'ZERO_SCHEME':
        return ZeroScheme()
    elif scheme_name.upper() == 'TEST_SCHEME':
        return TestScheme(state_dict)
    elif scheme_name.upper() == 'ONE_SCHEME':
        return OneScheme()
    elif scheme_name.upper() == 'SUMK_SCHEME':
        return SumKScheme(state_dict)

################
## INTERFACES ##
################

class GenericWeightScheme(ABC):

    @abstractmethod
    def get_weight(self, node_id: int, nodes_df: pd.DataFrame):
        pass

    def test_weight(self, weights: np.ndarray):
        pass
    
    def test_allweight(self, nodes_df: np.ndarray, weight_colname: str):
        pass

    def tear_down(self):
        pass

class ParallelizableScheme(ABC):
    
    @abstractmethod
    def get_weight_parallel(self, nodes_df: pd.DataFrame):
        pass

class SumToOneScheme(GenericWeightScheme):

    def test_weight(self, weights: np.ndarray):
        assert abs(1 - sum(weights)) < EPSILON 
    
    def test_allweight(self, nodes_df: np.ndarray, weight_colname: str):
        assert abs(1 - nodes_df[nodes_df['Status'].isin({0, 1})][weight_colname].sum()) < EPSILON

##################
## TEST SCHEMES ##
##################

class TrueScheme(SumToOneScheme, ParallelizableScheme):
    """
    Mainly scheme to test correctness of methods
    Value computed should match true subtree size absolutely
    """

    def __init__(self, state_dict: dict):
        nodes_df = state_dict['nodes_df']
        if 'SubtreeSize' not in nodes_df.columns:
            calculate_subtree_size(nodes_df)
            if tree:
                to_sqlite(nodes_df, tree)

    def get_weight(self, node_id: int, nodes_df: pd.DataFrame):
        kids = nodes_df[nodes_df['ParentID'] == node_id]
        weights = kids['SubtreeSize'] / (nodes_df.loc[node_id, 'SubtreeSize'] - 1) # subtree size include the root node
        return weights
    
    def get_weight_parallel(self, nodes_df: pd.DataFrame):
        assert 0 in nodes_df.index
        res_df = pd.DataFrame.copy(nodes_df.reset_index()[['NodeID', 'ParentID', 'SubtreeSize']]).iloc[1:, :]
        parent_subtreesize = get_parent_column('SubtreeSize', nodes_df)
        res_df.loc[:, 'Weight'] = res_df['SubtreeSize'] / (parent_subtreesize - 1)
        return res_df

class ZeroScheme(GenericWeightScheme, ParallelizableScheme):
    """
    Mainly scheme to test correctness of methods
    Value computed should all be 0
    """
    def get_weight(self, node_id: int, nodes_df: pd.DataFrame):
        return [0] * (nodes_df['ParentID'] == node_id).sum()
    
    def get_weight_parallel(self, nodes_df: pd.DataFrame):
        assert 0 in nodes_df.index
        res_df = pd.DataFrame.copy(nodes_df.reset_index()[['NodeID', 'ParentID']].iloc[1:, :])
        res_df['Weight'] = 0
        return res_df

class OneScheme(GenericWeightScheme, ParallelizableScheme):
    """
    All children's weights are identical to parent's
    """
    def get_weight(self, node_id: int, nodes_df: pd.DataFrame):
        return [1] * (nodes_df['ParentID'] == node_id).sum()
    
    def get_weight_parallel(self, nodes_df: pd.DataFrame):
        assert 0 in nodes_df.index
        res_df = pd.DataFrame.copy(nodes_df.reset_index()[['NodeID', 'ParentID']].iloc[1:, :])
        res_df['Weight'] = 1
        return res_df

    def test_weight(self, weights: np.ndarray):
        assert len(np.unique(weights)) == 1

class TestScheme(GenericWeightScheme, ParallelizableScheme):
    """
    Reversed test case where weights are assigned to match randomly assigned node weights
    Value computed should match randomly assigned node weights
    """

    def __init__(self, state_dict: dict):
        nodes_df = state_dict['nodes_df']
        if 'RandomTrueNodeWeight' not in nodes_df.columns:
            nodes_df['RandomTrueNodeWeight'] = np.random.random(len(nodes_df))
            nodes_df.loc[0, 'RandomTrueNodeWeight'] = 1

    def get_weight(self, node_id: int, nodes_df: pd.DataFrame):
        kids = nodes_df[nodes_df['ParentID'] == node_id]
        par_w = nodes_df.loc[node_id, 'RandomTrueNodeWeight']
        return (kids['RandomTrueNodeWeight'] / par_w).values
    
    def get_weight_parallel(self, nodes_df: pd.DataFrame):
        assert 0 in nodes_df.index
        parent_trueweight = get_parent_column('RandomTrueNodeWeight', nodes_df)
        res_df = pd.DataFrame.copy(nodes_df).iloc[1:, :].loc[:, ['ParentID', 'RandomTrueNodeWeight']]
        res_df.loc[:, 'Weight'] = res_df['RandomTrueNodeWeight'] / parent_trueweight
        return res_df.reset_index().drop(columns=['RandomTrueNodeWeight'])



class SumKScheme(GenericWeightScheme, ParallelizableScheme):
    """
    Sum of children is k times parent's weight, where
    k is random positive float
    """
    def __init__(self, state_dict: dict):
        if 'k' not in state_dict:
            print('Default to k = 0.5 as k is not provided for sumKScheme')
            k = 0.5
        else:
            k = state_dict['k']
            assert 1 >= k >= 0, 'Provided K for sumKScheme must be between 0 and 1'
        self.k = k
    
    def get_weight(self, node_id: int, nodes_df: pd.DataFrame):
        num_kids = (nodes_df['ParentID'] == node_id).sum()
        weights = np.random.random(num_kids)
        return weights + (self.k / num_kids) - np.mean(weights) # average to self.k / num_kids

    def get_weight_parallel(self, nodes_df: pd.DataFrame):
        assert 0 in nodes_df.index
        res_df = pd.DataFrame.copy(nodes_df.iloc[1:, :])
        res_df['Weight'] = np.random.random(res_df.shape[0])
        parent_mean = res_df.groupby('ParentID').mean()['Weight']
        parent_count = res_df.groupby('ParentID').count()['Weight']

        # index swap
        res_df = res_df.reset_index().set_index('ParentID')
        res_df.loc[:, 'Mean'] = parent_mean
        res_df.loc[:, 'Count'] = parent_count
        res_df = res_df.reset_index().set_index('NodeID')
        res_df.loc[:, 'Weight'] = res_df['Weight'] - res_df['Mean'] + self.k / res_df['Count']
        res_df = res_df.reset_index()[['NodeID', 'ParentID', 'Weight']]
        assert np.all(np.abs(res_df.groupby(['ParentID']).sum()['Weight'] - self.k) < EPSILON)
        return res_df

####################
## WEIGHT SCHEMES ##
####################

class UniformScheme(SumToOneScheme, ParallelizableScheme):
    def get_weight(self, node_id: int, nodes_df: pd.DataFrame) -> List[float]:
        """ Assign uniform weights for a node's children """
        
        num_kids = nodes_df[nodes_df['ParentID'].isin([node_id])].shape[0]
        weights = [1 / num_kids] * num_kids

        assert abs(1 - sum(weights)) < EPSILON, 'Sum of weights not close to 1!'
        return weights

    def get_weight_parallel(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a dataframe of three columns:
            - NodeID
            - ParentID
            - Weight - weight of the node compares to its parent
        """
        assert 0 in nodes_df.index
        count = nodes_df.iloc[1:, :].groupby('ParentID').count()['NKids']
        temp = pd.DataFrame(nodes_df.iloc[1:, :].reset_index().set_index('ParentID')['NodeID'])
        temp['Weight'] = count
        weights = temp.reset_index().set_index('NodeID')
        weights.loc[:, 'Weight'] = 1 / weights['Weight']
        return weights.reset_index()

class SearchSpaceScheme(SumToOneScheme, ParallelizableScheme):
    def __init__(self, state_dict: dict):
        # init mappings
        self.mapdict = {}
        self.mappings = pd.DataFrame({'InfoDFName': {}, 'NodesDFName': {}})
        tree = state_dict['tree'] if 'tree' in state_dict else None
        info_df = state_dict['info_df'] if 'info_df' in state_dict else None

        if tree:
            # if tree is provided, find mapping file
            mapping_file = tree.replace('.sqlite', '.paths')
            # if mapping file is found, use it
            if Path(mapping_file).exists():
                mappings = pd.read_csv(mapping_file, sep='\t', header=None, quotechar="'")
                self.mapdict = mappings.set_index(0).to_dict()[1]
                self.mapdict.update(mappings.set_index(1).to_dict()[0])
                self.mappings = mappings.rename(columns={0: 'InfoDFName', 1: 'NodesDFName'})

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
        cands, mappings, par_domain, children_domain = find_split_variable(node_id, nodes_df, self.info_df, self.mapdict)

        if len(cands) == 1:
            # use split_variable
            split_variable = cands[0]
            par_size = len(par_domain[split_variable])
            
            # return 1/par_size, 1 - 1/par_size for unequal split, else uniform
            if is_unequal_split(nodes_df, kids.index):
                # check whether first kid is = or !=
                first_eq = kids.iloc[0,:]['Label'].find('!') == -1
                if first_eq:
                    weights = [1 / par_size, 1 - 1 / par_size]
                else:
                    weights = [1 - 1 / par_size, 1 / par_size]
            else:
                weights = [1 / kids.shape[0]] * kids.shape[0]
        else:
            # use uniform
            weights = [1 / kids.shape[0]] * kids.shape[0]
        
        return weights
    
    def get_weight_parallel(self, nodes_df: pd.DataFrame):
        """
        Return a dataframe of three columns:
            - NodeID
            - ParentID
            - Weight - weight of the node compares to its parent
        """
        if len(self.mappings) == 0:
            raise ValueError("Cannot run search space scheme in parallel without known mappings beforehand")

        # use vectorized string to get names of labels
        node_label = nodes_df['Label'].str.split('=', expand=True)[0].str.strip('! ').rename('LabelVarname')
        node_label = node_label[node_label != '']
        # use indexing to map nodes' labels names to info's domain names
        info_label = node_label.reset_index().set_index('LabelVarname')
        try:
            info_label['LabelName'] = self.mappings.set_index('NodesDFName')['InfoDFName']
        except:
            breakpoint()
        info_label = info_label.set_index('NodeID')
        # assign parent id to extract parent's domain for split variable
        info_label['ParentID'] = nodes_df['ParentID']
        info_label = info_label.drop_duplicates('ParentID')
        info_label = info_label.set_index('ParentID').reset_index().rename(columns={'ParentID': 'NodeID'}).set_index('NodeID')

        info_df['Label'] = info_label['LabelName']
        parent_domain_size = info_df[info_df['Label'].notna()].parallel_apply(lambda row: len(parse_info_string(row['Info'], early_stop=row['Label'])), axis=1) 
        del info_df['Label']

        nodes_df['HasUnequalSplit'] = nodes_df['Label'].str.find('!') > 0 # will label null domains as equal split, which works for us
        # transform index from nodeid to parentid to perform computation with 
        assert 0 in nodes_df.index
        weights = nodes_df[['ParentID', 'HasUnequalSplit']].reset_index().set_index('ParentID').iloc[1:,:]
        weights.loc[:, 'ParentDomainSize'] = parent_domain_size
        weights = weights.reset_index().set_index('NodeID')
        # if is unequal split, automatically receiver the heavier share, while if equal split, receive equal share
        weights['Weight'] = 1 / weights['ParentDomainSize'] + weights['HasUnequalSplit'] * (1 - 2 / weights['ParentDomainSize'])
        weights = weights.drop(columns=['HasUnequalSplit', 'ParentDomainSize']).reset_index()
        return weights

    def tear_down(self):
        # write mappings to a csv file
        if self.tree is not None and len(self.mappings) == 0 and len(self.mapdict) > 0:
            self.mappings = pd.DataFrame.from_dict(self.mapdict, orient='index', columns='InfoDFLabel') 
            self.mappings = self.mappings.reset_index().rename(columns={'index': 'NodesDFLabel'})
            self.mappings.to_csv(self.tree.replace('.sqlite', '.csv'), index=False, header=None)

class SubtreeSizeScheme(SumToOneScheme):

    def __init__(self, state_dict: dict):
        
        # get variables from dict
        assign_in_dfs_order = state_dict['assign_in_dfs_order']
        nodes_df = state_dict['nodes_df'] if 'nodes_df' in state_dict else None

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
        self.decay_rate = state_dict['decay_rate'] if 'decay_rate' in state_dict else 0.7

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

def assign_weight(state_dict: dict) -> None:
    """
    Assigns weight = 1 to root and propogates that weight down using a weighting scheme.
    Nodes which are backjumped over and pruned are ignored.
    
    Parameter:
        - state_dict: dictionary of parameters
    """
    
    # extract and initialize variables using state_dict
    weight_colname = state_dict['weight_colname']
    nodes_df = state_dict['nodes_df']
    ws = make_weight_scheme(state_dict)

    # init 0 column
    nodes_df[weight_colname] = 0
    invalid_status = [3] # pruned nodes that are jumped over without being considered
    valid_df = pd.DataFrame.copy(nodes_df[~nodes_df['Status'].isin(invalid_status)])
    valid_df.loc[0, weight_colname] = 1 # root node has weight 1

    # init travel index
    if state_dict['assign_in_dfs_order']:
        assert 'DFSOrdering' in valid_df.columns, 'Cannot assign in DFS order without known dfs ordering in dataframe'
        index_lst = valid_df[valid_df['NKids'] > 0].sort_values('DFSOrdering').index
    else:
        index_lst = valid_df[valid_df['NKids'] > 0].index
    
    # main weight_propogation 
    if state_dict['weight_scheme'] not in PARALLEL_SAFE or not state_dict['use_parallel']:
        for j in index_lst:
            par_weight = valid_df.loc[j, weight_colname]
            children = valid_df[valid_df['ParentID'].isin([j])]

            if children.shape[0] == 0:
                continue

            weights = ws.get_weight(j, valid_df)
            assert len(weights) == children.shape[0]
            ws.test_weight(weights)
            valid_df.loc[children.index, weight_colname] = par_weight * np.array(weights) 
    else:
        pandarallel.initialize(verbose=False)
        weights = ws.get_weight_parallel(valid_df)
        weights = sparse.csr_matrix((weights['Weight'].to_numpy(), (weights['ParentID'].to_numpy(), weights['NodeID'].to_numpy())),
                shape=(nodes_df.shape[0], nodes_df.shape[0]))
            
        running = weights
        # > 0 works for normal weight, but test weighting schemes include negative weights
        while np.any((running[0] != 0).data):
            temp = running[0].tocoo()
            col = temp.col[temp.data != 0]
            valid_df.loc[col, weight_colname] = temp.data[temp.data != 0]
            running = running * weights

        valid_df.loc[0, weight_colname] = 1
        
    # assign valid back to nodes, default invalid status nodes' weights to 0
    nodes_df.loc[valid_df.index, weight_colname] = valid_df[weight_colname]
    nodes_df[weight_colname].fillna(0, inplace=True)
    ws.test_allweight(nodes_df, weight_colname)
    ws.tear_down() 