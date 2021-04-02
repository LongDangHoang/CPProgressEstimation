"""
helper file
"""

import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import unittest
import os

from itertools import chain
from sqlalchemy import create_engine

EPSILON = 1e-4

def to_df(file: str, table: str) -> pd.DataFrame:
    
    """
    Return pandas' dataframe representation of a table in sqlite file (tree search log)

    Parameters:
        - file: string of the path to the sqlite3 database
        - table: table to be saved  
    Returns:
        - list of dataframes, one for each table in the database

    """

    assert table.upper() in  {'NODES', 'INFO', 'BOOKMARKS', 'NOGOODS'}
    conn = sqlite3.connect(file)
    query = 'SELECT * FROM {}'.format(table.upper())
    result_df = pd.read_sql_query(query, conn)
    conn.close()
    return result_df

def to_sqlite(nodes_df: pd.DataFrame, tree: str) -> None:
    """
    Write dataframe to sqlite and ensure readability by
    MiniZinc
    """
    engine = create_engine('sqlite:///' + tree)
    strict_order = ['NodeID', 'ParentID', 'Alternative', 'NKids', 'Status', 'Label']
    column_order =          strict_order + \
                    sorted(list(set(nodes_df.reset_index().columns) - set(strict_order)))
    write_df = nodes_df.sort_values('NodeID').reset_index()\
                       .reindex(columns=column_order)
    write_df.to_sql('Nodes', engine, if_exists='replace', index=False)

def get_all_trees(benchmark_folder: str) -> list:
    """
    Return list of string paths to all sqlite files
    """
    trees = []
    for root, dirs, files in os.walk(benchmark_folder):
        trees.extend([os.path.join(root, name) for name in files if '.sqlite' in names])
    return trees

def parse_info_string(node_info: str) -> dict:
    """
    Parse node info string to dict of variables and their domains.
    Domains can be ranges, or other collectors that support a function len() for caculating their size.
    Domains must not be empty (otherwise the node will have been pruned)

    Returns:
        - {variable {str}: domain{range} }
    """
    x = node_info.replace("\n", "").replace("\t", "") # replace new lines and tabs
    x = x.strip('{"domains": ').strip('"}') # strip JSON notation
    x = x.split(';')[:-1]

    info_dict = {}
    for i in range(len(x)):
        x[i] = x[i].replace("var ", "")
        domain, varname = x[i].split(': ')
        
        if 'array_union' in domain:
            domain = domain.strip(' array_union([])').strip('])').split(',')
        else:
            domain = [domain.strip()]
        
        values = []
        for value_range in domain:
            if '..' in value_range:
                start, end = (int(num) for num in value_range.split('..'))
                values.append(range(start, end + 1))

        if ' = ' in varname:
            varname, value = varname.split(" = ")
            if str.isdigit(value):
                values.append(range(int(value), int(value) + 1))
            else:
                values.append([value])
        
        info_dict[varname] = set.union(*[set(inner) for inner in values])

        try:
            assert len(info_dict[varname]) >= 1
        except AssertionError as e:
            raise e
        except ValueError:
            raise ValueError("Cannot get size of domain of a variable with len!")

    return info_dict

def get_domain_size(node_info: str) -> int:
    """
    Get the size of all possible configurations possible for a node
    """
    
    info_dict = parse_info_string(node_info)
    size = 1
    for domain in info_dict.values():
        size *= len(domain)

    return size

def make_dfs_ordering(nodes_df: pd.DataFrame) -> list:
    """
    Return list of nodeids in the order they were completely solved by
    the depth-first search algorithm.
    We ignored restart nodes.
    """
    dfs_ordering = [0]
    boundary = nodes_df[(nodes_df['ParentID'] == 0) & (nodes_df['Status'] != 3)]\
                .sort_values('Alternative', ascending=False).index.to_list()

    # run simulated dfs on tree
    while len(boundary) > 0:
        nxt = boundary.pop()
        dfs_ordering.append(nxt)
        boundary.extend(nodes_df[(nodes_df['ParentID'] == nxt) & (nodes_df['Status'] != 3)]\
                             .sort_values('Alternative', ascending=False).index.to_list())

    assert set(dfs_ordering) == (set(nodes_df.index) - set(nodes_df[nodes_df['Status'] == 3].index))
    return dfs_ordering

def make_post_ordering(nodes_df: pd.DataFrame, node_id: int=0, post_ordering: list=None) -> list:
    """
    Return list of node ids in post order, i.e. left subtree - right subtree - root

    We ignored restart nodes.
    """
    if post_ordering is None:
        post_ordering = []

    # base case: current node is a leaf
    if nodes_df.loc[node_id, 'Status'] in {0, 1}:
        post_ordering.append(node_id)
        return post_ordering

    # else, travel through children
    children = nodes_df[(nodes_df['ParentID'] == node_id) & (nodes_df['Status'] != 3)]\
                .sort_values('Alternative', ascending=True).index
    
    # run recursive algorithm
    for child in children:
        make_post_ordering(nodes_df, child, post_ordering)
    post_ordering.append(node_id)

    if len(post_ordering) == len(nodes_df[nodes_df['Status'] != 3]):
        assert set(post_ordering) == (set(nodes_df[nodes_df['Status'] != 3].index))
    
    return post_ordering

def get_cum_weight(nodes_df: pd.DataFrame, weight_column: str, ordering: list) -> pd.Series:
    """
    Calculate cumulative sum according to an ordering using weight_column
    """
    if set(ordering) != (set(nodes_df.index) - set(nodes_df[nodes_df['Status'] == 3].index)):
        raise ValueError("Ordering is not a correct ordering of nodes. Some nodes are missing or redundant.")
    elif weight_column not in nodes_df.columns:
        raise ValueError("Weight column not in dataframe's columns.")

    cumulative = nodes_df.reindex(ordering)
    cumsum = cumulative[cumulative['Status'].isin({0, 1})][weight_column]\
                         .cumsum()\
                         .reindex(ordering)\
                         .reset_index(drop=True)\
                         .fillna(method='ffill').fillna(0)  

    return cumsum

def plot_goodness(cum_sums: 'dict[str, pd.Series]', ax_title: str='Weighting Performance', 
        figsize: tuple=(5, 5)) -> 'fig, ax':
    """
    Plot various cumulative sums of different weighting / progress measures.
    All progress measures must have:
        - a range between 0 and 1,
        - the same length
    """
    # check range and length
    prev_length = None
    if len(cum_sums) == 0:
        raise ValueError('Nothing to graph!')
    for name, cum_sum in cum_sums.items():
        if cum_sum.max() - 1 > EPSILON or abs(cum_sum.min()) > EPSILON: # use EPSILON cause precision
            import pdb; pdb.set_trace()
            raise ValueError(f'Invalid range for cumulative measure: {name}_scheme')
        elif prev_length and len(cum_sum) != prev_length:
            raise ValueError('Cumulative measures\' lengths do not match!')
        elif prev_length is None:
            prev_length = len(cum_sum)
    if prev_length is None:
        raise ValueError('Nothing to graph!')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=True)
    ax.set_title(ax_title)
    ax.plot(pd.Series(range(prev_length)) / (prev_length - 1), label='ground truth')
    
    for name, cum_sum in cum_sums.items():
        ax.plot(cum_sum, label=f'{name} scheme')
    ax.tick_params(axis='x', rotation=90)
    ax.legend()

    return fig, ax

def calculate_subtree_size(nodes_df: pd.DataFrame) -> None:
    """
    Get size of subtree rooted at each node in the tree. 
    Leaf node has a subtree size
    Pruned nodes are ignored (considered non-existent)

    :complexity: O(nodes_df.shape[0])
    """

    def get_subtree_size(node_id: int) -> int:
        # recursive function that returns node_id's subtree size

        # base case
        if nodes_df.loc[node_id, 'Status'] in {0, 1, 3}:
            return 1 if nodes_df.loc[node_id, 'Status'] in {0, 1} else 0

        count = sum([get_subtree_size(child_id) for child_id in nodes_df[nodes_df['ParentID'].isin({node_id})].index])
        count += 1
        nodes_df.loc[node_id, 'SubtreeSize'] = count
        return count

    nodes_df['SubtreeSize'] = (nodes_df['Status'] <= 1).astype(int)
    get_subtree_size(0)

def pairwise_diff(lst: list) -> bool:
    # helper for find_split_variable
    return len(set(lst)) == len(lst)

def is_unequal_split(nodes_df: int, children_idx: list) -> bool:    
    # helper for find_split_variable
    # return whether the split is binary (==, !=) or distribution of values
    return set(nodes_df.loc[children_idx, 'Label']\
                   .str.split(' ', expand=True)[1]\
                   .unique())\
        == set(['=', '!='])

def find_split_variable(par_idx: int, nodes_df: pd.DataFrame, 
            info_df: pd.DataFrame, mappings: dict={}) -> list:
    # 1. no goods domains are unreliable (one or more variable should have a null domain)
    # but the split variable's domain should still be reliable (?)
    # 2. some split labels are not appearing in the domains and vice versa due to the domains outputing{}
    # only certain names in flatzinc, and some variables are never split on but filled by 
    # propogation
    # 3. if sum of two children domains do not add up to parent, is it a split variable?
    # 4. is it true that mappings are unique, aka we can reuse it and not that a new introduced variables is used later?
    # 5. no good can have none label too, occurs when earlier sibling is a split node, and thus has already been fully explored. Apparently
    # the label isn't remembered after that much time
    # 6. the != sign of the unequal split sometimes has an assigned value, i.e. they don't necessarily add up to parent
    """
    Given a node, find the split variable among the domains (if possible) that leads to its children
    """
    
    children_idx = nodes_df[nodes_df['ParentID'] == par_idx].index
    if len(children_idx) <= 1: # can't find split_variable in this case, and does not need to
        return [], mappings, None, None
    
    label_var = nodes_df.loc[children_idx[0], 'Label'].split(' ')[0]
    par_domain = parse_info_string(info_df.loc[par_idx, 'Info'])
    children_domain = [
        parse_info_string(info_df.loc[child_id, 'Info']) for child_id in children_idx
    ]

    if label_var in mappings: # find in dictionary first
        if mappings[label_var] in par_domain: # sometimes csv compiled file do not match actual label
            return [mappings[label_var]], mappings, par_domain, children_domain

    split_vals = nodes_df.loc[children_idx, 'Label']\
                        .str.split('=', expand=True)[1]
    
    # if no good has no label,
    if split_vals.isna().sum() > 0:
        if len(children_idx) == 2: # we expect unequal split if two children
            assert len(split_vals) == 2
            assert split_vals.isna().values[1] == True # we also expect the nogood no label to be the second child 
            split_vals = [int(split_vals.values[0])] # account for nogoods with no labels
            uneq_split = True
        else: # we expect equal split if more children
            # we cannot expect to find a split variable
            return [], mappings, None, None
    else:
        split_vals = split_vals.astype(int).unique().flatten()
        uneq_split = is_unequal_split(nodes_df, children_idx)
    cands = []

    # children domain may include domains of no-goods which are unreliable
    # for now we ignore this thorny problem

    if not uneq_split:
        assert len(split_vals) == len(children_idx)
        for variable in par_domain:
            # each child should have a label = split_value
            rule_1 = all([children_domain[i][variable] == {split_vals[i]} for i in range(len(children_domain))])
            # all split domains add up to parent
            rule_2 = set.union(*[children_domain[i][variable] for i in range(len(children_domain))])\
                        == par_domain[variable]
            # split values should be different
            rule_3 = pairwise_diff(split_vals)

            if rule_1 and rule_2 and rule_3:
                cands.append(variable)      
    else:
        assert len(children_domain) == 2
        assert len(split_vals) == 1
        split_val = split_vals[0]
        for variable in par_domain:
            child_1, child_2 = children_domain
            # find child with equal sign and child with not equal sign
            child_eq, child_no = (child_1, child_2) if '!' not in nodes_df.loc[children_idx[0], 'Label'] \
                                        else (child_2, child_1)

            # case 1: 
            # child_eq has split_val and child_no not
            rule_1 = ({split_val} == child_eq[variable]) and (split_val not in child_no[variable])
            # child_1 + child_2 = par
            rule_2 = child_no[variable].union({split_val}) == par_domain[variable]

            if (rule_1 and rule_2):
                cands.append(variable)

    # if cand is already set in parent, remove
    cands = [name for name in cands if not len(par_domain[name]) == 1]
                    
    if len(cands) == 1:
        mappings[label_var] = cands[0]
        mappings[cands[0]] = label_var
    
    return cands, mappings, par_domain, children_domain

###################
#### UNIT TEST ####
###################

class TestHelperMethods(unittest.TestCase):

    def setUp(self):
        """
        Using this random tree:
                   0
                  / \   \
                 1   2   3  
               / /\  /\  /\
              4 5 6 7  8 9 10
                           /\ \
                         11 12 13

        Where status of the leaf nodes are: {4: 0, 5: 1, 6: 1, 7: 0, 8: 3, 9: 1, 11: 1, 12: 3, 13: 3}

        Inorder (Visit order): 0, 1, 4, 5, 6, 2, 7, 3, 9, 10, 11
        Postorder (Finish order): 4, 5, 6, 1, 7, 2, 9, 11, 10, 3, 0

        """

        self.test_df = test_df = pd.DataFrame({
            'NodeID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            'ParentID': [-1, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 10, 10, 10],
            'Alternative': [-1, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 2],
            'NKids': [3, 3, 2, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            'Status': [2, 2, 2, 2, 0, 1, 1, 0, 3, 1, 2, 1, 3, 3],
            'Label': ['' for _ in range(14)],
            'NodeWeight': [1, 1/3, 1/3, 1/3, 1/9, 1/9, 1/9, 1/3, 0, 1/6, 1/6, 1/6, 0, 0]

        })

    def test_todict(self):
        k = to_df('./benchmark_models/golomb/trees/04.sqlite', 'nodes')
        self.assertEqual(k.loc[0, 'NodeID'], 0)
        self.assertEqual(k.shape[0], 36)
        self.assertTrue(all([k.loc[idx, 'NodeID'] == idx for idx in range(k.shape[0])]))
        self.assertEqual(k.loc[26, 'Status'], 0)
        
        correct_order = ['NodeID', 'ParentID', 'Alternative', 'NKids', 'Status', 'Label']

        self.assertTrue(len(k.columns) >= len(correct_order))
        self.assertTrue(
            k.columns.to_list()[:len(correct_order)] == correct_order
        )

    def test_parseinfostring(self):
        test_infostring = \
        '''
        {
            "domains": "var 5..25: X_INTRODUCED_4_;
        var 13..79: VAR_2;
        var int: X_INTRODUCED_1_ = 0;
        var array_union([30..38, 50..57]): VAR_3;
        var 3..23: X_INTRODUCED_2_;
        "
        }
        '''
        info_dict = parse_info_string(test_infostring)
        self.assertEqual(len(info_dict), 5)
        self.assertEqual(set(info_dict.keys()), {'VAR_2', 'VAR_3', 'X_INTRODUCED_1_', 'X_INTRODUCED_2_', 'X_INTRODUCED_4_'})
        self.assertEqual(info_dict['VAR_2'], set(range(13, 80)))
        self.assertEqual(info_dict['VAR_3'], set(range(30, 39)).union(range(50, 58)))
        self.assertEqual(info_dict['X_INTRODUCED_1_'], set(range(0, 1)))
        self.assertEqual(info_dict['X_INTRODUCED_2_'], set(range(3, 24)))
        self.assertEqual(info_dict['X_INTRODUCED_4_'], set(range(5, 26)))
        self.assertEqual(sum([len(x) for x in info_dict.values()]), 1 + 9 + 8 + 67 + 21 + 21)
    
    def test_makedfsordering(self):
        """
        Use above random tree
        """

        ordering = make_dfs_ordering(self.test_df)
        self.assertEqual(ordering, [0, 1, 4, 5, 6, 2, 7, 3, 9, 10, 11])

    def test_getcumweight(self):
        """
        Use above random tree
        """
        
        ordering = make_dfs_ordering(self.test_df)
        cumsum = get_cum_weight(self.test_df, 'NodeWeight', ordering)

        self.assertTrue(all(np.abs(cumsum.values - pd.Series([0, 0, 1/9, 2/9, 1/3, 1/3, 2/3, 2/3, 5/6, 5/6, 1])) < EPSILON))

    def test_calculatesubtreesize(self):
        """
        Use above random tree
        """
        calculate_subtree_size(self.test_df)
        self.assertEqual(self.test_df['SubtreeSize'].to_list(), [11, 4, 2, 4, 1, 1, 1, 1, 0, 1, 2, 1, 0, 0])

    # def test_makepostordering(self):
    #     ordering = make_post_ordering(self.test_df)
    #     self.assertEqual(ordering, [4, 5, 6, 1, 7, 2, 9, 11, 10, 3, 0])
        
    def find_split_variable(self):
        # use the test tree: mario_easy_3

        tree = 'benchmark_models/mario/trees/mario_easy_3.sqlite'
        info_df = to_df(tree, 'info').set_index('NodeID')
        nodes_df = to_df(tree, 'nodes').set_index('NodeID')

        assert 'X_INTRODUCED_0_' == find_split_variable(0, nodes_df, info_df, {})[0][0]
        assert 'X_INTRODUCED_2_' == find_split_variable(1, nodes_df, info_df, {})[0][0]
        assert 'X_INTRODUCED_0_' == find_split_variable(2, nodes_df, info_df, {})[0][0]
        assert 'X_INTRODUCED_3_' == find_split_variable(3, nodes_df, info_df, {})[0][0]
        assert 'X_INTRODUCED_2_' == find_split_variable(25299, nodes_df, info_df, {})[0][0]
        assert 'X_INTRODUCED_11_' == find_split_variable(4, nodes_df, info_df, {})[0][0]
        assert 'X_INTRODUCED_5_' == find_split_variable(23031, nodes_df, info_df, {})[0][0]

if __name__ == '__main__':
    unittest.main()



    