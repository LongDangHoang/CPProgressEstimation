"""
helper file
"""

import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import unittest

EPSILON = 10e-7

class DomainRange:

    def __init__(self, values: list):
        
        assert isinstance(values, list)
        assert len(values) > 0

        self.values = values # store pass in list of values

    def __len__(self):
        
        length = 0
        for value in self.values:
            try:
                length += len(value)
            except ValueError:
                length += 1 # assume value is a single value
        
        return length

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
                values.append(value)
        
        info_dict[varname] = DomainRange(values)

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

def get_cum_weights(nodes_df: pd.DataFrame, ordering: list) -> pd.Series:
    """
    Calculate cumulative sum according to an ordering
    """
    if set(ordering) != (set(nodes_df.index) - set(nodes_df[nodes_df['Status'] == 3].index)):
        raise ValueError("Ordering is not a correct ordering of nodes. Some nodes are missing or redundant")

    cumulative = nodes_df.reindex(ordering)
    cumsum = cumulative[cumulative['Status'].isin({0, 1})]['NodeWeight']\
                         .cumsum()\
                         .reindex(ordering)\
                         .reset_index(drop=True)\
                         .fillna(method='ffill').fillna(0)  

    return cumsum

###################
#### UNIT TEST ####
###################

class TestHelperMethods(unittest.TestCase):

    def test_todict(self):
        k = to_df('./benchmark_models/golomb/trees/04.sqlite', 'nodes')
        self.assertEqual(k.loc[0, 'NodeID'], 0)
        self.assertEqual(k.shape[0], 36)
        self.assertTrue(all([k.loc[idx, 'NodeID'] == idx for idx in range(k.shape[0])]))
        self.assertEqual(k.loc[26, 'Status'], 0)
        self.assertEqual(
            {'NodeID', 'Status', 'NKids', 'ParentID', 'Alternative', 'Label'},
            set(k.columns)
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
        self.assertEqual(set(info_dict['VAR_2'].values), set([range(13, 80)]))
        self.assertEqual(set(info_dict['VAR_3'].values), set([range(30, 39), range(50, 58)]))
        self.assertEqual(set(info_dict['X_INTRODUCED_1_'].values), set([range(0, 1)]))
        self.assertEqual(set(info_dict['X_INTRODUCED_2_'].values), set([range(3, 24)]))
        self.assertEqual(set(info_dict['X_INTRODUCED_4_'].values), set([range(5, 26)]))
        self.assertEqual(sum([len(x) for x in info_dict.values()]), 1 + 9 + 8 + 67 + 21 + 21)
    
    def test_makedfsordering(self):
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
        """

        test_df = pd.DataFrame({
            'NodeID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            'ParentID': [-1, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 10, 10, 10],
            'Alternative': [-1, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 2],
            'NKids': [3, 3, 2, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            'Status': [2, 2, 2, 2, 0, 1, 1, 0, 3, 1, 2, 1, 3, 3]
        }) 

        ordering = make_dfs_ordering(test_df)
        self.assertEqual(ordering, [0, 1, 4, 5, 6, 2, 7, 3, 9, 10, 11])

    def test_getcumweight(self):
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
        """
        
        test_df = pd.DataFrame({
            'NodeID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            'ParentID': [-1, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 10, 10, 10],
            'Alternative': [-1, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 2],
            'NKids': [3, 3, 2, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            'Status': [2, 2, 2, 2, 0, 1, 1, 0, 3, 1, 2, 1, 3, 3],
            'NodeWeight': [1, 1/3, 1/3, 1/3, 1/9, 1/9, 1/9, 1/3, 0, 1/6, 1/6, 1/6, 0, 0]
        }) 

        ordering = make_dfs_ordering(test_df)
        cumsum = get_cum_weights(test_df, ordering)

        self.assertTrue(all(np.abs(cumsum.values - pd.Series([0, 0, 1/9, 2/9, 1/3, 1/3, 2/3, 2/3, 5/6, 5/6, 1])) < EPSILON))

if __name__ == '__main__':
    unittest.main()



    