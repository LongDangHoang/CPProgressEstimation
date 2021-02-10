"""
helper file
"""

import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt

import unittest

EPSILON = 10e-7

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
    return pd.read_sql_query(query, conn)

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
        x[i] = x[i].strip("var ")
        domain, varname = x[i].split(': ')
        if '..' in domain:
            start, end = (int(num) for num in domain.split('..'))
            info_dict[varname] = range(start, end + 1)
        else:
            varname, value = varname.split(" = ")
            if str.isdigit(value):
                info_dict[varname] = range(int(value), int(value) + 1)
            else:
                info_dict[varname] = value
            
            try:
                assert len(info_dict[varname]) > 1
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
    for domain_range in info_dict.values():
        size *= len(domain_range)

    return size

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

###################
#### UNIT TEST ####
###################

class TestHelperMethods(unittest.TestCase):

    def test_todict(self):
        k = to_df('./trees/golomb_4_1.sqlite', 'nodes')
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
        var 30..38: VAR_3;
        var 3..23: X_INTRODUCED_2_;
        "
        }
        '''
        info_dict = parse_info_string(test_infostring)
        self.assertEqual(len(info_dict), 5)
        self.assertEqual(set(info_dict.items()), 
                        set([('X_INTRODUCED_1_', range(0, 1)), 
                             ('VAR_2', range(13, 80)),
                             ('VAR_3', range(30, 39)),
                             ('X_INTRODUCED_2_', range(3, 24)),
                             ('X_INTRODUCED_4_', range(5, 26))]))
    
    def test_makedfsordering(self):
        pass

if __name__ == '__main__':
    unittest.main()



    