"""
helper file
"""

import sqlite3
import pandas as pd
import unittest


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


###################
#### UNIT TEST ####
###################

class TestHelperMethods(unittest.TestCase):

    def test_todict(self):
        k = to_df('./trees/golomb_4.sqlite', 'nodes')
        self.assertEqual(k.loc[0, 'NodeID'], 0)
        self.assertEqual(k.shape[0], 36)
        self.assertTrue(all([k.loc[idx, 'NodeID'] == idx for idx in range(k.shape[0])]))
        self.assertEqual(k.loc[26, 'Status'], 0)
        self.assertEqual(
            {'NodeID', 'Status', 'NKids', 'ParentID', 'Alternative', 'Label'},
            set(k.columns)
        )


if __name__ == '__main__':
    unittest.main()



    