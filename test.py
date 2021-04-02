"""
Implement more rigorous testing for key functions
"""

import pandas as pd
import numpy as np
import unittest
import random

from argparse import ArgumentParser
from collections import deque

from helper import calculate_subtree_size, get_cum_weight, make_dfs_ordering
from tree_weight import assign_weight

##########################nodes_df.shape[0]
# Test utility functions #
##########################

def make_binary_tree(height: int) -> pd.DataFrame:
    """
    Return a dataframe of node_id, parent_id that completes a binary tree

    The tree has the columns NodeID, ParentID, and Status
    """
    assert height >= 10
    index = pd.Series(range(1, 2**height))
    parent = index // 2
    status = (index < 2**(height - 1)) + 1
    nkid = (status == 2) * 2
    alternative = (index % 2 == 1).astype(int)
    alternative.iloc[0] = -1

    # select random subset to be 0, i.e solutions
    status.loc[status[status == 1].sample(random.randint(1, 50)).index] = 0

    return pd.DataFrame({
        'NodeID': index - 1, 
        'ParentID': parent - 1,
        'Status': status,
        'NKids': nkid,
        'Alternative': alternative
    })

def destroy_random_branch(nodes_df: pd.DataFrame, height: int) -> None:
    
    if 'MathSubtreeSize' not in nodes_df:
        depth = np.floor(np.log2(nodes_df.index.to_numpy() + 1))
        math_subtree_size = np.power(2, height - depth) - 1
        nodes_df.loc[:, 'MathSubtreeSize'] = math_subtree_size
    
    # randomly prune a branch of a tree and updates the subtree sizes
    
    # select random node with children
    rand_node = int(nodes_df[(nodes_df['Status'] == 2) & (nodes_df.index != 0)].sample().index[0]) + 1
    original_size = int(nodes_df.loc[0, 'MathSubtreeSize'])

    # eradicate rand_node's subtree
    to_drop = []
    cands = deque([rand_node]) # cands count from 1, i.e node_id = 0 is cand 1
    while len(cands) > 0:
        par = cands.popleft()
        if par * 2 - 1 in nodes_df.index:
            cands.append(par * 2)
        if par * 2 in nodes_df.index:
            cands.append(par * 2 + 1)
        
        if par != rand_node:
            to_drop.append(par - 1)

    loss = nodes_df.loc[rand_node - 1, 'MathSubtreeSize'] - 1
    nodes_df.loc[rand_node - 1, 'Status'] = random.randint(0, 1)
    nodes_df.loc[rand_node - 1, 'MathSubtreeSize'] = 1
    nodes_df.loc[rand_node - 1, 'NKids'] = 0

    # reduce math subtree size for parent
    parent = rand_node // 2 # parent counts from 1, i.e. node 0 is parent 1, parent 0 is node -1
    while parent != 0:
        nodes_df.loc[parent - 1, 'MathSubtreeSize'] -= loss
        parent = parent // 2

    assert nodes_df['MathSubtreeSize'].isna().sum() == 0
    assert nodes_df.loc[0, 'MathSubtreeSize'] == original_size - loss
    nodes_df.drop(index=to_drop, inplace=True)

class StressTest(unittest.TestCase):

    @classmethod
    def setUpClass(StressTest):
        # make full random binary tree
        StressTest.rand_h = random.randint(13, 15)
        StressTest.rand_bin_tree = make_binary_tree(StressTest.rand_h).set_index('NodeID')

        # calculate mathematically correct subtree size 
        depth = np.floor(np.log2(StressTest.rand_bin_tree.index.to_numpy() + 1))
        math_subtree_size = np.power(2, StressTest.rand_h - depth) - 1
        StressTest.rand_bin_tree['MathSubtreeSize'] = math_subtree_size
        if StressTest.rand_bin_tree['MathSubtreeSize'].isna().sum() != 0:
            raise RuntimeError("Math tree size includes NaN")

        # copy tree for random pruning
        StressTest.rand_tree = pd.DataFrame.copy(StressTest.rand_bin_tree)

        # randomly destroy branches
        for _ in range(1000):
            destroy_random_branch(StressTest.rand_tree, StressTest.rand_h)
            if StressTest.rand_tree.shape[0] <= 2 ** (StressTest.rand_h - 3) - 1: # avoid reducing tree too much
                break

        # fix rand_tree index to be continous for parallel operations
        temp = pd.DataFrame.copy(StressTest.rand_tree.reset_index().reset_index())
        node_newid = temp[['index', 'NodeID']]
        temp.loc[(temp['ParentID'] != -1), 'ParentID'] = node_newid.set_index('NodeID').loc[temp['ParentID'].iloc[1:], 'index'].values
        temp = temp.drop(columns=['NodeID']).rename(columns={'index': 'NodeID'})
        StressTest.rand_tree = temp.set_index('NodeID')

        StressTest.use_parallel = True # to set to true, rand_tree must have continous index 
        print(f"Testing parallelization methods: {StressTest.use_parallel}")

    def test_calculatesubtreesize_bintree(self):
        calculate_subtree_size(StressTest.rand_bin_tree)
        self.assertTrue(np.all(StressTest.rand_bin_tree['MathSubtreeSize'] == StressTest.rand_bin_tree['SubtreeSize']))

    def test_calculatesubtreesize_randomtree(self):
        calculate_subtree_size(StressTest.rand_tree)
        self.assertTrue(np.all(StressTest.rand_tree['SubtreeSize'] == StressTest.rand_tree['MathSubtreeSize']))

    def test_assignweight_bintree(self):
        """
        Test weight assignment through true weight scheme (as we know true subtree size)
        """

        nodes_df = pd.DataFrame.copy(StressTest.rand_bin_tree)
        nodes_df = nodes_df.rename(columns={'MathSubtreeSize': 'SubtreeSize'})

        assign_weight(
            nodes_df=nodes_df,
            weight_scheme='true_scheme',
            assign_in_dfs_order=False,
            use_parallel=StressTest.use_parallel
        )

        # calculate math true weight as 2**(1-depth)
        nodes_df['TrueMathWeight'] = np.power(2, -np.floor(np.log2(nodes_df.index.to_numpy() + 1)))
        self.assertTrue(np.all(nodes_df['TrueMathWeight'] == nodes_df['NodeWeight']))

    def test_assignweight_zero(self):

        nodes_df = pd.DataFrame.copy(StressTest.rand_tree)
        nodes_df = nodes_df.rename(columns={'MathSubtreeSize': 'SubtreeSize'})

        assign_weight(
            nodes_df=nodes_df,
            weight_scheme='zero_scheme',
            assign_in_dfs_order=False,
            use_parallel=StressTest.use_parallel
        )
        self.assertTrue(np.all(nodes_df.iloc[1:, :]['NodeWeight'] == 0))
    
    def test_assignweight_one(self):

        nodes_df = pd.DataFrame.copy(StressTest.rand_tree)
        nodes_df = nodes_df.rename(columns={'MathSubtreeSize': 'SubtreeSize'})

        assign_weight(
            nodes_df=nodes_df,
            weight_scheme='one_scheme',
            assign_in_dfs_order=False,
            use_parallel=StressTest.use_parallel
        )

        self.assertTrue(np.all(nodes_df['NodeWeight'] == 1))
    
    def test_assignweight_sumzero(self):

        nodes_df = pd.DataFrame.copy(StressTest.rand_tree)
        nodes_df = nodes_df.rename(columns={'MathSubtreeSize': 'SubtreeSize'})

        assign_weight(
            nodes_df=nodes_df,
            weight_scheme='sumK_scheme',
            assign_in_dfs_order=False,
            k=0,
            use_parallel=StressTest.use_parallel
        )

        # check sum of all leaves = 0
        # this assumption is false when tree is not perfect
        # self.assertAlmostEqual(nodes_df.loc[nodes_df['Status'].isin({0, 1}), 'NodeWeight'].sum(), 0)

        # check sum of all siblings = 0
        self.assertTrue(np.all(np.abs(nodes_df.iloc[1:,:].groupby(['ParentID']).sum()['NodeWeight']) < 1e-8))

    def test_assignweight_sumrandom(self):

        nodes_df = pd.DataFrame.copy(StressTest.rand_tree)
        nodes_df = nodes_df.rename(columns={'MathSubtreeSize': 'SubtreeSize'})

        k = 1 / random.randint(1, 100)

        assign_weight(
            nodes_df=nodes_df,
            weight_scheme='sumK_scheme',
            assign_in_dfs_order=False,
            k=k,
            use_parallel=StressTest.use_parallel
        )

        # check sum of all siblings = k * parent 
        sum_sibs = nodes_df.iloc[1:,:].groupby(['ParentID']).sum()['NodeWeight']
        self.assertTrue(np.all(np.abs(sum_sibs - k * nodes_df.loc[sum_sibs.index, 'NodeWeight']) < 1e-8))

    def test_assignweight_randomtree(self):
        """
        For a random tree, cumulative true scheme should always return the straightline through the origin
        """

        nodes_df = pd.DataFrame.copy(StressTest.rand_bin_tree)
        nodes_df = nodes_df.rename(columns={'MathSubtreeSize': 'SubtreeSize'})

        assign_weight(
            nodes_df=nodes_df,
            weight_scheme='test_scheme',
            assign_in_dfs_order=False,
            use_parallel=StressTest.use_parallel
        )

        self.assertTrue(np.all(np.abs(nodes_df['NodeWeight'] - nodes_df['RandomTrueNodeWeight']) < 1e-8))

if __name__ == '__main__':
    unittest.main()