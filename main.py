import pandas as pd
import traceback
import os

from helper import to_df, make_dfs_ordering, plot_goodness, get_cum_weight, to_sqlite
from tree_weight import assign_weight, make_weight_scheme
from pathlib import Path

from typing import List


def make_graph_from_tree(tree: str,
        schemes: List[str], 
        write_to_sqlite: bool=True, 
        save_image: bool=True,
        image_folder: str='graphs/',
        forced_recompute: list=None,
        assign_in_dfs_order: bool=False,
        use_parallel: bool=False
    ) -> '(fig, ax), pd.DataFrame, pd.DataFrame dict[str, pd.Series]':
    """
    Make a graph comparing all weighting schemes for a given tree

    Returns:
        - the figure and axs
        - the dataframe after weights assignment (if any)
        - the cumulative weights used
    """
    nodes_df = to_df(tree, 'nodes').set_index('NodeID')
    info_df = to_df(tree, 'info').set_index('NodeID')

    cum_sums = {}

    # make dfs ordering
    if 'DFSOrdering' in nodes_df.columns:
        dfs_ordering = nodes_df[~nodes_df['Status'].isin({3})].sort_values('DFSOrdering').index.to_list()
    else:
        dfs_ordering = make_dfs_ordering(nodes_df)
        # save dfs_ordering to dataframe
        nodes_df['DFSOrdering'] = -1
        nodes_df.loc[dfs_ordering, 'DFSOrdering'] = range(len(dfs_ordering))
        nodes_df.loc[:, 'DFSOrdering'] = nodes_df['DFSOrdering'].astype(int)

    for scheme in schemes:
        scheme_name = scheme.split('_')[0]
        weight_col = scheme_name[0].upper() + scheme_name[1:] + 'NodeWeight'
        orig_cols = nodes_df.columns # keep track of original columns in case scheme fails

        # dictionary to hold all parameters
        state_dict= {
            'nodes_df': nodes_df,
            'info_df': info_df,
            'weight_colname': weight_col,
            'tree': tree,
            'assign_in_dfs_order': assign_in_dfs_order,
            'use_parallel': use_parallel,
            'weight_scheme': scheme
        }

        try:
            if (weight_col not in nodes_df.columns) or (forced_recompute is not None and scheme in forced_recompute):
                assign_weight(state_dict)
            cum_sums[scheme_name] = get_cum_weight(nodes_df, weight_col, dfs_ordering)
        except Exception as e:
            # we want to move on as only one scheme may fail
            print(f"{scheme} computation fails! Moving on to next scheme...")
            traceback.print_exc()
            # reset all variables related to that schemenodes_df
            nodes_df = nodes_df[orig_cols]
            if scheme_name in cum_sums:
                del cum_sums['scheme_name']
  
    ax_title = tree.split('/')[1] + '_' + tree.split('/')[-1].strip('.sqlite')
    fig, ax = plot_goodness(cum_sums, ax_title=ax_title)
    fig.savefig(image_folder + ax_title)

    if write_to_sqlite:
        # write to sqlite file so we do not waste time recomputing
        to_sqlite(nodes_df, tree)
    
    fig.show()

    return (fig, ax), nodes_df, info_df, cum_sums

if __name__ == '__main__':
    
    image_folder = 'graphs/'
    tree = 'benchmark_models/mario/trees/mario_easy_3.sqlite'
    (fig, ax), _, _, _  = make_graph_from_tree(
        tree, 
        schemes=[
            'uniform_scheme',
            'true_scheme',
            'searchSpace_scheme',
            'subtreeSize_scheme'
        ],
        forced_recompute=[
            # 'subtreeSize_scheme',
            'uniform_scheme',
            # 'domain_scheme',
            # 'searchSpace_scheme'x:w

        ],
        write_to_sqlite=False,
        image_folder=image_folder,
        save_image=True,
        use_parallel=True,
        assign_in_dfs_order=True
    )

    fig.show()
