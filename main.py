import pandas as pd
import os

from typing import List
from helper import to_df, make_dfs_ordering, plot_goodness, get_cum_weight, to_sqlite
from tree_weight import assign_weight 

def make_graph_from_tree(image_folder: str, tree: str,
        schemes: List[str], 
        write_to_sqlite: bool=True, 
        save_image: bool=True, 
        forced_recompute: list=None,
        assign_in_dfs_order: bool=False,
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

    if 'DFSOrdering' in nodes_df.columns:
        dfs_ordering = nodes_df[~nodes_df['Status'].isin({3})].sort_values('DFSOrdering').index.to_list()
    else:
        dfs_ordering = make_dfs_ordering(nodes_df)
        nodes_df['DFSOrdering'] = -1
        nodes_df.loc[dfs_ordering, 'DFSOrdering'] = range(len(dfs_ordering))
        nodes_df.loc[:, 'DFSOrdering'] = nodes_df['DFSOrdering'].astype(int)

    for scheme in schemes:
        scheme_name = scheme.split('_')[0]
        weight_col = scheme_name[0].upper() + scheme_name[1:] + 'NodeWeight'
        
        if (weight_col not in nodes_df.columns) or (forced_recompute is not None and scheme in forced_recompute):
            assign_weight(nodes_df, scheme, 
                weight_colname=weight_col, info_df=info_df, 
                tree=tree, assign_in_dfs_order=assign_in_dfs_order)
        cum_sums[scheme_name] = get_cum_weight(nodes_df, weight_col, dfs_ordering)

  
    ax_title = tree.split('/')[1] + '_' + tree.split('/')[-1].strip('.sqlite')
    fig, ax = plot_goodness(cum_sums, ax_title=ax_title)
    fig.savefig(image_folder + ax_title)

    if write_to_sqlite:
        # write to sqlite file so we do not waste time recomputing
        to_sqlite(nodes_df, tree)
    
    fig.show()

    return (fig, ax), nodes_df, info_df, cum_sums

# if __name__ == '__main__':
    
#     image_folder = 'graphs/'
#     make_graph_from_tree(image_folder)