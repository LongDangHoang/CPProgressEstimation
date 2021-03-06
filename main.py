import pandas as pd
import os

from sqlalchemy import create_engine
from typing import List, Callable
from helper import to_df, make_dfs_ordering, plot_goodness, get_cum_weight
from tree_weight import assign_weight

def make_graph_from_tree(image_folder: str, tree: str, schemes: List[Callable], 
        write_to_sqlite: bool=True) -> '(fig, ax), pd.DataFrame, pd.DataFrame dict[str, pd.Series]':
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
        scheme_name = scheme.__name__.split('_')[0]
        weight_col = scheme_name[0].upper() + scheme_name[1:] + 'NodeWeight'
        
        if weight_col not in nodes_df.columns:
            assign_weight(nodes_df, scheme, weight_colname=weight_col, info_df=info_df)
        cum_sums[scheme_name] = get_cum_weight(nodes_df, weight_col, dfs_ordering)

  
    ax_title = tree.split('/')[1] + '_' + tree.split('/')[-1].strip('.sqlite')
    fig, ax = plot_goodness(cum_sums, ax_title=ax_title)
    fig.savefig(image_folder + ax_title)

    if write_to_sqlite:
        # write to sqlite file so we do not waste time recomputing
        engine = create_engine('sqlite:///' + tree)
        strict_order = ['NodeID', 'ParentID', 'Alternative', 'NKids', 'Status', 'Label']
        column_order =          strict_order + \
                       sorted(list(set(nodes_df.reset_index().columns) - set(strict_order)))
        write_df = nodes_df.reset_index().reindex(columns=column_order)
        write_df.to_sql('Nodes', engine, if_exists='replace', index=False)
    
    fig.show()

    return (fig, ax), nodes_df, info_df, cum_sums

# if __name__ == '__main__':
    
#     image_folder = 'graphs/'
#     make_graph_from_tree(image_folder)