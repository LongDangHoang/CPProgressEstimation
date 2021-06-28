import pandas as pd
import traceback
import yaml
import os

from helper import to_df, get_existing_dfs_ordering, make_dfs_ordering, get_exp_smoothed_cum_weight, plot_goodness, get_cum_weight, to_sqlite
from tree_weight import assign_weight, make_weight_scheme
from pathlib import Path
from argparse import ArgumentParser

from typing import List


def make_graph_from_tree(tree: str,
        schemes: List[str],
        exponential_smoothing: dict={},
        scheme_settings: dict={},
        write_to_sqlite: bool=True, 
        save_image: bool=True,
        image_folder: str='graphs/',
        forced_recompute: list=None,
        use_parallel: bool=False,
        image_settings: dict={}
    ) -> '(fig, ax), pd.DataFrame, pd.DataFrame, dict[str, pd.Series]':
    """
    Make a graph comparing all weighting schemes for a given tree

    Returns:
        - the figure and axs
        - the dataframe after weights assignment (if any)
        - the cumulative weights used
    """
    nodes_df = to_df(tree, 'nodes').set_index('NodeID')
    info_df = to_df(tree, 'info').set_index('NodeID')
    print("Loading file complete.")

    cum_sums = {}

    # make dfs ordering
    if 'DFSOrdering' in nodes_df.columns:
        dfs_ordering = get_existing_dfs_ordering(nodes_df)
    else:
        dfs_ordering = make_dfs_ordering(nodes_df)
        # save to file
        if write_to_sqlite:
            to_sqlite(nodes_df, tree)
    print("DFS Ordering computation complete.")

    for scheme in schemes:
        print(f"Starting on scheme: {scheme}")
        scheme_name = scheme.split('_')[0]
        weight_col = scheme_name[0].upper() + scheme_name[1:] + 'NodeWeight'
        orig_cols = nodes_df.columns # keep track of original columns in case scheme fails

        # dictionary to hold all parameters
        state_dict= {
            'nodes_df': nodes_df,
            'info_df': info_df,
            'weight_colname': weight_col,
            'tree': tree,
            'use_parallel': use_parallel,
            'weight_scheme': scheme,
            'write_to_sqlite': write_to_sqlite
        }

        if scheme in scheme_settings:
            state_dict_new = scheme_settings[scheme]
            state_dict_new.update(state_dict)
            state_dict = state_dict_new

        try:
            if (weight_col not in nodes_df.columns) or (forced_recompute is not None and scheme in forced_recompute):
                assign_weight(state_dict)
                print("Weight assignment complete.")
                if write_to_sqlite:
                    to_sqlite(nodes_df, tree)

            cum_sums[scheme_name] = get_cum_weight(nodes_df, weight_col, dfs_ordering)
            print("Cumulative sum complete")
            if scheme in exponential_smoothing: # FIXME: scheme and scheme name are different and hella confusing
                if exponential_smoothing[scheme] is not None:
                    cum_sums[scheme_name + '_exponential_smoothed'] = get_exp_smoothed_cum_weight(nodes_df, cum_sums[scheme_name])
                    print("Exponential smoothing complete")
            print(f"Complete scheme {scheme}")
        except Exception as e:
            # we want to move on as only one scheme may fail
            print(f"{scheme} computation fails! Moving on to next scheme...")
            traceback.print_exc()
            # reset all variables related to that scheme
            nodes_df = nodes_df[orig_cols]
            if scheme_name in cum_sums:
                del cum_sums[scheme_name]
  
    ax_title = tree.split('/')[1] + '_' + tree.split('/')[-1].strip('.sqlite') if image_settings['title'] is None else image_settings['title']
    figsize = (15, 15) if image_settings['size'] is None else tuple(image_settings['size'])
    fig, ax = plot_goodness(cum_sums, title=ax_title, size=figsize)
    fig.savefig(image_folder + ax_title)
    fig.show()

    return (fig, ax), nodes_df, info_df, cum_sums

if __name__ == '__main__':

    parser = ArgumentParser(description="Configure weight estimation")
    parser.add_argument('-c', '--config', type=Path, default='./settings.yaml', nargs='?')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
 
    (fig, ax), _, _, _  = make_graph_from_tree(
        **config
    )

    fig.show()