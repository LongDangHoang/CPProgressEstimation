"""
Implements an offline tree-weight estimation or a constrained search tree

1. Loads tree information as a json object { {node_id, node_parent, etc. } } where node_id correlates
        with the order the node is visited
2. Assigns weight = 1 to root and propogates that weight down
3. Apply weighting scheme at every split
4. Measure weight as we go along node_id and compare it with actual number of nodes we eventually visit


"""