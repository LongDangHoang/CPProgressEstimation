# CPProgressEstimation

AIM: predict the size of a Depth-First search tree commonly used in Constrained Programming.
These scripts help compute various tree weighting schemes and relevant information given a search profile in `.sqlite` format

## Commands

First, a search profile must be produced. This can be done from MiniZincIDE.

To run the computation script on a given setting, use the command:

```bash
python3 main.py -c {path/to/setting}.yaml
```

If the configuration option is ignored, the file `settings.yaml` in the same directory is used by default.
Once completed, the script will:
1. Save all computed tree weights into the original `.sqlite` file. Recomputation will be ignored unless forced in the settings.
2. Save a graph visualising the performance of different schemes into the specified `image_folder` in the settings.

## Settings

These are general settings. For settings specific to each tree weighting scheme, see the below section.

|Setting |Description |
|:---: | :--- |
|**`tree`**|Path (relative or absolute) to the problem instance `.sqlite` file|
|**`schemes`**|List of allowable schemes. Must match an option in `tree_weight.make_weight_scheme`|
|**`schemes_settings`**|Dictionary of settings for specfic weighting schemes. Keys of the dictionary must match entry in `schemes`. Empty dictionariy is allowed, and scheme computed without an entry in this setting is computed with default values if applicable|
|**`forced_recompute`**|List of schemes to be recomputed, ignoring existing (if any) result. Must match an option in `tree_weight.make_weight_scheme`|
|**`exponential_smoothing`**|Dictionary of settings for double exponential smoothing computation of cumulative weight values. Empty dictionariy is allowed, and scheme computed without an entry in this setting is computed with default values if applicable.<ul><li>`a`: decay factor for fitted weight value;</li><li>`b`: decay factor for fitted slope value</li></ul>|
|**`write_to_sqlite`**|Set to `True` to saves all relevant computed values into the original `.sqlite` profile. For example, depth-first-search ordering, subtree sizes, node weight for different schemes. Recommended.|
|**`save_image`**|Set to `True` to save a graph visualsing estimation performance, `False` otherwise| 
|**`image_folder`**|Path (relative or absolute) to the folder where output images are saved. Only relevant if `save_image` is true|
|**`image_settings`**|Settings for outputted graph. Only relevant if `save_image` is true.<ul><li>`size`: figure size in `plt.figure`, null for default value</li><li>`title`: graph title, null for default value</li></ul>|
|**`use_parallel`**|Set to `True` to use parallelization optimisation for a *single* problem instance, not parallelization strategy for multiple instances at the same time|

## Schemes

### True Scheme
A scheme to test the maximum accuracy using tree weights approach.

### Uniform Scheme
The most standard scheme, which assigns equal weight to every child node. Best used as a benchmark for other schemes.

### SubtreeSize Scheme
Learn a characteristic weight that describes the distribution of children nodes' subtree sizes. This scheme uses the assumption that the first child usually have a larger subtree to search, while the second child benefits from the work done on the first child and thus has a smaller subtree. Then, the scheme assumes there exist a number $w < 1$ such that the second child's subtree size is $w$ times the first child's subtree size, and the third child's subtree is $w$ times the second's, and so on.

The scheme thus uses an assumed $w$ on a downward pass. Once a subtree has been fully explored and the scheme moves back on the split parent node, it updates $w$ using an exponential moving average calculated from the explored subtree. $w$ is fitted at every split-level to be the mean of the ratios between subsequent children. 

Parameters:
- `decay_factor`: for calculating the exponential moving average of $w$.

### Search Space Scheme

Assign weight based on the splitting variable, under the assumption that under an unequal split, the node with the larger domain tend to have a larger subtree size. I.e., given a split $x = 3$ and $x > 3$, there should be more nodes on the side of $x > 3$.

To substantially speed up this scheme, the scripts can rely on a `.paths` file relating the variable names in the profile's domains and the `.fzn` file. To compute this file, use the following command:

```bash
minizinc -c {.mzn} {.dzn} --output-paths-to-file {paths file name}.paths --no-output-ozn --output-fzn-to-stdout 1>/dev/null
```

The `.paths` file must be in the same directory as the `.sqlite` profile.
To output all such files for all problem instances, execute the `make_paths.py` file in the following file structure with the appropriate path to the `problem_folder`:
```
.
+-- problem_folder
|   +-- problem_class_1
|   |   +-- problem.mzn
|   |   +-- instance1.dzn
|   |   +-- instance2.dzn
|   +-- trees
|   |   +-- instance1.sqlite
|   |   +-- instance2.sqlite
+-- make_paths.py
```

MiniZinc: https://github.com/MiniZinc
