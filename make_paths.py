import os

from pathlib import Path

benchmark_folder = Path('/home/longdang/WorkStation/University/cpTreeEstimation/benchmark_models/')

p_classes = [name for name in benchmark_folder.iterdir() if name.is_dir()]
for p_class in p_classes:
    # print(p_class.name)
    mzn_file = [f for f in p_class.iterdir() if f.name.find('.mzn') > 0][0]
    dzn_files = [f for f in p_class.iterdir() if f.name.find('.dzn') > 0]
    tree_dir = [d for d in p_class.iterdir() if d.name == 'trees' and d.is_dir()][0]
    make_path_command = 'minizinc -c {} {} --output-paths-to-file {}/{}.paths --no-output-ozn --output-fzn-to-stdout 1>/dev/null' 

    for dzn_f in dzn_files:
        c = make_path_command.format(
            str(mzn_file.resolve()),
            str(dzn_f.resolve()),
            str(tree_dir.resolve()),
            str(dzn_f.name).replace('.dzn', '')
        )
        os.system(c)