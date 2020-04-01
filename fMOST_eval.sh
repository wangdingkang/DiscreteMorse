#!/bin/bash
# -cwd
# -l himem
# -l m_mem_free=64G

input_swc=''
discretized_swc=''
ground_truth_swc=''


# Discretize the result, so that neighboring nodes are within 2 unit length.
python ./Code/fMOST_Evaluation/Discretize.py $input_swc $discretized_swc

# Evaluation. Compute precision/recall/F1-score.
python ./Code/fMOST_Evaluation/Evaluate.py $discretized_swc $ground_truth_swc




