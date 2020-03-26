#!/bin/bash
# -cwd
# -l himem
# -l m_mem_free=64G

dataset_name="fMOST_sample"
image_stack="./${dataset_name}/image_stack"
work_space="./${dataset_name}/DM/"





## WPCA module
final_radius=4

# Remove duplicate edges in DM output.
python ./Code/Vector/remove_dup_edges.py $work_space

# WPCA
python ./Code/Vector/wpca_vector_generation_bnb.py $work_space $image_stack

# Vector Diffusion
python ./Code/Vector/vector_diffusion_3d.py $work_space $image_stack

# Compute Paths, may need to recompile the cpp code
./Code/Vector/3d_paths_src/compute_paths $work_space

# Generate final output.txt
python ./Code/Vector/3d_edge_vector_info.py $image_stack $work_space $final_radius                  


## Simplification module

rm "{$work_space}edge.txt"
mv "{$work_space}output.txt" "{$work_space}edge.txt"

# Generate shortest path forest, need to have multi_roots.txt file under $work_space.
python ./Code/Simplification/ShortestPathForest.py $work_space

# For STP data, use LeafBurner. The resolution is [10, 10, 50] along x, y, z axes.
python ./Code/Simplification/Simplifier.py $work_space ROOT -r [10,10,50]

# optional. BranchSelector.
# python ./Code/Simplification/BranchSelector.py $work_space -n 20


