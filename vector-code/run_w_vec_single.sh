#!/bin/bash
#$ -cwd
# -l himem
#$ -l m_mem_free=64G

dataset_name="diphat_STP_Jai"
image_stack="./${dataset_name}/image_stack"
work_space="./${dataset_name}/DM_2/"

# Triangulation.
intensity_threshold=0
gaussian=1
echo "Triangulation"
# python threshold_base_triangulation.py $image_stack $work_space $intensity_threshold $gaussian
# python threshold_triangulation2.py $image_stack $work_space $gaussian $intensity_threshold
# python base_triangulation.py $image_stack $work_space $gaussian

# DM
dm_input="${work_space}SC.bin"
persistence_threshold=1
dimension=3
echo "DM"
#./spt_cpp/spt_cpp $dm_input $work_space $persistence_threshold $dimension

# Gen No Dup
python remove_dup_edges.py $work_space

# WPCA
echo "WPCA"
python wpca_vector_generation_bnb.py $work_space $image_stack

# Vector Diffusion
echo "Vec Diffusion"
python vector_diffusion_3d.py $work_space $image_stack

# Compute Paths
echo "Compute Paths"
./3d_paths_src/compute_paths $work_space

# Generate final output.txt
echo "Gen final output.txt"
final_radius=4
python 3d_edge_vector_info.py $image_stack $work_space $final_radius                  