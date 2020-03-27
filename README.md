# DiscreteMorse
Discrete Morse-based pipeline for neuron tracing on tracer injection and single-neuron data sets. It will automatically generate a forest (a set of trees) summarization of the given neuron imaging data. The biorxiv version is on https://www.biorxiv.org/content/10.1101/2020.03.21.000323v1. 

## Step-by-step workflow

Our pipeline is divided in three modules, which is a little different from the paper. In the paper, we have so-called preprocessing module, here we don't. Because the sample inputs/data are already preprocessed and packed in vtk/image_stack format. Here the first module is DIPHA which generates the graph construction. The second one is vector-score module calculating the vector score. The last one is the simplification module.

### DIPHA. (Placeholder for Lucas)




### Vector-score module. (Placeholders for Lucas, there are some scripts need to be explained).
The module is used for calculating the vector scores. We have five scripts in total and will be introduced in order.

1). Code/Vector/remove_dup_edges.py This script will remove duplicate edges of the output from Dipha.

2). Code/Vector/wpca_vector_generation_bnb.py

3). Code/Vector/vector_diffusion_3d.py

4). Code/Vector/3d_paths_src/compute_paths This one is a c++ program, and it might need to recompile. This program will compute the paths (non-degree 2 node to another non-degree 2 node). The path information will then be used for smoothing the vector information.

5). Code/Vector/3d_edge_vector_info.py

### Simplification module.
The module corresponds to the simplification module in the paper. We use the vector information (calculated in the previous step) and density score information (will be calculated in this module) to decide which edges should be dropped. We have 4 scripts in this module.

1). [Only for STP data] Code/Simplification/BoundaryEdgeRemover.py Due to the zero-value background and degenerated gradient on those pixels in the cleaned STP data. we will have redundant segments in those regions, so we use this script to remove such segments.

2). Code/Simplification/ShortestPathForest.py Given the root file "multi_roots.txt", this script will extract a spanning forest out of the graph produced by previous steps.

3). Code/Simplification/Simplifier.py Simplification.

4). [Optional]  Code/Simplification/BranchSelector.py This script will select top n branches based on the length. It can provide a high-level abstraction without losing much information.

## Bash script and test samples.
The bash script for running the pipeline is available (pipeline_STP.sh and pipeline_fMOST.sh for STP and fMOST data respectively). The sample testing data are under folder STP_sample and fMOST_sample.

## Data
Datasets.zip contains all the data in vtk format. The ground truth for fMOST neurons are also included.

## Requirements
Please make sure Python3 is installed on your computer before running.
The following packages are also required:
