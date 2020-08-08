# DiscreteMorse
Discrete Morse-based pipeline for neuron tracing on tracer injection and single-neuron data sets. It will automatically generate a forest (a set of trees) summarization of the given neuron imaging data. The biorxiv version is on https://www.biorxiv.org/content/10.1101/2020.03.21.000323v1. 

## Step-by-step workflow

Our pipeline is divided in three modules, which is a little different from the paper. In the paper, we have so-called preprocessing module, here we don't. Because the sample inputs/data are already preprocessed and packed in vtk/image_stack format. Here the first module is Discrete Morse Graph Reconstruction module,  which generates the graph construction. The second one is vector-score module calculating the vector score. The last one is the simplification module.

### Discrete Morse Graph Reconstruction module.

DIPHA is used to efficiently compute persistence pairs and values within the Discrete Morse Graph Reconstruction algorithm.  The original DIPHA Code can be found at https://github.com/DIPHA/dipha - we made modifications to this code to output the information needed by the Discrete Morse Graph Algorithm - this modified code is included.

**1.) write_dipha_file_3d.py - Used to generate dipha input file from an image stack** 

parameters:  
input_dir - directory containing image stack - if image stack needs  any preprocessing (such as gaussian filter) this must be applied before and saved as an image stack.

dipha_output_filename - filename for dipha input file that gets created (binary file).

vert_filename - filename that stores vertex information (txt file).

**2.) dipha-graph-recon**

To build dipha, go to dipha-graph-recon directory and perform the following commands:  
```bash
mkdir build  
cd build  
cmake ..  
make  
```

Here is a sample command to run DIPHA:  
```bash
mpiexec -n 32 ./dipha  path/to/input.bin path/to/output.bin path/to/output_for_morse.bin nx ny nz
```

mpiexec -n 32 is only required if running with multiple (in this case 32) processes. The rest of the command can be used to run on a single process.

path/to/input.bin - path to dipha input file generated in step 1  
path/to/output.bin - path to traditional dipha output file - this file is not used by our pipeline  
path/to/output_for_morse.bin - path to output file our pipeline uses, contains persistence information of edges  
nx ny nz - dimensions of image stack - if you are not sure what these values are, they are the last line printed in step 1 python script  

**3.) load_persistence_diagram.m**

This matlab script converts the dipha output file with edge persistence information to text format usable for step 4  
parameters:  
input file - outputted file from DIPHA  
output file - path to where edge file will be written  

**4.) dipha-output/src/** 

This outputs the actual Discrete Morse graph.  To build, simply enter the following command:  
```bash
g++ ComputeGraphReconstruction.cpp  
```

4 parameters:  
vert filename - vert file created in step 1  
edge filename - edge file created in step 3  
persistence threshold - perform the algorithm with the specific threshold - the higher the threshold the more simplified the output  
output dir - directory output will be written to (one vert file and one edge file)  


### Vector-score module. 
The module is used for calculating the vector scores. We have five scripts in total and will be introduced in order.

**1). Code/Vector/remove_dup_edges.py** 

This script will remove duplicate edges of the output from

parameters:  
input_filename - edge output file from Discrete Morse Graph Reconstruction module  
output_filename - name of output edge file that will not contain duplicate edges

**2). Code/Vector/wpca_vector_generation_bnb.py**

Computes a weighted principle component vector at each vertex of discrete morse output.  These vectors are used to calculate the local density flow at each vertex.

parameters:  
out_dir - the directory containing the discrete morse outputs, files generated by this file will be written here as well  
d_name - directory contain image stack.

**3). Code/Vector/vector_diffusion_3d.py**

Performs a gaussian smoothing of the vectors computed in step 2.  Takes the same two parameters as above, also outputting files containing the updated vectors to the same directory.

**4). Code/Vector/3d_paths_src/compute_paths**  

This one is a c++ program.  To compile, simply enter the following command: 
```bash
g++ ComputePaths.cpp 
```

This program will compute the paths (non-degree 2 node to another non-degree 2 node) of the output.

parameters:  
takes the directory of the morse output.  This is currently hard coded, please go to line 174 to change.

NOTE: remove dup edges must be run before this.

**5). Code/Vector/3d_edge_vector_info.py**

Outputs a file containing 4 scores for each edge - intensity at each node and a score between 0 and 1 at each node indicating how well alligned with true flow morse output is at the given node.  This information is used to further simplify the morse output in the simplification module.

parameters:  
IMAGE_DIR - directory of image stack.  
INPUT_FOLDER - the morse output folder containing morse output and now all additional vector information.  
PATH_RADIUS - Used to determine how many verts along a path will be used to estimate flow of morse output at a given vector.  We have found 4 to be effective.

### Simplification module.
The module corresponds to the simplification module in the paper. We use the vector information (calculated in the previous step) and density score information (will be calculated in this module) to decide which edges should be dropped. We have 4 scripts in this module.

**1). [Only for STP data] Code/Simplification/BoundaryEdgeRemover.py**

Due to the zero-value background and degenerated gradient on those pixels in the cleaned STP data. we will have redundant segments in those regions, so we use this script to remove such segments.

**2). Code/Simplification/ShortestPathForest.py**  

Given the root file "multi_roots.txt", this script will extract a spanning forest out of the graph produced by previous steps.

**3). Code/Simplification/Simplifier.py**  

Simplification. Provide options for Leafburner (LEAF) and Rootgrower (ROOT).

**4). [Optional]  Code/Simplification/BranchSelector.py** 

This script will select top n branches based on the length. It can provide a high-level abstraction without losing much information.

## Evaluation code for fMOST results
The evaluation code for fMOST results is also included in Code/fMOST_Evaluation. There is a sample bash script fMOST_eval.sh for evaluation.  

**1). Code/fMOST_Evaluation/Discretize.py**  

Given an output swc file, we first discretize it so that each pair of neighboring tree nodes are within small distance. This step is necessary because some methods (e.g., APP2) will generate very sparse tree outputs.

**2). Code/fMOST_Evaluation/Evaluate.py**

Compute recall/precision/F1-score given the path to discretized swc and the ground-truth swc files. The details for computing these metrics are elaborated in the manuscript.

## Bash script and test samples
The bash script for running the pipeline is available (pipeline_STP.sh and pipeline_fMOST.sh for STP and fMOST data respectively). The sample testing data are under folder STP_sample and fMOST_sample.  
The bash script for evaluting fMOST outputs is also attached (fMOST_eval.sh).

## Data
Datasets.zip contains all the data in vtk format. The data need to be first converted to image sequences before feeding into our pipeline (the DIPHAT module). The ground truth for fMOST neurons are also included.

The converted image stacks are also available under ImageSequences/.

## Other tools
Under Code/tools, there is a script for projecting swc files onto 2D planes, i.e., converting swc to image sequence.
To use it, you need to change the values of variables ``z_range, length, width`` in the code, these variables correspond to number of images and the size of the image.
```bash
python Projector.py $path_to_swc_file
```
The output will be in the same folder as the input swc file.

## Requirements
Please make sure Python3 is installed on your computer before running.
The following packages are also required:  
```
numpy  
scipy  
networkx  
scikit-learn  
```
