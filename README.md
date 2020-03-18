# DiscreteMorse
Discrete Morse-based pipeline for neuron tracing on tracer injection and single-neuron data sets. It will automatically generate a forest (a set of trees) summarization of the given neuron imaging data. Paper:...

## Step-by-step workflow

Our pipeline is composed of two main modules, the first one is discrete-morse based graph reconstruction module. The second one is simplification module to remove potential false positives from DM outputs.

### Discrete-Morse based graph reconstruction.

The input to our DM-based module is a image stack, currently we support 8-bit or 16-bit grayscale TIFF format image sequences.

The outputs are vert.txt, edge.txt corresponding the graph extracted by our DM-based module. 



### Simplification and summarization.

The simplification module simply takes the output from the last step, i.e., the vert.txt and edge.txt files as input. The user also need to provide a multi_root.txt file for specifying the posistion of the roots for forest summarization.

The outputs are multiple trees, each tree will have its own vert and edge files.


## Data
The data in vtk format are available at . The ground truth for fMOST neurons are also available.

## Requirements
Please make sure Python3 is installed on your computer before running.
The following packages are also required:
