## Work in progress.
More code about regional analysis will be uploaded later.

### Python scripts...
STP_mouselight_searcher.py
This python script will find all the neurons in the given directory, which have soma location in the injection site.
```
File paths are hard-coded, plz change accordingly.
```

Projector.py
This python script will create a image stack from a swc file. Usage:
```
python Projector.py $swc_file_path $z $x $y
```
It will automatically create a folder using the swc filename, and output $z images (sections) into that folder.

### Workflow...
![Image of overlapping workflow](https://raw.githubusercontent.com/wangdingkang/DiscreteMorse/master/Code/Mouselight_Atlas_Space_Analysis/Workflow_images/Mouselight_Overlapping_Workflow.png)

### In MouseLight directory...
MouseLightMOinfo.txt: The list containing ids of all the candidate mouselight neurons.

Injection atlas: The injection site of the STP 180830 brain in atlas space.

jsons: The json files for all mouselight neuron, basically it is point cloud file, but also contains other meta info about each neuron.
There are multiple json files, each contains the information of several mouselight neurons.
