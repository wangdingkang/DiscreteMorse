Run swc2ve to first convert the swc file to vert and edge file.

swc2ve.py #swc_file

Then run projector.py to produce section wise geojson files.

Projector.py #vert_file #edge_file #z_range(number of sections) #length #width

You can also use ve2swc.py to convert vert and edge files back to swc for correctness check.
