#!/bin/bash
# the root folder for all the data
data_root=/home/dell/Data/newdata_2/rect/
# the binary file of depthmap-meshing program
recon_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/depthmap_triangle_mesh_main
startimg=1890
endimg=1895
param_prefix=param_scale_4
#image_prefix=img_00_scale_4
depth_prefix=depth_00_slic_cropped_scale_4_filtered_masked
output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
outputdir=$output_root"output-depthmap-meshing"
mkdir $outputdir
set -e
dd_factor=0
maxcamdist=30
margin=0

$recon_bin --in-root $data_root --out $outputdir --depth-prefix $depth_pref --param-prefix $param_prefix --dd_factor $dd_factor --max-camera-distance $maxcamdist --startimage $startimg --endimage $endimg --voxel_length 0.2 --flatten --margin $margin
#--dd_factor 2.0
