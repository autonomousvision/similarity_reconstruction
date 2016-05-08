#!/bin/bash
# the root folder for all the data
data_root=/home/dell/Data/test1/
cam_root=$data_root/image_03/
# the binary file of reconstruction program
bin_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/
recon_bin=$bin_root/fisheye_depths_fusion

# the input tsdf binary file for fuse reconstructions 
#input_tsdf_file=/home/dell/link_to_results/output-3d-model-camera2-test1/ply-0-200-0.5-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-1.6_bin_tsdf_file.bin
input_tsdf_file=/home/dell/link_to_results/output-3d-model-camera23-ntest1/ply-0-200-1.0-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-1.6_bin_tsdf_file.bin
# parameters
startimg=0
endimg=9
param_prefix=param2
image_prefix=img_00
depth_prefix=filter_depth_00
# weight: the minimum allowed weight for meshing
weight=0.8
# niteration: how many times the diffusion smoothing is executed
niteration=0
# voxel length
voxellen=0.2
# truncation distance
maxdist=$(echo "$voxellen * 8" | bc)
echo $maxdist

# output directory
output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
output_dir=$output_root"output-3d-model-camera3-ntest-withcam2-no-satarate/"
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi
outputfile_conf=$output_dir/ply-$startimg-$endimg-$weight-conf-flatten-$niteration-voxellen-$voxellen-maxdist-$maxdist
set -e

echo $recon_bin --in-root $cam_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix --input-tsdf-filepath $input_tsdf_file
$recon_bin --in-root $cam_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix --input-tsdf-filepath $input_tsdf_file
