#!/bin/bash
# this is without semantic...
# the root folder for all the data
data_root=/home/dell/Data/data_1/2013_05_28_drive_0000_sync/image_02/rect/
# the binary file of reconstruction program
recon_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/fisheye_depths_fusion
# the starting and ending image numbers for reconstruction
input_tsdf_file=/home/dell/link_to_results/output-3d-model-semantic2/ply-0-600-0.5-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-2.0_bin_tsdf_file.bin
startimg=0
endimg=1000
param_prefix=param2
image_prefix=img_00
depth_prefix=simplefusion-spherr-supp-0_5-newdata1-keep-saturate-801-1301
#depth_prefix=simplefusion-spherr-supp-0.5-newdata1-801-1301
# semantic folder name inside data_root
# semantic_pref=manual_annotation_1
# output directory
output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
outputdir=$output_root"output-3d-model-cam2-test1-no-move-center"
mkdir $outputdir
set -e

# weight: the minimum allowed weight for meshing
# niteration: how many times the diffusion smoothing is executed
for weight in 0.6
do
for niteration in 0
do
for voxellen in 0.2
do
maxdist=$(echo "$voxellen * 8" | bc)
echo $maxdist
outputfile_conf=$outputdir/ply-$startimg-$endimg-$weight-newtest-conf-flatten-$niteration-noclean-voxellen-$voxellen-maxdist-$maxdist
#outputfile_conf_tsdfbin=$outputdir/tsdf-$startimg-$endimg-$weight-newtest-conf-flatten-$niteration-noclean-voxellen-$voxellen-maxdist-$maxdist
echo "$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --tsdf-filepath $outputfile_conf"_bin_tsdf_file.bin" --use_confidence --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix  --tsdf-filepath $input_tsdf_file"

#$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --tsdf-filepath $outputfile_conf"_bin_tsdf_file.bin" --use_confidence --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix 
#$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --use_confidence --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix --input-tsdf-filepath $input_tsdf_file
echo "$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --use_confidence --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix"
$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --use_confidence --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix
done
done
done
