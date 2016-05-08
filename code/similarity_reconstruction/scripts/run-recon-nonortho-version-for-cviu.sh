#!/bin/bash
# this is without semantic...
# the root folder for all the data
#data_root=/home/dell/Data/data_1/2013_05_28_drive_0000_sync/image_03/rect/
data_root=/home/dell/Data/data-4-10/2013_05_28_drive_0000_sync/image_02/rect
# the binary file of reconstruction program
recon_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/fisheye_depths_fusion
# the starting and ending image numbers for reconstruction
#input_tsdf_file=/home/dell/link_to_results/output-3d-model-semantic2/ply-0-600-0.5-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-2.0_bin_tsdf_file.bin

startimg=1470
endimg=1790
param_prefix=param_scale_4
image_prefix=img_00_scale_4
#depth_prefix=simplefusion-spherr-supp-0.5-newdata1-keep-saturate-801-1301
#depth_prefix=simplefusion-spherr-supp-0.5-newdata1-seq6050
depth_prefix=depth_00_slic_cropped_scale_4_filtered
# semantic folder name inside data_root
# semantic_pref=manual_annotation_1
# output directory
output_root=/home/dell/results_5/reconstruction_viewray1/
if [ ! -d $output_root ]; then
    mkdir $output_root
fi
outputdir=$output_root"output-3d-model-cam2test-seq1470"
if [ ! -d $outputdir ]; then
    mkdir $outputdir
fi
set -e

pos_trunc_limit=1.2
neg_trunc_limit=-1.0
# weight: the minimum allowed weight for meshing
# niteration: how many times the diffusion smoothing is executed
for weight in 0.0
do
for niteration in 0
do
for voxellen in 0.2
do
maxdist=$(echo "$voxellen * 6" | bc)
echo $maxdist
outputfile_conf=$outputdir/ply-$startimg-$endimg-$weight-newtest-conf-flatten-$niteration-noclean-voxellen-$voxellen-maxdist-$maxdist
#outputfile_conf_tsdfbin=$outputdir/tsdf-$startimg-$endimg-$weight-newtest-conf-flatten-$niteration-noclean-voxellen-$voxellen-maxdist-$maxdist
echo "$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --tsdf-filepath $outputfile_conf"_bin_tsdf_file.bin" --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix  --tsdf-filepath $input_tsdf_file"

#$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --tsdf-filepath $outputfile_conf"_bin_tsdf_file.bin" --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix 
$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --pos_truncation_limit $pos_trunc_limit --neg_truncation_limit $neg_trunc_limit --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix 
done
done
done
