#!/bin/bash
# this is without semantic...
# the root folder for all the data
data_root=/home/dell/Data/newdata_2/rect/
# the binary file of reconstruction program
recon_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/fisheye_depths_fusion

#input_tsdf_file=/home/dell/link_to_results/output-3d-model-newdata-cam2/ply-0-3000-1.0-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-1.6_bin_tsdf_file.bin
# the starting and ending image numbers for reconstruction
#input_tsdf_file=/home/dell/link_to_results/output-3d-model-semantic2/ply-0-600-0.5-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-2.0_bin_tsdf_file.bin
#input_tsdf_file=/home/dell/link_to_results/output-3d-model-newdata-cam2-new3/ply-642-1043-0-newtest-conf-flatten-0-noclean-voxellen-0.2-posmaxdist-1.6-negmaxdist--.8_bin_tsdf_file.bin
startimg=1930
endimg=1960
param_prefix=param_scale_4
image_prefix=img_00_scale_4
#depth_prefix=simplefusion-spherr-supp-0.5-newdata1-keep-saturate-801-1301
#depth_prefix=simplefusion-spherr-supp-0.5-newdata1
#depth_prefix=simplefusion-spherr-supp-0.5-newdata1-6050-debug-keep-satur-more-crop
depth_prefix=depth_00_slic_cropped_scale_4_filtered_masked
#depth_prefix=test0
# semantic folder name inside data_root
# semantic_pref=manual_annotation_1
# output directory
output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
outputdir=$output_root"output-debug-newdata6"
mkdir $outputdir
set -e

# weight: the minimum allowed weight for meshing
# niteration: how many times the diffusion smoothing is executed
neighbor_block_add_range=1
maxcamdist=30
margin=0
for weight in 0
do
for niteration in 0
do
for voxellen in 0.2
do
pos_maxdist=$(echo "$voxellen * 8" | bc -l)
neg_maxdist=$(echo "$voxellen * -4" | bc -l)
# in [fullweight_delta, 0] the weight will remain 1
negative_fullweight_delta=$(echo "-$voxellen/5.0 " | bc -l)
# in [inflection_distance, fullweight_delta] the weight will go from 1 to inflection_weight
negative_inflection_distance=$(echo "$voxellen * -3" | bc -l)
negative_inflection_weight=0.05

echo $pos_maxdist
echo $neg_maxdist
echo $negative_fullweight_delta
echo $negative_inflection_distance
echo $negative_inflection_weight

outputfile_conf=$outputdir/ply-$startimg-$endimg-$weight-newtest-noconf-flatten-$niteration-noclean-voxellen-$voxellen-posmaxdist-$pos_maxdist-negmaxdist-$neg_maxdist
#outputfile_conf_tsdfbin=$outputdir/tsdf-$startimg-$endimg-$weight-newtest-conf-flatten-$niteration-noclean-voxellen-$voxellen-maxdist-$maxdist

#$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --tsdf-filepath $outputfile_conf"_bin_tsdf_file.bin" --use_confidence --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix 
#echo $recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix 
#echo $recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --pos_truncation_limit $pos_maxdist --neg_truncation_limit $neg_maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix
echo $recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --pos_truncation_limit $pos_maxdist --neg_truncation_limit $neg_maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix --neighbor_add_limit $neighbor_block_add_range --max-camera-distance $maxcamdist --not_use_side_column_length $margin --neg_full_weight_delta $negative_fullweight_delta --neg_weight_dist_thresh $negative_inflection_distance --neg_weight_thresh $negative_inflection_weight
$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --pos_truncation_limit $pos_maxdist --neg_truncation_limit $neg_maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix --neighbor_add_limit $neighbor_block_add_range --max-camera-distance $maxcamdist --not_use_side_column_length $margin --neg_full_weight_delta $negative_fullweight_delta --neg_weight_dist_thresh $negative_inflection_distance --neg_weight_thresh $negative_inflection_weight
#--input-tsdf-filepath $input_tsdf_file
done
done
done
