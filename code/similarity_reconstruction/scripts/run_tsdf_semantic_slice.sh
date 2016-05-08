#!/bin/bash

# the root folder for all the data
data_root=/ps/geiger/czhou/2013_05_28/2013_05_28_drive_0000_sync/image_02/rect/
# the binary file for semantic slice
exe_bin=../../../urban_reconstruction_build/hashmap/bin/test_tsdf_semantic_slice
# the root of output data
output_root=../../../urban_reconstruction_build/hashmap/
# the input 3d model containing semantic label
in_model=$output_root/output-3d-model-semantic/ply-0-200-0.5-newtest-conf-flatten-1-noclean_bin_tsdf_file.bin
# annotation file folder
in_box_param_folder=$data_root"manual_annotation_1/"
# camera info file folder
in_camera_param_folder=$data_root"param2/"
# percent of semantic label points in the minimum bounding box
support_thresh=0.95

voxel_length=0.1
output_dir=$output_root"model_objects/"
if [ ! -d $output_dir ]; then
mkdir $output_dir
fi
out_ply=$output_dir/model_object.ply

echo "$exe_bin --in-model $in_model --in-box-param $in_box_param_folder --in-camera-param-folder $in_camera_param_folder --voxel_length $voxel_length --out $out_ply --save_tsdf_bin --support-thresh $support_thresh"
$exe_bin --in-model $in_model --in-box-param $in_box_param_folder --in-camera-param-folder $in_camera_param_folder --voxel_length $voxel_length --out $out_ply --save_tsdf_bin --support-thresh $support_thresh
