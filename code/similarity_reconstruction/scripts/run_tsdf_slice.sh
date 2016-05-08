#!/bin/bash

test_joint_align_bin=~/3d-reconstruction/zc_tsdf_hashing/test_tsdf_slice
in_model=~/3d-reconstruction/zc_tsdf_hashing/output-compare-newconf-semantic/ply-0-600-0.5-newtest-conf-flatten-1-noclean_bin_tsdf_file.bin
#in_model=~/3d-reconstruction/zc_tsdf_hashing/output-compare-newconf-semantic3/ply-0-50-0.5-newtest-conf-flatten-1-noclean_bin_tsdf_file.bin
in_box_param=~/3d-reconstruction/zc_tsdf_hashing/toy_sample_3house_slice/tsdf_slice_boxes
voxel_length=0.1
output_dir=~/3d-reconstruction/zc_tsdf_hashing/house-sliced-semantic-res-2Dthresh-1
if [ ! -d $output_dir ]; then
mkdir $output_dir
fi
out_ply=$output_dir/h-semantic.ply

#echo $test_tsdf_align_bin
gdb --args $test_joint_align_bin --in-model $in_model --in-box-param $in_box_param --voxel_length $voxel_length --out $out_ply --save_tsdf_bin
