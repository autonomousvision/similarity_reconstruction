#!/bin/bash

test_joint_align_bin=~/3d-reconstruction/zc_tsdf_hashing/test_joint_align
#in_model=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house1-n1-1.restored_tsdf.bin
#in_models=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house-poisson-1-1.restored_tsdf.bin
in_models=/is/ps2/czhou/3d-reconstruction/zc_tsdf_hashing/house-meshes-joint-align-2/h-joint-align_tsdf_modelply_1th_iter3.bin
#in_template=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house-poisson-1-1.restored_tsdf.bin
#in_template=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house-poisson-1-1-rotate30-1.restored_tsdf.bin
output_dir=~/3d-reconstruction/zc_tsdf_hashing/house-meshes-test_transform
if [ ! -d $output_dir ]; then
mkdir $output_dir
fi
out_ply=$output_dir/h-joint-align.ply

#echo $test_tsdf_align_bin
gdb --args $test_joint_align_bin --in-models $in_models --out $out_ply --save_tsdf_bin
