#!/bin/bash

test_tsdf_align_bin=~/3d-reconstruction/zc_tsdf_hashing/test_tsdf_align
#in_model=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house1-n1-1.restored_tsdf.bin
#in_model=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house-poisson-1-1.restored_tsdf.bin
#in_model=/is/ps2/czhou/3d-reconstruction/zc_tsdf_hashing/test_joint_align_multiple_houses_2/house1-n1.restored_tsdf_init2_tsdf_model_0.bin
in_model=~/3d-reconstruction/zc_tsdf_hashing/debug-transform/h-joint-align-res_tsdf_modelply_0th_iter1.bin
#in_template=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house-poisson-1-1.restored_tsdf.bin
in_template=/is/ps2/czhou/3d-reconstruction/zc_tsdf_hashing/test_joint_align_multiple_houses_2/h-joint-align-res_init2_tsdf_template_.bin
output_dir=~/3d-reconstruction/zc_tsdf_hashing/debug-transform-res-2
mkdir $output_dir
out_ply=$output_dir/h-aligned.ply

#echo $test_tsdf_align_bin
gdb --args $test_tsdf_align_bin --in-model $in_model --in-template $in_template --out $out_ply --save_tsdf_bin
