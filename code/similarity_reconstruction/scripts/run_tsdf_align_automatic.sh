#!/bin/bash

test_tsdf_align_bin=~/3d-reconstruction/zc_tsdf_hashing/test_tsdf_align_automatic
#in_model=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house1-n1-1.restored_tsdf.bin
in_template=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house-poisson-1-1.restored_tsdf.bin
#in_template=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house-poisson-1-1.restored_tsdf.bin
#in_template=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house-poisson-1-1-rotate30-1.restored_tsdf.bin
output_dir=~/3d-reconstruction/zc_tsdf_hashing/house-meshes-usdf-usergrad-average-updated-2
mkdir $output_dir
out_ply=$output_dir/h-aligned.ply

echo $test_tsdf_align_bin
$test_tsdf_align_bin --in-template $in_template --out $out_ply --save_tsdf_bin
