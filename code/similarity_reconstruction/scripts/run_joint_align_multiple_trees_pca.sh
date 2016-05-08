#!/bin/bash
set -e
original_mesh_dir=/is/ps2/czhou/3d-reconstruction/zc_tsdf_hashing/tree-sliced-semantic-obb-2D-0.97a-3/
test_tsdf_align_bin=~/3d-reconstruction/zc_tsdf_hashing/test_joint_align_pca
output_dir=~/3d-reconstruction/zc_tsdf_hashing/joint_align_pca_trees_1/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi

for pca_num in 1 2
do
cnt=0
argument_input=""
for i in {1..6} 
do
    output_tsdf_restored_bin=$original_mesh_dir"tree-semantic_tsdf_sliced_"$i".bin"
    in_model_newhouse[$cnt]=$output_tsdf_restored_bin
    argument_input=$argument_input" "$output_tsdf_restored_bin
    cnt=$((cnt + 1))
done
    out_ply=$output_dir"/out_tree.ply"
    echo $out_ply

    echo $test_tsdf_align_bin --in-models $argument_input  --out $out_ply --save_tsdf_bin --pca_number $pca_num --max_iter 30
    $test_tsdf_align_bin --in-models $argument_input --out $out_ply --save_tsdf_bin --pca_number $pca_num --max_iter 30
done
