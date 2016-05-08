#!/bin/bash
build_root=../../../urban_reconstruction_build/hashmap
data_bin_dir=/ps/geiger/czhou/cars_semi_convex_hull/plys_converted_incomplete/
test_tsdf_align_bin=$build_root/bin/test_joint_align_pca
output_dir=$build_root/pca_cars_1_deflation_ortho
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi

set -e
for pca_num in 3
do
cnt=0
argument_input=""
for i in {1..6} 
do
    output_tsdf_restored_bin=$data_bin_dir"car_"$i".restored.restored_tsdf.bin"
    in_model_newhouse[$cnt]=$output_tsdf_restored_bin
    argument_input=$argument_input" "$output_tsdf_restored_bin
    cnt=$((cnt + 1))
done

    out_ply=$output_dir"/out_car.ply"
    echo $out_ply

    echo $test_tsdf_align_bin --in-models $argument_input  --out $out_ply --save_tsdf_bin --pca_number $pca_num --max_iter 30
    $test_tsdf_align_bin --in-models $argument_input --out $out_ply --save_tsdf_bin --pca_number $pca_num --max_iter 30
done
