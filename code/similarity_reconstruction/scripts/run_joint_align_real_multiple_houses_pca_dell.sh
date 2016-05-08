#!/bin/bash
set -e
data_dir=/home/dell/Data/results/aligned-houses/
test_tsdf_align_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/test_joint_align_pca
output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
output_dir=$output_root/pca_houses_1_deflation_ortho_origin
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi

for pca_num in 1
do
cnt=0
argument_input=""
for i in 0 1 2 
do
    output_tsdf_bin=$data_dir"h-joint-align_tsdf_sliced_"$i"_tsdf_model_"$i".bin"
    in_model_newhouse[$cnt]=$output_tsdf_bin
    echo ${in_model_newhouse[$cnt]}
    argument_input=$argument_input" "$output_tsdf_bin
    cnt=$((cnt + 1))
done
out_ply=$output_dir"/h-joint-align-res-pca-realdata.ply"
echo $out_ply
echo $test_tsdf_align_bin --in-models $argument_input --out $out_ply --save_tsdf_bin --pca_number $pca_num --max_iter 30
$test_tsdf_align_bin --in-models $argument_input --out $out_ply --save_tsdf_bin --pca_number $pca_num --max_iter 30
done
