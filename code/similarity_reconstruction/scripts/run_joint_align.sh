#!/bin/bash
build_root=../../../urban_reconstruction_build/hashmap/
data_dir=$build_root/model_objects/
test_tsdf_align_bin=$build_root/bin/test_joint_align
output_dir=$build_root/align-objects/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi
set -e

# get the names of input files
cnt=0
input_model_names=""
for i in {1..5}
do
    tsdf_bin=$data_dir"model_object_tsdf_sliced_"$i".bin"
    tsdf_bin_models[$cnt]=$output_tsdf_restored_bin
    echo ${tsdf_bin_models[$cnt]}
    input_model_names=$input_model_names" "$tsdf_bin
    cnt=$((cnt + 1))
done

out_ply=$output_dir"/joint-align-res.ply"
echo $out_ply
echo $test_tsdf_align_bin --in-models $input_model_names --out $out_ply --save_tsdf_bin
$test_tsdf_align_bin --in-models $input_model_names --out $out_ply --save_tsdf_bin
