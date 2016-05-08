#!/bin/bash
build_root=../../../urban_reconstruction_build/hashmap
data_dir=/home/dell/Data/results/house-sliced-res-1/
test_tsdf_align_bin=$build_root/bin/test_joint_align

# get the names of input files
cnt=0
for i in 0 1 2 
do
    output_tsdf_restored_bin=$data_dir"h-joint-align_tsdf_sliced_"$i".bin"
    in_model_newhouse[$cnt]=$output_tsdf_restored_bin
    echo ${in_model_newhouse[$cnt]}
    cnt=$((cnt + 1))
done

output_dir=$build_root/align-real-houses/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi
out_ply=$output_dir"/h-joint-align-res-realdata.ply"
echo $out_ply
echo $test_tsdf_align_bin --in-models ${in_model_newhouse[0]} ${in_model_newhouse[1]} ${in_model_newhouse[2]} --out $out_ply --save_tsdf_bin
$test_tsdf_align_bin --in-models ${in_model_newhouse[0]} ${in_model_newhouse[1]} ${in_model_newhouse[2]} --out $out_ply --save_tsdf_bin

# rotate/translate/scale the house models in the program to test the correctness of alignment
output_dir=$build_root/align-real-houses-rotate/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi
out_ply=$output_dir"/h-joint-align-res-realdata-rotate.ply"
echo $out_ply
echo $test_tsdf_align_bin --in-models ${in_model_newhouse[0]} ${in_model_newhouse[1]} ${in_model_newhouse[2]} --out $out_ply --save_tsdf_bin --rotate-model
$test_tsdf_align_bin --in-models ${in_model_newhouse[0]} ${in_model_newhouse[1]} ${in_model_newhouse[2]} --out $out_ply --save_tsdf_bin --rotate-model
