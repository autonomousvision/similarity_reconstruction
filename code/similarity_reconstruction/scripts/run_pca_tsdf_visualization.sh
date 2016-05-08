#!/bin/bash
# the binary file of reconstruction program
set -e

. ./init.sh

visualization_bin=$bin_dir/pca_tsdf_visualization
output_root=$result_root/rpca_test
if [ ! -d $output_root ]; then
    mkdir $output_root
fi
output_dir=$output_root/test3_maxweight5_sq_extended
output_dir=$output_root/test3_maxweight5_rob_noextend
output_dir=$output_root/test3_maxweight5_sq_extended3
output_dir=$output_root/test3_maxweight5_rob_extendedneg
output_dir=$output_root/test3_maxweight5_rob_extendedneg
output_dir=$output_root/test3_maxweight5_rob_origin
output_dir=$output_root/test3_maxweight5_qs_origin
output_dir=$output_root/test3_maxweight5_rob_origin
output_dir=$output_root/test3_maxweight5_sq_origin
output_dir=$output_root/test3_maxweight5_rob_originalign
output_dir=$output_root/test3_maxweight5_sq_originalign
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/res62_rob_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/res62_sq_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/res63_sq_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/res67_rob_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/res68_origin_rob_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/res68_origin_rob_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/res68_origin_sq_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/res12_originalign_rob_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/res12_originalign_sq_000.mat
input_scene_file=/home/dell/results_5/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-6050-ed-6350-vlen-0.2-rampsz-6-try1/recon-6050-6350-vlen-0.2-rampsz-6_tsdf.bin
boundingbox_size=" 51 51 51 "
mesh_min_weight=0

echo $visualization_bin --input_matlab_mat $input_matlab_mat --input_scene_file $input_scene_file --boundingbox_size $boundingbox_size --output_dir $output_dir --mesh_min_weight $mesh_min_weight --alsologtostderr
$visualization_bin --input_matlab_mat $input_matlab_mat --input_scene_file $input_scene_file --boundingbox_size $boundingbox_size --output_dir $output_dir --mesh_min_weight $mesh_min_weight --alsologtostderr
