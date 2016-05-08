#!/bin/bash
# the binary file of reconstruction program
set -e

. ./init.sh

visualization_bin=$bin_dir/tsdf_visualization

output_root=$result_root/tsdf_vis_debug2
if [ ! -d $output_root ]; then
    mkdir $output_root
fi

output_dir=$output_root/test_sq_grassman_weighted_houses_only_inlier
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/res3_rob_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/grassmanres2_iter1_originalign_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/grassmanres_car1_iter0_meanalign_cars_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/grassmanres_car1_iter0_meanalign_onlyinlier_cars_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/grassmanres_res12_originalign_inlier_sq_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/grassmanres_h1_iter0_meanalign_onlyinlier_cars_000.mat
input_matlab_mat=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/grassmanres_res_h12_originalign_inlier_sq_000.mat
data_var_name="data_mat"
weight_var_name="weight_mat"
data_var_name="reconstructed_data"
weight_var_name="reconstructed_weight"
input_scene_file=/home/dell/results_5/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-6050-ed-6350-vlen-0.2-rampsz-6-try1/recon-6050-6350-vlen-0.2-rampsz-6_tsdf.bin
boundingbox_size=" 51 51 51 "
mesh_min_weight=0
max_dist_pos=4.6
max_dist_neg=-1

echo $visualization_bin --input_matlab_mat $input_matlab_mat --input_scene_file $input_scene_file --boundingbox_size $boundingbox_size --output_dir $output_dir --mesh_min_weight $mesh_min_weight --alsologtostderr --data_var_name $data_var_name --weight_var_name $weight_var_name --max_dist_pos $max_dist_pos --max_dist_neg $max_dist_neg
$visualization_bin --input_matlab_mat $input_matlab_mat --input_scene_file $input_scene_file --boundingbox_size $boundingbox_size --output_dir $output_dir --mesh_min_weight $mesh_min_weight --alsologtostderr --data_var_name $data_var_name --weight_var_name $weight_var_name --max_dist_pos $max_dist_pos --max_dist_neg $max_dist_neg
