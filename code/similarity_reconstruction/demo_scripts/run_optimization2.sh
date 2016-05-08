#!/bin/bash
#lambda_avg_scale=0
#lambda_regularization=0
set -e
optimization_bin=$bin_dir/test_tsdf_optimization
scene_file=$cleaned_scene_model
#detect_obb_file=$detect_obb_file
output_dir=$run_root/joint_reconstruction-$pc_num-lavgscale-$lambda_avg_scale-lreg-$lambda_regularization/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi
if [ $do_joint_learn -gt 0 ]; then
    echo $optimization_bin --scene_file $scene_file --detect_obb_file $detect_obb_file --output_dir $output_dir --lambda_avg_scale $lambda_avg_scale --lambda_reg $lambda_regularization --noise_observation_thresh $noise_obs_thresh --lambda_outlier $lambda_outlier --lambda_obs $lambda_obs
    sleep 1
    $optimization_bin --scene_file $scene_file --detect_obb_file $detect_obb_file --output_dir $output_dir --lambda_avg_scale $lambda_avg_scale --lambda_reg $lambda_regularization --noise_observation_thresh $noise_obs_thresh --lambda_outlier $lambda_outlier --lambda_obs $lambda_obs
fi

#############################
opt_bin_old=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/test_joint_cluster_model
lambda_average_scale=$lambda_avg_scale
lambda_reg_rot=$lambda_regularization
lambda_reg_scale=$lambda_regularization
lambda_reg_trans=$lambda_regularization
lambda_reg_zscale=0
max_iter=3
detected_box_txt=$output_dir/oldobbs.txt
old_out_dir=$output_dir/oldresult
if [ ! -d $old_out_dir ]; then
    mkdir $old_out_dir
fi
out_ply=$old_out_dir/oldres.ply
echo $opt_bin_old --in-model $scene_file --detect-box-txt $detected_box_txt -out-prefix $out_ply --pca_number $pc_num --optimize-max-iter $max_iter --mesh_min_weight 0.00 --lambda-average-scale $lambda_average_scale --lambda-obs $lambda_obs --lambda-reg-rot $lambda_reg_rot --lambda-reg-scale $lambda_reg_scale --lambda-reg-trans $lambda_reg_trans --noise_clean_counter_thresh $noise_obs_thresh --noise_connected_component_thresh -1 --lambda_reg_zscale 0 --lambda_out $lambda_outlier
sleep 1
$opt_bin_old --in-model $scene_file --detect-box-txt $detected_box_txt --out-prefix $out_ply --pca_number $pc_num --optimize-max-iter $max_iter --mesh_min_weight 0.00 --lambda-average-scale $lambda_average_scale --lambda-obs $lambda_obs --lambda-reg-rot $lambda_reg_rot --lambda-reg-scale $lambda_reg_scale --lambda-reg-trans $lambda_reg_trans --noise_clean_counter_thresh $noise_obs_thresh --noise_connected_component_thresh -1 --lambda_reg_zscale 0 --lambda_out $lambda_outlier
#############################

joint_opt_outdir=$output_dir
joint_output_tsdf=$joint_opt_outdir/final_result/"merged_scene_tsdf.bin"
output_obb=$joint_opt_outdir/final_result/final_obbs.txt
visualization_txt=$joint_learn_our_dir/visualization/visualization.txt

