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
    # echo $optimization_bin --scene_file $scene_file --detect_obb_file $detect_obb_file --output_dir $output_dir --lambda_avg_scale $lambda_avg_scale --lambda_reg $lambda_regularization --noise_observation_thresh $noise_obs_thresh --lambda_outlier $lambda_outlier --lambda_obs $lambda_obs
    # sleep 1
    $optimization_bin --scene_file $scene_file --detect_obb_file $detect_obb_file --output_dir $output_dir --lambda_avg_scale $lambda_avg_scale --lambda_reg $lambda_regularization --noise_observation_thresh $noise_obs_thresh --lambda_outlier $lambda_outlier --lambda_obs $lambda_obs
fi

joint_opt_outdir=$output_dir
joint_output_tsdf=$joint_opt_outdir/final_result/"merged_scene_tsdf.bin"
output_obb=$joint_opt_outdir/final_result/final_obbs.txt
visualization_txt=$joint_learn_our_dir/visualization/visualization.txt

