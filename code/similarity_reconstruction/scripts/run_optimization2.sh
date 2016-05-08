#!/bin/bash
set -e
optimization_bin=$bin_dir/test_tsdf_optimization
scene_file=$cleaned_scene_model
#detect_obb_file=$detect_obb_file
output_root=$seq_output_root
output_dir=$output_root/joint_reconstruction-$pc_num-lavgscale-$lambda_avg_scale-lreg-$lambda_regularization/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi
output_prefix=$output_dir
out_ply=$output_dir"/joint-opt.ply"
if [ $do_joint_learn -gt 0 ]; then
    echo $optimization_bin --scene_file $scene_file --detect_obb_file $detect_obb_file --output_prefix $output_prefix --lambda_avg_scale $lambda_avg_scale --lambda_reg $lambda_regularization --noise_observation_thresh $noise_obs_thresh --lambda_outlier $lambda_outlier --lambda_obs $lambda_obs
    sleep 1
    $optimization_bin --scene_file $scene_file --detect_obb_file $detect_obb_file --output_prefix $output_prefix --lambda_avg_scale $lambda_avg_scale --lambda_reg $lambda_regularization --noise_observation_thresh $noise_obs_thresh --lambda_outlier $lambda_outlier --lambda_obs $lambda_obs
fi
joint_learn_outdir=$output_dir
joint_output_tsdf=$joint_learn_outdir/comp_num_$pc_num/".res.merged_tsdf_tsdf.bin"
output_obb=$joint_learn_outdir/comp_num_$pc_num/.res.plyfinalres.txt
