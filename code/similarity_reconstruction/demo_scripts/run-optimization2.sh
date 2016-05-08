#!/bin/bash
# the binary file of reconstruction program
optimization_bin=$bin_dir/test_tsdf_optimization
scene_file=$merged_model
detect_obb_file=$merged_detect_box_txt
output_root=$joint_learn_output_root
output_dir=$output_root/refractor-joint-pca-$pca_num-lavgscale-$lambda_average_scale-lobs-$lambda_obs-lregrot-$lambda_reg_rot/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi
output_prefix=$output_dir
out_ply=$output_dir"/joint-opt.ply"
lambda_outlier=999999
if [ $do_joint_learn -gt 0 ]; then
    echo "#'$$$$$$$$$$$$$$$$$$$$'$ do joint learn "
    echo $optimization_bin --scene_file $scene_file --detect_obb_file $detect_obb_file --output_prefix $output_prefix --lambda_avg_scale $lambda_average_scale --lambda_obs $lambda_obs --lambda_reg $lambda_reg_rot --noise_observation_thresh $noise_counter_thresh --lambda_outlier $lambda_outlier
    sleep 2
    $optimization_bin --scene_file $scene_file --detect_obb_file $detect_obb_file --output_prefix $output_prefix --lambda_avg_scale $lambda_average_scale --lambda_obs $lambda_obs --lambda_reg $lambda_reg_rot --noise_observation_thresh $noise_counter_thresh --lambda_outlier $lambda_outlier
fi
joint_learn_outdir=$output_dir
joint_output_tsdf=$joint_learn_outdir/comp_num_$test_pcanum/".res.merged_tsdf_tsdf.bin"
