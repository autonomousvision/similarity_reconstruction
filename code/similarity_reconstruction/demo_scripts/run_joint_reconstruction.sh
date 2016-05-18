#!/bin/bash
set -e

#. ./init_paths.sh

if ! [ "$detection_root" ]; then
    detection_root=$result_root/run_detection/
fi

run_root=$result_root/run_optimization/
if [ ! -d $run_root ]; then
    mkdir $run_root
fi
mesh_min_weight=0
max_cam_distance=30

if [[ $run_joint_opt && $run_joint_opt -eq 0 ]]; then
    do_init_consistency_check=0
    do_joint_learn=0
    do_final_consistency_check=0
else
    do_init_consistency_check=1
    do_joint_learn=1
    do_final_consistency_check=1
fi

echo "################ loading detections ####################"
scene_model=$scene_model_bin
detect_obb_file=$detect_res_txt
echo scene_model $scene_model
echo detect_obb_file $detect_obb_file

seq_output_root=$run_root/joint_reconstruction/
if [ ! -d $seq_output_root ]; then
    mkdir $seq_output_root
fi

echo "################## initial noise clean #################"
consistency_check_root=$seq_output_root/init_consistency_check/
echo consistency_check_root $consistency_check_root
if [ ! -d $consistency_check_root ]; then
    mkdir $consistency_check_root
fi
do_consistency_check=$do_init_consistency_check
check_tsdf=1
depthmap_check=1
skymap_check=1
filter_noise=60
. ./consistency_check.sh
#consistent_tsdf_output=$out".tsdf_consistency_cleaned_tsdf.bin"

echo "################## joint optimization ###################"
cleaned_scene_model=$consistent_tsdf_output
lambda_avg_scale=100
lambda_regularization=50
noise_obs_thresh=2
lambda_obs=0.2
pc_num=0
. ./run_optimization.sh

echo "############### final noise cleaning ####################"
scene_model=$joint_output_tsdf
detect_obb_file=$output_obb
do_consistency_check=$do_final_consistency_check
check_tsdf=0
depthmap_check=1
skymap_check=1
filter_noise=120
consistency_check_root=$seq_output_root/final_consistency_check/
echo consistency_check_root $consistency_check_root
if [ ! -d $consistency_check_root ]; then
    mkdir $consistency_check_root
fi
. ./consistency_check.sh
echo "result output to: " $consistent_tsdf_output_ply

if [[ $display && $display -gt 0 ]]; then
    ./visualization.sh "final_result" "$visualization_txt"
fi
