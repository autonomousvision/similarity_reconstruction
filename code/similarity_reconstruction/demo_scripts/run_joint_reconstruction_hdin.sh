#!/bin/bash
set -e

. ./init_paths.sh

run_root=$result_root/run_optimization_"$startimg"/
if [ ! -d $run_root ]; then
    mkdir $run_root
fi
mesh_min_weight=0
max_cam_distance=30
lambda_obs=0.2
pc_num=0
if ! [ $display ]; then
    display=0
fi

echo "################ loading detections ####################"
if ! [ "$detection_scene_model" ]; then 
    detection_scene_model=/home/dell/results_5/demo_1470_1790_bilinear_w0.1/reconstruction_baseline-1470-1790/vri_fusing_result_side2_with_side1-1470-1790/recon-1470-1790_tsdf.bin
fi
# set the root for detection result.
if ! [ "$detection_root" ]; then
    detection_root=$result_root/run_detection/
fi
detect_obb_file=$detection_root/detect_res_all_obb_nmsed.txt
echo scene_model $detection_scene_model
echo detect_obb_file $detect_obb_file

echo "################## initial noise clean #################"
consistency_check_root=$run_root/init_consistency_check/
echo consistency_check_root $consistency_check_root
if [ ! -d $consistency_check_root ]; then
    mkdir $consistency_check_root
fi
do_consistency_check=1
check_tsdf=1
depthmap_check=1
skymap_check=1
filter_noise=60
. ./consistency_check2.sh

echo "################## joint optimization ###################"
do_joint_learn=1
cleaned_scene_model=$consistent_tsdf_output
lambda_avg_scale=100
lambda_regularization=50
lambda_outlier=999999999
noise_obs_thresh=2
. ./run_optimization2.sh

echo "############### final noise cleaning ####################"
scene_model=$joint_output_tsdf
detect_obb_file=$output_obb
do_consistency_check=1
check_tsdf=0
depthmap_check=0
skymap_check=1
filter_noise=120
consistency_check_root=$seq_output_root/final_consistency_check/
echo consistency_check_root $consistency_check_root
if [ ! -d $consistency_check_root ]; then
    mkdir $consistency_check_root
fi
. ./consistency_check2.sh
echo "result output to: " $consistent_tsdf_output_ply

if [ $display -gt 0 ]; then
    ./visualization.sh "result" "$visualization_txt"
fi
