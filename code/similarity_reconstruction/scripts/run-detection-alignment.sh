#!/bin/bash
# the binary file of reconstruction program
set -e

. ./init.sh

train_detector_bin=$bin_dir/test_train_object_detector2
detection_bin=$bin_dir/test_sliding_window_object_detector2
pr_compute_bin=$bin_dir/compute_precision_recall_curve

# input_tsdf
# input_tsdf=$outputprefix
input_tsdf=/home/dell/upload2/4-10/reconstruction/closest2/merge_vri_recon2-1890-2090-rampsz-5_tsdf.bin

# input positive samples
detected_obb_file=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_building2.txt
#detected_obb_file=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_house_origin_detection_debug.txt

# parameters
vx=9
vy=9
vz=6
output_suffix="_try8"

detect_delta_x=0.5
detect_delta_y=0.5
detect_delta_rot=7.5 # in degree
mesh_min_weight=0.0
total_thread=8
jitter_num=60

# output_prefix
trained_svm_output_dir=$result_root/train_svm-feat_voxlen-$vx-$vy-$vz-$output_suffix/
trained_svm_path=$trained_svm_output_dir"/house_newtsdf1"
svm_model_suffix="trained_svm_model.svm"

do_NMS=1

### 1. train svm
echo "#################### train svm #####################"
if [ ! -d "$trained_svm_output_dir" ]; then
    mkdir $trained_svm_output_dir
fi
echo $train_detector_bin --in-model $input_tsdf --detected-obb-file $detected_obb_file --out-dir-prefix $trained_svm_path --sample_num 1000 --jitter_num 0 --mesh-min-weight $mesh_min_weight --sample_voxel_sidelengths $vx $vy $vz --total_thread $total_thread --detect_deltas $detect_delta_x $detect_delta_y $detect_delta_rot
$train_detector_bin --in-model $input_tsdf --detected-obb-file $detected_obb_file --out-dir-prefix $trained_svm_path --sample_num 1000 --jitter_num $jitter_num --mesh-min-weight $mesh_min_weight --sample_voxel_sidelengths $vx $vy $vz --total_thread $total_thread --detect_deltas $detect_delta_x $detect_delta_y $detect_delta_rot

trained_svm_path=$trained_svm_path".trained_svm_model.svm"
detect_output_dir=$result_root/svm_detect_res-$vx-$vy-$vz-$output_suffix/
detect_output_prefix=$detect_output_dir"/house_newtsdf1"
### 2. test svm
echo "#################### test svm #####################"
input_svm_path=$trained_svm_path
if [ ! -d "$detect_output_dir" ]; then
    mkdir $detect_output_dir
fi
echo $detection_bin --in-model $input_tsdf --out-dir-prefix $detect_output_prefix --detected-obb-file $detected_obb_file --mesh-min-weight $mesh_min_weight --svm_model $input_svm_path --delta_x $detect_delta_x --delta_y $detect_delta_y --rotate_degree $detect_delta_rot --total_thread $total_thread  --sample_voxel_sidelengths $vx $vy $vz
$detection_bin --in-model $input_tsdf --out-dir-prefix $detect_output_prefix --detected-obb-file $detected_obb_file --mesh-min-weight $mesh_min_weight --svm_model $input_svm_path --delta_x $detect_delta_x --delta_y $detect_delta_y --rotate_degree $detect_delta_rot --total_thread $total_thread  --sample_voxel_sidelengths $vx $vy $vz

detect_res_path=$detect_output_prefix"_SlidingBoxDetectionResults_Parallel_Final.txt"

### 3. compute pr curve
echo "#################### compute pr curve #####################"
pr_curve_output_dir=$result_root/svm_detect_pr-$vx-$vy-$vz-$output_suffix/
if [ ! -d "$pr_curve_output_dir" ]; then
    mkdir $pr_curve_output_dir
fi
pr_curve_output_prefix=$pr_curve_output_dir"/house_1"
nms_res=$pr_curve_output_prefix"NMS_res.txt"
if [ $do_NMS -gt 0 ]; then
    nms_option=""
else
    nms_option="--input_nms_file $nms_res"
fi
echo $pr_compute_bin --out-dir-prefix $pr_curve_output_prefix --mesh-min-weight $mesh_min_weight --detect_output_file $detect_res_path --sample_voxel_sidelengths $vx $vy $vz --detected-obb-file $detected_obb_file $nms_option
sleep 2
$pr_compute_bin --out-dir-prefix $pr_curve_output_prefix --mesh-min-weight $mesh_min_weight --detect_output_file $detect_res_path --sample_voxel_sidelengths $vx $vy $vz --detected-obb-file $detected_obb_file $nms_option

### 
