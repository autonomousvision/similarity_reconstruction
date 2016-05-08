#!/bin/bash
# the binary file of reconstruction program
set -e

. ./init.sh

train_detector_bin=$bin_dir/test_train_object_detector2

# input_tsdf
# input_tsdf=$outputprefix
#input_tsdf=/home/dell/upload2/4-10/reconstruction/closest2/merge_vri_recon2-1890-2090-rampsz-5_tsdf.bin

# input positive samples
#detected_obb_file=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_building2.txt
#detected_obb_file=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_house_origin_detection_debug.txt

# parameters
#vx=9
#vy=9
#vz=7
#detect_delta_x=0.5
#detect_delta_y=0.5
#detect_delta_rot=7.5 # in degree
#mesh_min_weight=0.0
#total_thread=8
#jitter_num=60

train_output_suffix="voxelsides-$vx-$vy-$vz-dx-$train_detect_delta_x-dy-$train_detect_delta_y-dr-$train_detect_delta_rot-jitter-$jitter_num-svmw1-$svm_w1-svmc-$svm_c"
# output_prefix
trained_svm_output_dir=$batch_output_root/train-svm-$train_output_suffix/
if [ ! -d "$trained_svm_output_dir" ]; then
    mkdir $trained_svm_output_dir
fi
trained_svm_path=$trained_svm_output_dir"/sample"
svm_model_suffix=".trained_svm_model.svm"

if [ $do_train_detector -gt 0 ]; then
echo "#################### train svm #####################"
echo $train_detector_bin --in-model $input_tsdf --detected-obb-file $detected_obb_file --out-dir-prefix $trained_svm_path --sample_num $sample_num --jitter_num $jitter_num --mesh-min-weight $mesh_min_weight --sample_voxel_sidelengths $vx $vy $vz --total_thread $total_thread --detect_deltas $train_detect_delta_x $train_detect_delta_y $train_detect_delta_rot --svm_param_c $svm_c --svm_param_w1 $svm_w1
sleep 5
$train_detector_bin --in-model $input_tsdf --detected-obb-file $detected_obb_file --out-dir-prefix $trained_svm_path --sample_num $sample_num --jitter_num $jitter_num --mesh-min-weight $mesh_min_weight --sample_voxel_sidelengths $vx $vy $vz --total_thread $total_thread --detect_deltas $train_detect_delta_x $train_detect_delta_y $train_detect_delta_rot --svm_param_c $svm_c --svm_param_w1 $svm_w1
fi

trained_svm_path=$trained_svm_path".trained_svm_model.svm"
