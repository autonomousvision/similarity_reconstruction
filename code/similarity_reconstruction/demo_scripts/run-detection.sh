#!/bin/bash
# the binary file of reconstruction program
set -e

. ./init.sh

run_root=$result_root/refactor_detection_test2/
if [ ! -d $run_root ]; then
    mkdir $run_root
fi

sample_size=(9 9 6)
train_detect_delta=(0.5 0.5 1)
train_detect_delta=(1 1 1)
train_detect_delta=(1 1 2.5)
#train_detect_delta=(2 2 2.5)
#train_detect_delta=(0.2 0.2 1)
#train_detect_delta=(1 1 2)
# car: 0.2 0.2 2.5

detect_delta=(0.5 0.5 1)
detect_delta=(1 1 1)
detect_delta=(1 1 2.5)
#detect_delta=(0.2 0.2 1)
total_thread=10
min_score_to_keep=(-0.5 -0.5 0.1)

startimg_arr=(1470 1890 3320 6050)
endimg_arr=(1790 2090 3530 6350)
train_bin=$bin_dir/train_detectors_main
detect_bin=$bin_dir/detect_main

for i in 1 2 3
do
    stimg=${startimg_arr[$i]}
    edimg=${endimg_arr[$i]}
    scene_file=/home/dell/results_5/detection_training_data/training_data_$stimg/gt_cropped_$stimg-$edimg-building.cropped_tsdf_tsdf.bin
    annotations=/home/dell/results_5/detection_training_data/training_data_$stimg/annotated_obbs.txt
    output_prefix=/home/dell/results_5/detection_test5_$stimg-delta-${detect_delta[0]}-${detect_delta[1]}-${detect_delta[2]}-c100-w10/
    output_prefix=/home/dell/results_5/detection_test5_$stimg-delta-${detect_delta[0]}-${detect_delta[1]}-${detect_delta[2]}-c100-w10-random/
    #output_prefix=/home/dell/results_5/detection_test5_$stimg-delta-${detect_delta[0]}-${detect_delta[1]}-${detect_delta[2]}/
    #output_prefix=/home/dell/results_5/detection_test_$stimg-2/
    if [ ! -d $output_prefix ]; then
        mkdir $output_prefix
    fi
    scene_file_detection=/home/dell/results_5/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-$stimg-ed-$edimg-vlen-0.2-rampsz-6-try1/recon-$stimg-$edimg-vlen-0.2-rampsz-6_tsdf.bin
    detector_file=$output_prefix

    echo $train_bin --scene_file $scene_file --annotations $annotations --output_prefix $output_prefix --sample_size ${sample_size[@]} --total_thread $total_thread --detect_deltas ${train_detect_delta[@]} --svm_param_c 100 --svm_param_w1 10
    $train_bin --scene_file $scene_file --annotations $annotations --output_prefix $output_prefix --sample_size ${sample_size[@]} --total_thread $total_thread --detect_deltas ${train_detect_delta[@]} --svm_param_c 100 --svm_param_w1 10 --detect_deltas ${train_detect_delta[@]}
    echo $detect_bin --scene_file $scene_file_detection --detector_file $detector_file --output_prefix $output_prefix --total_thread $total_thread --detect_deltas ${detect_delta[@]} --min_score_to_keep ${min_score_to_keep[@]} --detect_deltas ${train_detect_delta[@]}
    $detect_bin --scene_file $scene_file_detection --detector_file $detector_file --output_prefix $output_prefix --total_thread $total_thread --detect_deltas ${detect_delta[@]} --min_score_to_keep ${min_score_to_keep[@]} --detect_deltas ${train_detect_delta[@]}
done
