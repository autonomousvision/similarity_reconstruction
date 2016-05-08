#!/bin/bash
# the binary file of reconstruction program
set -e

. ./init.sh
run_root=$result_root/seperate-seq-batch-all-detector-try3-cameraready-noscale-iccvres3-interleave-align-pca20-allsamples-refactered1/
if [ ! -d $run_root ]; then
    mkdir $run_root
fi
sample_size=(9 9 6)
train_detect_delta=(0.5 0.5 2)
train_detect_delta=(0.2 0.2 1)
#train_detect_delta=(1 1 2)
# car: 0.2 0.2 2.5

detect_delta=(0.5 0.5 2)
detect_delta=(0.2 0.2 1)
total_thread=10

startimg_arr=(1470 1890 3320 6050)
endimg_arr=(1790 2090 3530 6350)
train_bin=$bin_dir/train_detectors_main
detect_bin=$bin_dir/detect_main
optimization_bin=$bin_dir/test_tsdf_optimization

for i in 0
do
    stimg=${startimg_arr[$i]}
    edimg=${endimg_arr[$i]}
    scene_file=/home/dell/results_5/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-$stimg-ed-$edimg-vlen-0.2-rampsz-6-try1/recon-$stimg-$edimg-vlen-0.2-rampsz-6_tsdf.bin
    scene_file=/home/dell/results_5//seperate-seq-batch-all-detector-try3-cameraready-noscale-iccvres3-interleave-align-pca20-allsamples-refactered-none//batch-res2-1470-1790-building-svmw1-10-svmc-100/joint_learn-rebut-0-mergescore_thresh--0-noisecounter-3-noisecompo--1-iter-3-pcanum-0/init_consistency_check-0-mergescore_thresh--0-noisecounter-3-noisecompo--1-iter-3-pcanum-0--1-2/out.tsdf_consistency_cleaned_tsdf.bin

    detect_obb_file=/home/dell/results_5/detection_results/seq-$stimg-$edimg/building/merged_model.obb_infos.txt
    #scene_file=/home/dell/results_5/detection_training_data/training_data_$stimg/gt_cropped_$stimg-$edimg-building.cropped_tsdf_tsdf.bin
    #annotations=/home/dell/results_5/detection_training_data/training_data_$stimg/annotated_obbs.txt
    output_prefix=/home/dell/results_5/test_optimization_$stimg-5-delta-0.2-0.2-1/
    if [ ! -d $output_prefix ]; then
        mkdir $output_prefix
    fi
    #scene_file_detection=/home/dell/results_5/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-$stimg-ed-$edimg-vlen-0.2-rampsz-6-try1/recon-$stimg-$edimg-vlen-0.2-rampsz-6_tsdf.bin
    #detector_file=$output_prefix

    #echo $train_bin --scene_file $scene_file --annotations $annotations --output_prefix $output_prefix --sample_size ${sample_size[@]} --total_thread $total_thread --detect_deltas ${train_detect_delta[@]} --svm_param_c 1 --svm_param_w1 10
#   # $train_bin --scene_file $scene_file --annotations $annotations --output_prefix $output_prefix --sample_size ${sample_size[@]} --total_thread $total_thread --detect_deltas ${train_detect_delta[@]} --svm_param_c 1 --svm_param_w1 10
    #echo $detect_bin --scene_file $scene_file_detection --detector_file $detector_file --output_prefix $output_prefix --total_thread $total_thread --detect_deltas ${detect_delta[@]}
    #$detect_bin --scene_file $scene_file_detection --detector_file $detector_file --output_prefix $output_prefix --total_thread $total_thread --detect_deltas ${detect_delta[@]}
    echo $optimization_bin --scene_file $scene_file --detect_obb_file $detect_obb_file --output_prefix $output_prefix
    $optimization_bin --scene_file $scene_file --detect_obb_file $detect_obb_file --output_prefix $output_prefix
done
