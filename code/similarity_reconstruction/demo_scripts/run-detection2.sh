#!/bin/bash
set -e

. ./init2.sh

run_root=$result_root/refactor_detection_test_smalldetectdeltas/
run_root=$result_root/refactor_detection_test_smalldetectdeltas-minoccupy0.01-emptyfeatne2g/
run_root=$result_root/refactor_detection_test_smalldetectdeltas-minoccupy0.01-emptyfeatneg3-extend/
run_root=$result_root/detection_old_detectors/
run_root=/home/dell/results_5/refactor_detection_final2_adjust/
run_root=/home/dell/results_5/refactor_detection_final1_replaced_adjusted_atfinal3/
if [ ! -d $run_root ]; then
    mkdir $run_root
fi
#detect_deltas=(3 3 5)
#detect_deltas=(0.5 0.5 1)
detect_deltas=(1 1 1)
test_detect_deltas=(0.5 0.5 1)
#detect_deltas=(0.5 0.5 1)
detect_sample_size=(9 9 6)
total_thread=10
min_score_to_keep=(-0.5 -0.5 0.1)

startimg_arr=(1470 1890 3320 6050)
endimg_arr=(1790 2090 3530 6350)
train_bin=$bin_dir/train_detectors_main
detect_bin=$bin_dir/detect_main

for i in 0
do
    stimg=${startimg_arr[$i]}
    edimg=${endimg_arr[$i]}
    train_scene_file=$detector_train_data_dir/training_data_$stimg/gt_cropped_$stimg-$edimg-building.cropped_tsdf_tsdf.bin
    annotations=$detector_train_data_dir/training_data_$stimg/annotated_obbs.txt
    output_prefix=$run_root/detection-$stimg-min_occupy_adaptive/
    if [ ! -d $output_prefix ]; then
        mkdir $output_prefix
    fi
    detection_scene_file=$result_root/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-$stimg-ed-$edimg-vlen-0.2-rampsz-6-try1/recon-$stimg-$edimg-vlen-0.2-rampsz-6_tsdf.bin
    detector_file_dir=$output_prefix

    echo $train_bin --scene_file $train_scene_file --annotations $annotations --output_prefix $output_prefix --sample_size ${detect_sample_size[@]} --total_thread $total_thread --svm_param_c 100 --svm_param_w1 10 
    $train_bin --scene_file $train_scene_file --annotations $annotations --output_prefix $output_prefix --sample_size ${detect_sample_size[@]} --total_thread $total_thread --svm_param_c 100 --svm_param_w1 10 --detect_deltas ${detect_deltas[@]}

    echo $detect_bin --scene_file $detection_scene_file --detector_file $detector_file_dir --output_prefix $output_prefix --total_thread $total_thread --min_score_to_keep ${min_score_to_keep[@]} 
    $detect_bin --scene_file $detection_scene_file --detector_file $detector_file_dir --output_prefix $output_prefix --total_thread $total_thread --min_score_to_keep ${min_score_to_keep[@]} --detect_deltas ${test_detect_deltas[@]}
done
