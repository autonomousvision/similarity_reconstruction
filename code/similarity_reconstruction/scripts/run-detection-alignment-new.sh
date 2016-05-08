#!/bin/bash
# the binary file of reconstruction program
set -e

. ./init.sh

merge_bin=$bin_dir/merge_tsdfs_obbs

run_root=$result_root/batch_run8
if [ ! -d $run_root ]; then
    mkdir $run_root
fi

vx=9
vy=9
vz=6

train_detect_delta_x=1
train_detect_delta_y=1
train_detect_delta_rot=7.5 # in degree

detect_delta_x=1
detect_delta_y=1
detect_delta_rot=7.5 # in degree

mesh_min_weight=0.0
total_thread=8
jitter_num=10
sample_num=2000
output_suffix="try1-voxelsides-$vx-$vy-$vz-dx-$detect_delta_x-dy-$detect_delta_y-dr-$detect_delta_rot-jitter-$jitter_num"
lowest_score_to_supress=-0.5

category_arr=(building car)
startimg_arr=(1470 1890 3320 6050)
endimg_arr=(1790 2090 3530 6350)

do_train_detector=0
do_detection=0
do_NMS=1
do_pr=0
do_merge_tsdf_obb=0
do_joint_learn=1

declare -A detected_obb_file_arr
for seq_i in 0 1 2 3
do
    startimg=${startimg_arr[$seq_i]}
    endimg=${endimg_arr[$seq_i]}
    input_tsdf_arr[$seq_i]=$result_root/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-$startimg-ed-$endimg-vlen-0.2-rampsz-6-try1/recon-$startimg-$endimg-vlen-0.2-rampsz-6_tsdf.bin

    for ci in 0 #1
    do
        category=${category_arr[$ci]}
        detected_obb_file_arr[$seq_i $ci]=/home/dell/Data/download_labels/label_4_12/gt_4_12_$startimg-$endimg-$category/gt_$startimg-$endimg-$category".txt"
        #echo ${detected_obb_file_arr[$seq_i $ci]}
    done
done

test_seq_idx[0]="0 1 2 3"
test_seq_idx[1]="1 0 2 3"
test_seq_idx[2]="2 0 1 3"
test_seq_idx[3]="3 0 1 2"

for ci in 0 #1
do
for seq_i in 0 #1 2 3
do
    startimg=${startimg_arr[$seq_i]}
    endimg=${endimg_arr[$seq_i]}
    category=${category_arr[$ci]}
    input_tsdf=${input_tsdf_arr[$seq_i]}
    detected_obb_file=${detected_obb_file_arr[$seq_i $ci]}
    echo startimg $startimg
    echo endimg $endimg
    echo category $category
    echo input_tsdf $input_tsdf
    echo detected_obb_file $detected_obb_file
    sleep 1
    batch_output_root=$run_root/batch-detect-$startimg-$endimg-$category
    if [ ! -d $batch_output_root ]; then
        mkdir $batch_output_root
    fi
    . ./run-train-detector.sh 
    # trained_svm_path=$trained_svm_path".trained_svm_model.svm"
    # trained_svm_path=$trained_svm_path".trained_svm_model.svm"

    echo "################## run detection ###################"
    for test_seq_i in ${test_seq_idx[$seq_i]}
    do
        echo "test seq index: " $test_seq_i
        test_startimg=${startimg_arr[$test_seq_i]}
        test_endimg=${endimg_arr[$test_seq_i]}
        test_input_tsdf=${input_tsdf_arr[$test_seq_i]}
        test_detected_obb_file=${detected_obb_file_arr[$test_seq_i $ci]}

        echo test_startimg $test_startimg
        echo test_endimg $test_endimg
        echo category $category
        echo test_input_tsdf $test_input_tsdf
        echo test_detected_obb_file $test_detected_obb_file
    #    sleep 1
        echo "run detection"
        detect_output_root=$batch_output_root/detect-res5-$test_startimg-$test_endimg/
        if [ ! -d $detect_output_root ]; then
            mkdir $detect_output_root
        fi
        . ./run-sliding-window-detect.sh 
        # detect_res_path=$detect_output_prefix"_SlidingBoxDetectionResults_Parallel_Final.txt"

        echo "compute pr curve"
    #    sleep 1
        pr_output_root=$detect_output_root
        . ./compute_pr_curve.sh
        # nms_res=$pr_curve_output_prefix"NMS_res.txt"
        # detect_obb_infos=$pr_curve_output_dir"_obb_infos.txt"
        cur_detect_obb_infos[$test_seq_i]=$detect_obb_infos
    done

    echo "################## merge tsdf/obbs ###################"
    merge_output_dir=$batch_output_root/merge_tsdf_obb_output-trainingseq-$seq_i
    if [ ! -d $merge_output_dir ]; then
        mkdir $merge_output_dir
   # else
   #     rm -r $merge_output_dir
   #     mkdir $merge_output_dir
    fi
    merge_output_prefix=$merge_output_dir/merged_model
    model_fileoption="--in-models "$(IFS=$' '; echo "${input_tsdf_arr[*]}")
    if [ $do_merge_tsdf_obb -gt 0 ]; then
        echo $merge_bin  $model_fileoption --in-obbs $(IFS=$' '; echo "${cur_detect_obb_infos[*]}") --out $merge_output_prefix --obb-min-score 0
        $merge_bin  $model_fileoption --in-obbs $(IFS=$' '; echo "${cur_detect_obb_infos[*]}") --out $merge_output_prefix --obb-min-score 0
    fi
    merged_model=$merge_output_prefix"_tsdf.bin"
    merged_detect_box_txt=$merge_output_prefix".obb_infos.txt"
    
    echo "################## joint learn ###################"
    joint_learn_output_root=$batch_output_root/joint_learn-25building-clean-$seq_i
    if [ ! -d $joint_learn_output_root ]; then
        mkdir $joint_learn_output_root
   # else
   #     rm -r $merge_output_dir
   #     mkdir $merge_output_dir
    fi
    
    . ./joint_cluster_model.sh

done
done
