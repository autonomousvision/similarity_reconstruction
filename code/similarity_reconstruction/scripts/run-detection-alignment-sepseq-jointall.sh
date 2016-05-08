u!/bin/bash
# the binary file of reconstruction program
set -e

. ./init.sh

merge_bin=$bin_dir/merge_tsdfs_obbs

#run_root=$result_root/seperate-seq-batch-seq-6050-6350-cropped-train-test2/
run_root=$result_root/seperate-seq-batch-all-seq0-detector-debug7/
#run_root=$result_root/seperate-seq-batch-all-seq0-detector-prevdetect/
if [ ! -d $run_root ]; then
    mkdir $run_root
fi

vx_arr=(9 6)
vy_arr=(9 6)
vz_arr=(6 5)

train_detect_delta_x_arr=(1 0.2)
train_detect_delta_y_arr=(1 0.2)
train_detect_delta_rot_arr=(2.5 2.5) # degree 

#detect_delta_x_arr=(0.5 0.4)
#detect_delta_y_arr=(0.5 0.4)
#detect_delta_rot_arr=(5 5)

detect_delta_x_arr=(1 0.4)
detect_delta_y_arr=(1 0.4)
detect_delta_rot_arr=(2.5 5)

#detect_delta_x_arr=(10 10)
#detect_delta_y_arr=(10 10)
#detect_delta_rot_arr=(30 30)

mesh_min_weight_arr=(0.0 0.0)
total_thread_arr=(10 10)
jitter_num_arr=(10 20)
sample_num_arr=(2000 2000)
lowest_score_to_supress_arr=(-0.5 0)
noise_counter_thresh_arr=(3 3)
noise_comp_thresh_arr=(-1 -1)

category_arr=(building2 car)
startimg_arr=(1470 1890 3320 6050)
endimg_arr=(1790 2090 3530 6350)

merge_score_thresh_arr=(0.01 1)
do_train_detector_arr=(0 0)
do_detection_arr=(0 0)
do_NMS_arr=(0 0)
do_pr_arr=(0 0)
do_adjust_obbs_arr=(0 1)
do_merge_tsdf_obb_arr=(0 1)
do_init_consistent_check_arr=(0 0)
do_final_merge=0
do_final_joint_carbuildinglearn=0
do_joint_learn_arr=(1 1 $do_final_joint_carbuildinglearn)
do_consistency_check=1

svm_w1_arr=(10 10)
svm_cs_arr=(100 100)


declare -A detected_obb_file_arr
declare -A train_detected_obb_file_arr
declare -A train_input_tsdf_arr

declare -A merged_model_arr
declare -A merged_detect_box_txt_arr

for seq_i in 0 1 2 3
do
    startimg=${startimg_arr[$seq_i]}
    endimg=${endimg_arr[$seq_i]}
    input_tsdf_arr[$seq_i]=$result_root/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-$startimg-ed-$endimg-vlen-0.2-rampsz-6-try1/recon-$startimg-$endimg-vlen-0.2-rampsz-6_tsdf.bin

    for ci in 0 1
    do
        category=${category_arr[$ci]}
        detected_obb_file_arr[$seq_i $ci]=/home/dell/Data/download_labels/label_4_12/gt_4_12_$startimg-$endimg-$category/gt_$startimg-$endimg-$category".txt"
        train_input_tsdf_arr[$seq_i $ci]=$result_root/cropped_tsdf_for_training/gt_cropped_$startimg-$endimg-$category/gt_cropped_$startimg-$endimg-$category".cropped_tsdf_tsdf.bin"
        train_detected_obb_file_arr[$seq_i $ci]=$result_root/cropped_tsdf_for_training/gt_cropped_$startimg-$endimg-$category/gt_cropped_$startimg-$endimg-$category".obb_info.txt"
        #echo ${detected_obb_file_arr[$seq_i $ci]}
    done
done

test_seq_idx[0]="0 1 2 3"
test_seq_idx[1]="1 0 2 3"
test_seq_idx[2]="2 0 1 3"
test_seq_idx[3]="3 0 1 2"

#svm_w1s=(1)
#svm_cs=(10)

#for svm_w1 in ${svm_w1s[@]}; do
#    for svm_c in ${svm_cs[@]}; do

for ci in 0 #1 #1
do
   ## if [ $svm_w1 -eq 1 && $svm_c -eq 0.1 ]; then continue; fi
   ## if [ $svm_w1 -eq 1 && $svm_c -eq 1 ]; then continue; fi
for seq_i in 1 #2 3 #1 2 3
do
    do_train_detector=${do_train_detector_arr[$ci]}
    do_detection=${do_detection_arr[$ci]}
    do_NMS=${do_NMS_arr[$ci]}
    do_pr=${do_pr_arr[$ci]}
    do_merge_tsdf_obb=${do_merge_tsdf_obb_arr[$ci]}
    do_joint_learn=${do_joint_learn_arr[$ci]}

    svm_w1=${svm_w1_arr[$ci]}
    svm_c=${svm_cs_arr[$ci]}

    vx=${vx_arr[$ci]}
    vy=${vy_arr[$ci]}
    vz=${vz_arr[$ci]}

    train_detect_delta_x=${train_detect_delta_x_arr[$ci]}
    train_detect_delta_y=${train_detect_delta_y_arr[$ci]}
    train_detect_delta_rot=${train_detect_delta_rot_arr[$ci]} # degree

    detect_delta_x=${detect_delta_x_arr[$ci]}
    detect_delta_y=${detect_delta_y_arr[$ci]}
    detect_delta_rot=${detect_delta_rot_arr[$ci]}

    mesh_min_weight=${mesh_min_weight_arr[$ci]}
    total_thread=${total_thread_arr[$ci]}
    jitter_num=${jitter_num_arr[$ci]}
    sample_num=${sample_num_arr[$ci]}
    lowest_score_to_supress=${lowest_score_to_supress_arr[$ci]}
    noise_counter_thresh=${noise_comp_thresh_arr[$ci]}
    noise_comp_thresh=${noise_comp_thresh_arr[$ci]}
    merge_score_thresh=${merge_score_thresh_arr[$ci]}

    detect_output_suffix="detect-try1-voxelsides-$vx-$vy-$vz-dx-$detect_delta_x-dy-$detect_delta_y-dr-$detect_delta_rot-jitter-$jitter_num"
    output_suffix="try1-voxelsides-$vx-$vy-$vz-dx-$train_delta_x-dy-$train_delta_y-dr-$train_delta_rot-jitter-$jitter_num"
    echo detect_output_suffix $detect_output_suffix
    echo output_suffix $output_suffix
    startimg=${startimg_arr[$seq_i]}
    endimg=${endimg_arr[$seq_i]}
    category=${category_arr[$ci]}
    input_tsdf=${input_tsdf_arr[$seq_i]}
    detected_obb_file=${detected_obb_file_arr[$seq_i $ci]}
    train_input_tsdf=${train_input_tsdf_arr[$seq_i $ci]}
    train_detected_obb_file=${train_detected_obb_file_arr[$seq_i $ci]}
    batch_output_root=$run_root/batch-res-$startimg-$endimg-$category-svmw1-$svm_w1-svmc-$svm_c
    if [ ! -d $batch_output_root ]; then
        mkdir $batch_output_root
    fi

    echo startimg $startimg
    echo endimg $endimg
    echo category $category
    echo input_tsdf: $input_tsdf
    echo detected_obb_file: $detected_obb_file
    echo train_input_tsdf: $train_input_tsdf
    echo train_detected_obb_file: $train_detected_obb_file
    echo batch_output_root $batch_output_root
    echo svm_c $svm_c
    echo svm_w1 $svm_w1
    sleep 1

    echo "################## run training ###################"
    # in: input_tsdf, detected_obb_file
    tmp_detected_obb_file=$detected_obb_file
    tmp_input_tsdf=$input_tsdf
    input_tsdf=$train_input_tsdf
    detected_obb_file=$train_detected_obb_file
    . ./run-train-detector.sh 
    detected_obb_file=$tmp_detected_obb_file
    input_tsdf=$tmp_input_tsdf
    # trained_svm_path=$trained_svm_path".trained_svm_model.svm"

    echo "################## run detection ###################"
##    #test_seq_i=$(echo $seq_i+1 | bc -l)
#    test_seq_i=$seq_i
#    #for test_seq_i in ${test_seq_idx[$seq_i]}
#    #do
#    echo "test seq index: " $test_seq_i
#    test_startimg=${startimg_arr[$test_seq_i]}
#    test_endimg=${endimg_arr[$test_seq_i]}
#
#    #test_input_tsdf=${input_tsdf_arr[$test_seq_i]}
#    #test_detected_obb_file=${detected_obb_file_arr[$test_seq_i $ci]}
#    test_input_tsdf=${train_input_tsdf_arr[$test_seq_i $ci]}
#    test_detected_obb_file=${train_detected_obb_file_arr[$test_seq_i $ci]}
#    detect_output_root=$batch_output_root/detect-test-res-smallseq-$test_startimg-$test_endimg/
#    if [ ! -d $detect_output_root ]; then
#        mkdir $detect_output_root
#    fi
#    echo test_startimg $test_startimg
#    echo test_endimg $test_endimg
#    echo category $category
#    echo test_input_tsdf $test_input_tsdf
#    echo test_detected_obb_file $test_detected_obb_file
#    echo detect_output_root $detect_output_root
#    sleep 2
#    echo "run detection"
#    . ./run-sliding-window-detect.sh 
#    # detect_res_path=$detect_output_prefix"_SlidingBoxDetectionResults_Parallel_Final.txt"
#
#    echo "compute pr curve"
#    sleep 1
#    pr_output_root=$detect_output_root
#    . ./compute_pr_curve.sh
#    # nms_res=$pr_curve_output_prefix"NMS_res.txt"
#    # detect_obb_infos=$pr_curve_output_dir"_obb_infos.txt"
#    cur_detect_obb_infos[$test_seq_i]=$detect_obb_infos
#    #done
##
    ##############################################
    test_seq_i=$seq_i
    test_startimg=${startimg_arr[$test_seq_i]}
    test_endimg=${endimg_arr[$test_seq_i]}
    test_input_tsdf=${input_tsdf_arr[$test_seq_i]}
    test_detected_obb_file=${detected_obb_file_arr[$test_seq_i $ci]}
    detect_output_root=$batch_output_root/detect-test-res-fullseq-$test_startimg-$test_endimg/
    if [ ! -d $detect_output_root ]; then
        mkdir $detect_output_root
    fi
    echo test_startimg $test_startimg
    echo test_endimg $test_endimg
    echo category $category
    echo test_input_tsdf $test_input_tsdf
    echo test_detected_obb_file $test_detected_obb_file
    echo detect_output_root $detect_output_root
    sleep 1
    echo "run detection"
    . ./run-sliding-window-detect.sh 
    # detect_res_path=$detect_output_prefix"_SlidingBoxDetectionResults_Parallel_Final.txt"
    echo detect_res_path $detect_res_path
    echo template_obb_path $template_obb_path

    echo "compute pr curve"
    sleep 1
    pr_output_root=$detect_output_root
    . ./compute_pr_curve.sh
    # nms_res=$pr_curve_output_prefix"NMS_res.txt"
    # detect_obb_infos=$pr_curve_output_dir"_obb_infos.txt"
    cur_detect_obb_infos[$test_seq_i]=$detect_obb_infos
    #done

    echo "######################## adjust detected bbs ################"
    do_adjust_obbs=${do_adjust_obbs_arr[$ci]}
    adjust_output_root=$detect_output_root
    . ./run-adjust-obbs.sh
    echo adjusted_obb_txt $adjusted_obb_txt
    detect_obb_infos=$adjusted_obb_txt

    ##############################################
    echo "################## merge tsdf/obbs ###################"
    merge_output_dir=$batch_output_root/merge_tsdf_obb_output-trainingseq-$seq_i-merge_score_thresh-$merge_score_thresh
    if [ ! -d $merge_output_dir ]; then
        mkdir $merge_output_dir
    fi
    merge_output_prefix=$merge_output_dir/merged_model
    #model_fileoption="--in-models "$(IFS=$' '; echo "${input_tsdf_arr[*]}")
    model_fileoption=""
    if [ $do_merge_tsdf_obb -gt 0 ]; then
        echo $merge_bin --in-models $test_input_tsdf --in-obbs $detect_obb_infos --out $merge_output_prefix --obb-min-score $merge_score_thresh
        $merge_bin --in-models $test_input_tsdf --in-obbs $detect_obb_infos --out $merge_output_prefix --obb-min-score $merge_score_thresh
    fi
    merged_model=$merge_output_prefix"_tsdf.bin"
    merged_detect_box_txt=$merge_output_prefix".obb_infos.txt"
    
    merged_model_arr[$test_seq_i $ci]=$merged_model
    merged_detect_box_txt_arr[$test_seq_i $ci]=$merged_detect_box_txt


    echo "################## joint learn ###################"
    joint_learn_suffix=$seq_i-merge_score_thresh-$merge_score_thresh-noisecounter-float-$noise_counter_thresh-noisecompo-$noise_comp_thresh-iter-3
    joint_learn_output_root=$batch_output_root/joint_learn-try2-$joint_learn_suffix
    if [ ! -d $joint_learn_output_root ]; then
        mkdir $joint_learn_output_root
   # else
   #     rm -r $merge_output_dir
   #     mkdir $merge_output_dir
    fi

    echo "################## initial consistency check ###########"
    do_consistency_check=${do_init_consistent_check_arr[$ci]}
    echo $do_consistency_check
    if [ $do_consistency_check -gt 0 ]; then
        joint_output_tsdf=$input_tsdf
        depthmap_check=1
        consistency_check_root=$joint_learn_output_root/init_consistency_check-$joint_learn_suffix
        echo consistency_check_root $consistency_check_root
        if [ ! -d $consistency_check_root ]; then
            mkdir $consistency_check_root
        fi
        . ./run-sky-consistency-checking.sh
        merged_model=$consistent_tsdf_output
    fi

    . ./joint_cluster_model.sh

    echo "############### consistency #########################"
    if [ $do_consistency_check -gt 0 ]; then
        joint_output_tsdf=$joint_learn_outdir/"joint-opt._merged_tsdf_tsdf.bin"
        consistency_check_root=$joint_learn_output_root/consistency_check-$joint_learn_suffix
        echo consistency_check_root $consistency_check_root
        depthmap_check=${depth_check_arr[$ci]}
        if [ ! -d $consistency_check_root ]; then
            mkdir $consistency_check_root
        fi
        . ./run-sky-consistency-checking.sh
    fi
#   # for iternum in 0 1 2
#   # do
#   #     iter_name_pattern=$joint_learn_outdir/joint-opt_whole_iter_"$iternum"._raftermerge_2ndclean_100_iter_0_0000000000.2_tsdf.bin
#   #     joint_output_tsdf=$iter_name_pattern
#   #     consistency_check_root=$joint_learn_output_root/final_consistency_check-iter-$iternum/
#   #     echo consistency_check_root $consistency_check_root
#   #     if [ ! -d $consistency_check_root ]; then
#   #         mkdir $consistency_check_root
#/   #     fi
#   #     . ./run-sky-consistency-checking.sh
#   # done
done
done
#done
#done

#return
###################################################################
echo "final ###################################################################"
noise_counter_thresh=2
noise_comp_thresh=-1
merge_score_thresh=-1

final_seq=0
batch_output_root=$run_root/final-joint-res-seq-$final_seq-svmw1-$svm_w1-svmc-$svm_c
if [ ! -d $batch_output_root ]; then
    mkdir $batch_output_root
fi
input_tsdf=${input_tsdf_arr[$final_seq]}
multi_category_detect="${merged_detect_box_txt_arr[$test_seq_i 0]} ${merged_detect_box_txt_arr[$test_seq_i 1]}" 
multi_category_labels="0 1"
# output:
### do final joint optimization

merge_output_dir=$batch_output_root/final_merge_tsdf_obb_output-seq-$seq_i-merge_score_thresh-$merge_score_thresh
merge_output_prefix=$merge_output_dir/merged_model_mulcate
merged_model=$merge_output_prefix"_tsdf.bin"
merged_detect_box_txt=$merge_output_prefix".obb_infos.txt"
if [ $do_final_merge -gt 0 ]; then
    echo "################## final merge tsdf/obbs ###################"
    if [ ! -d $merge_output_dir ]; then
        mkdir $merge_output_dir
    fi
    #model_fileoption="--in-models "$(IFS=$' '; echo "${input_tsdf_arr[*]}")
    model_fileoption=""
    echo $merge_bin --in-models $input_tsdf --in-obbs $multi_category_detect --in-category $multi_category_labels --out $merge_output_prefix --obb-min-score $merge_score_thresh
    $merge_bin --in-models $input_tsdf --in-obbs $multi_category_detect --in-category $multi_category_labels --out $merge_output_prefix --obb-min-score $merge_score_thresh
fi

echo "############### final joint learn #########################"
joint_learn_output_root=$batch_output_root/final_joint_learn-try2-$final_seq-merge_score_thresh-$merge_score_thresh-noisecounter-float-$noise_counter_thresh-noisecompo-$noise_comp_thresh-neighbor1-bothextground0.2-pca-1-iter-3-sampledenoise-newtestparam
echo joint_learn_output_root $joint_learn_output_root
if [ ! -d $joint_learn_output_root ]; then
    mkdir $joint_learn_output_root
fi
do_joint_learn=${do_joint_learn_arr[2]}
. ./joint_cluster_model.sh

joint_output_tsdf=$joint_learn_outdir/"joint-opt._merged_tsdf_tsdf.bin"
echo join_learn_outdir $joint_learn_outdir

#iter_name_pattern=joint-opt_whole_iter_0._raftermerge_2ndclean_100_iter_0_0000000000.2_mesh.ply
echo "############### final consistency #########################"
if [ $do_consistency_check -gt 0 ]; then
    consistency_check_root=$joint_learn_output_root/final_consistency_check6/
    echo consistency_check_root $consistency_check_root
    if [ ! -d $consistency_check_root ]; then
        mkdir $consistency_check_root
    fi
    . ./run-sky-consistency-checking.sh
fi

for iternum in 0 1 2
do
    iter_name_pattern=$joint_learn_outdir/joint-opt_whole_iter_"$iternum"._raftermerge_2ndclean_100_iter_0_0000000000.2_tsdf.bin
    joint_output_tsdf=$iter_name_pattern
    consistency_check_root=$joint_learn_output_root/final_consistency_check-iter-$iternum/
    echo consistency_check_root $consistency_check_root
    if [ ! -d $consistency_check_root ]; then
        mkdir $consistency_check_root
    fi
    . ./run-sky-consistency-checking.sh
done
