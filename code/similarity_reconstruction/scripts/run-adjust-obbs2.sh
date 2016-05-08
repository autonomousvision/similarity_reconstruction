#!/bin/bash
. ./init.sh

adjust_bin=$bin_dir/test_adjust_detection2
adjust_output_root=/home/dell/results_5/detection_results/seq-6050-6350/building
output_dir=$adjust_output_root/adjusted_obbs_score_distribution2
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

startimg=6050
endimg=6350
test_input_tsdf=$result_root/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-$startimg-ed-$endimg-vlen-0.2-rampsz-6-try1/recon-$startimg-$endimg-vlen-0.2-rampsz-6_tsdf.bin
#detect_obb_infos=/home/dell/results_5/detection_results/seq-6050-6350/building/merged_model.obb_infos_1.txt
detect_obb_infos=/home/dell/results_5/seperate-seq-batch-all-seq3-detector-building2-car-try3-cameraready-noscale-iccvres3-interleave-align-pca9/batch-res2-6050-6350-building-svmw1-10-svmc-100/detect-test-res-fullseq-6050-6350/adjusted_obbs/adjusted_obbs.txt_obb_infos.txt
svm_model=/home/dell/results_5/svm_models/6050_6350/sample.trained_svm_model.svm
sample_voxel_sidelengths="9 9 6"

in_model=$test_input_tsdf
detected_obb_file=$detect_obb_infos
#sample_voxel_sidelengths="$vx $vy $vz"
#svm_model=$trained_svm_path
out_dir_prefix=$output_dir/adjusted_obbs.txt
do_adjust_obbs=1

if [ $do_adjust_obbs -gt 0 ]; then
    echo $adjust_bin --in-model $in_model --detected-obb-file $detected_obb_file --sample_voxel_sidelengths $sample_voxel_sidelengths --out-dir-prefix $out_dir_prefix --svm_model $svm_model
    sleep 2
    $adjust_bin --in-model $in_model --detected-obb-file $detected_obb_file --sample_voxel_sidelengths $sample_voxel_sidelengths --out-dir-prefix $out_dir_prefix --svm_model $svm_model
fi

adjusted_obb_txt=$out_dir_prefix"_obb_infos.txt"
