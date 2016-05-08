#!/bin/bash
. ./init.sh

adjust_bin=$bin_dir/test_adjust_detection

output_dir=/home/dell/results_5/seperate-seq-batch-all-detector-try3-cameraready-noscale-iccvres3-interleave-align-pca20-allsamples/batch-res2-1890-2090-car-svmw1-10-svmc-100/detect-test-res-fullseq-1890-2090/adjusted_obbs_test
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

in_model=/home/dell/results_5/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-1890-ed-2090-vlen-0.2-rampsz-6-try1/recon-1890-2090-vlen-0.2-rampsz-6_tsdf.bin
detected_obb_file=/home/dell/results_5/seperate-seq-batch-all-detector-try3-cameraready-noscale-iccvres3-interleave-align-pca20-allsamples/batch-res2-1890-2090-car-svmw1-10-svmc-100/detect-test-res-fullseq-1890-2090/svm_detect_pr2-detect-try1-voxelsides-9-9-6-dx-0.4-dy-0.4-dr-5-jitter-20/sample_obb_infos.txt
in_score_file=/home/dell/results_5/seperate-seq-batch-all-detector-try3-cameraready-noscale-iccvres3-interleave-align-pca20-allsamples/batch-res2-1890-2090-car-svmw1-10-svmc-100/detect-test-res-fullseq-1890-2090/svm_detect_pr2-detect-try1-voxelsides-9-9-6-dx-0.4-dy-0.4-dr-5-jitter-20/sample_obb_scores.txt
sample_voxel_sidelengths="9 9 6"
svm_model=/home/dell/results_5/seperate-seq-batch-all-detector-try3-cameraready-noscale-iccvres3-interleave-align-pca20-allsamples/batch-res2-1890-2090-car-svmw1-10-svmc-100/train-svm-voxelsides-9-9-6-dx-0.2-dy-0.2-dr-2.5-jitter-20-svmw1-10-svmc-100/sample.trained_svm_model.svm
out_dir_prefix=$output_dir/adjusted_obbs_test.txt

$adjust_bin --in-model $in_model --detected-obb-file $detected_obb_file --detected-obb-score-file $in_score_file --sample_voxel_sidelengths $sample_voxel_sidelengths --out-dir-prefix $out_dir_prefix --svm_model $svm_model

adjusted_obb_txt=$out_dir_prefix"_obb_infos.txt"
