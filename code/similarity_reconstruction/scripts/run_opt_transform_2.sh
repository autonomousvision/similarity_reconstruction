#!/bin/bash
build_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap
test_tsdf_align_bin=$build_root/bin/test_opt_transform

#input_model=/home/dell/link_to_results/output-3d-model-camera3-test-withcam2-no-satarate/ply-0-1000-1.2-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-1.6_bin_tsdf_file.bin
#input_model=/home/dell/link_to_results/output-newdata1-satur-conf-cam23-new1/ply-642-1043-0.1-newtest-conf-flatten-0-noclean-voxellen-0.2-posmaxdist-1.6-negmaxdist--.8_bin_tsdf_file.bin
input_model=/home/dell/results_3/reconstruction_closest_test-5-29/vri-fusing-result-s3_cam_3_with_2-st-3320-ed-3530-vlen-0.2-rampsz-6-try1/recon-3320-3530-vlen-0.2-rampsz-6_tsdf.bin
#detect_box_txt=/home/dell/Data/results/test_joint_cluster_model/test_boxes7.txt
#detect_box_txt=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_house_debug1.txt
#detect_box_txt=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_house.txt
detect_box_txt=/home/dell/results_2/seperate-seq-batch-all-seq0-detector-building2-car-try2/batch-res-3320-3530-building-svmw1-10-svmc-100/detect-test-res-fullseq-3320-3530/svm_detect_pr2-detect-try1-voxelsides-9-9-6-dx-1-dy-1-dr-2.5-jitter-10/sample_obb_infos.txt

set -e

output_root=/home/dell/results_5/
output_dir=$output_root/joint-opt-trans-newdata1-obs-0.1-1/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi

out_ply=$output_dir"/h-joint-opt.ply"
#echo $out_ply
echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number 0 --optimize-max-iter 3 --mesh_min_weight 0 --lambda-average-scale 0.0 --lambda-obs 0.0 --lambda-reg-rot 0.0 --lambda-reg-scale 0.0 --lambda-reg-trans 0.0
$test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number 0 --optimize-max-iter 3 --mesh_min_weight 0 --lambda-average-scale 0.0 --lambda-obs 0.1 --lambda-reg-rot 0 --lambda-reg-scale 0 --lambda-reg-trans 0 
