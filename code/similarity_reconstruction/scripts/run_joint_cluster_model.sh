#!/bin/bash
set -e
test_tsdf_align_bin=$bin_dir/test_joint_cluster_model

input_model=/home/dell/link_to_results/output-3d-model-camera3-test-withcam2-no-satarate/ply-0-1000-1.2-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-1.6_bin_tsdf_file.bin
#input_model=$merged_model
#detect_box_txt=/home/dell/Data/results/test_joint_cluster_model/test_boxes_two_c2.txt
detect_box_txt=/home/dell/Data/results/test_joint_cluster_model/test_boxes9_old.txt
#out_prefix=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/joint-cluster-model-test1/test1.ply


for pca_num in 0
do

output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
output_dir=$output_root/joint-reassign-test9-old-do-noalign-pca-$pca_num-try6/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi

out_ply=$output_dir"/h-joint-opt.ply"
#echo $out_ply
#echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number 3 --optimize-max-iter 3 --mesh_min_weight 0.8 
echo "$test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter 20 --mesh_min_weight 0.8 "
$test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter 8 --mesh_min_weight 0.005

done
