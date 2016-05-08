#!/bin/bash
build_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap
data_dir=/home/dell/Data/results/house-sliced-res-1/
test_tsdf_align_bin=$build_root/bin/test_joint_cluster_model

input_model=/home/dell/link_to_results/output-debug-newdata3/ply-1890-2090-0.005-newtest-noconf-flatten-0-noclean-voxellen-0.2-posmaxdist-1.6-negmaxdist--.8_bin_tsdf_file.bin
#detect_box_txt=/home/dell/Data/results/test_joint_cluster_model/test_boxes_two_c2.txt
#detect_box_txt=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_house.txt
detect_box_txt=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_house.txt
#detect_box_txt=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_car.txt
#out_prefix=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/joint-cluster-model-test1/test1.ply

set -e

#uniform_weight="--uniform-weight"
uniform_weight=""
for pca_num in 0
do
lambdaobs=0.0

output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
output_dir=$output_root/joint3-reassign-newdata2-debug1-building2-avg-scale-0-lambdaobs-$lambdaobs-lambdareg-0-large-ext-newobsw-withweight-$uniform_weight/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi

out_ply=$output_dir"/h-joint-opt.ply"
#echo $out_ply
#echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number 3 --optimize-max-iter 3 --mesh_min_weight 0.8 
#echo "$test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter 20 --mesh_min_weight 0.8 "
#echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter 8 --mesh_min_weight 0.005 
#echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter 1 --mesh_min_weight 0.00 --lambda-average-scale 0.0 --lambda-obs 0.0 --lambda-reg-rot 0.0 --lambda-reg-scale 0.0 --lambda-reg-trans 0.0
echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter 1 --mesh_min_weight 0.00 --lambda-average-scale 0.0 --lambda-obs $lambdaobs --lambda-reg-rot 0.0 --lambda-reg-scale 0.0 --lambda-reg-trans 0.0 $uniform_weight
$test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter 1 --mesh_min_weight 0.00 --lambda-average-scale 0.0 --lambda-obs $lambdaobs --lambda-reg-rot 0.0 --lambda-reg-scale 0.0 --lambda-reg-trans 0.0 $uniform_weight

done
