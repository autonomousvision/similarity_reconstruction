#!/bin/bash
build_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap
data_dir=/home/dell/Data/results/house-sliced-res-1/
test_tsdf_align_bin=$build_root/bin/test_opt_transform

input_model=/home/dell/link_to_results/output-3d-model-camera3-test-withcam2-no-satarate/ply-0-1000-1.2-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-1.6_bin_tsdf_file.bin
detect_box_txt=/home/dell/Data/results/test_joint_cluster_model/test_boxes7.txt
#out_prefix=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/joint-cluster-model-test1/test1.ply

set -e

output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
output_dir=$output_root/joint-opt-trans-test7-txt-debug1/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi

out_ply=$output_dir"/h-joint-opt.ply"
#echo $out_ply
#echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number 0 --optimize-max-iter 3 --mesh_min_weight 1.2
#$test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number 0 --optimize-max-iter 3 --mesh_min_weight 1.2

echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number 0 --optimize-max-iter 3 --mesh_min_weight 0 --lambda-average-scale 0.0 --lambda-obs 0.0 --lambda-reg-rot 0.0 --lambda-reg-scale 0.0 --lambda-reg-trans 0.0
$test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number 0 --optimize-max-iter 3 --mesh_min_weight 0 --lambda-average-scale 0.0 --lambda-obs 0.0 --lambda-reg-rot 0.0 --lambda-reg-scale 0.0 --lambda-reg-trans 0.0
