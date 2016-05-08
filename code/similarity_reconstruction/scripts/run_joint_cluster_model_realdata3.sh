#!/bin/bash
set -e
build_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap
test_tsdf_align_bin=$build_root/bin/test_joint_cluster_model

input_model=/home/dell/link_to_results/output-debug-newdata5/ply-1890-2090-0-newtest-noconf-flatten-0-noclean-voxellen-0.2-posmaxdist-1.6-negmaxdist--.8_bin_tsdf_file.bin
#detect_box_txt=/home/dell/upload2/4-6/detection_new_1890_2090_cam2/house_1_obb_infos_highest_four.txt
detect_box_txt=/home/dell/upload2/4-6/script_files_for_align/001822_002032_house_new.txt

lambdaobs=0.0
pca_num=0
uniform_weight="--uniform-weight"
#uniform_weight=""

output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
output_dir=$output_root/building-align-3house-avgscale-0-lambdaobs-$lambdaobs-lambdarg-0$uniform_weight/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi
out_ply=$output_dir"/h-joint-opt.ply"

echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter 1 --mesh_min_weight 0.00 --lambda-average-scale 0.0 --lambda-obs $lambdaobs --lambda-reg-rot 0.0 --lambda-reg-scale 0.0 --lambda-reg-trans 0.0 $uniform_weight
$test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter 1 --mesh_min_weight 0.00 --lambda-average-scale 0.0 --lambda-obs $lambdaobs --lambda-reg-rot 0.0 --lambda-reg-scale 0.0 --lambda-reg-trans 0.0 $uniform_weight
