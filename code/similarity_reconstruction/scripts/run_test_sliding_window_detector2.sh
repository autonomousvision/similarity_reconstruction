#!/bin/bash
# the binary file of reconstruction program
test_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/test_sliding_window_object_detector2

# input_model
#input_model=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/output-3d-model-semantic/ply-0-600-0.5-newtest-conf-flatten-1-noclean_bin_tsdf_file.bin
#input_model=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/output-3d-model-semantic2/ply-0-600-0.5-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-2.0_bin_tsdf_file.bin
input_model=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/output-debug-newdata5/ply-1890-2090-0-newtest-noconf-flatten-0-noclean-voxellen-0.2-posmaxdist-1.6-negmaxdist--.8_bin_tsdf_file.bin

# input_sample
detected_obb_file=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_building2.txt
vx=9
vy=9
vz=6

# input svm model
#input_svm=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/tsdf_feature_test_600frame_debug_hardnegmining_train_svm_1_dx-1_dy-2_dr-15_2/house_1_trained_svm_model.svm
#input_svm=/home/dell/link_to_results/tsdf_feature_test_600frame_debug_hardnegmining_whether_bug_version_same_as_before/house_1_trained_svm_model.svm
#input_svm=/home/dell/link_to_results/tsdf_feature_test_600frame_debug_hardnegmining_linear/house_1_trained_svm_model.svm
#input_svm=/home/dell/link_to_results/tsdf_train_svm_test_1890_2090/house_newtest2_trained_svm_model.svm
input_svm=/home/dell/link_to_results/tsdf_train_svm_test_1890_2090/house_newtest2_trained_svm_model.svm
trynum=15
input_svm=/home/dell/link_to_results/tsdf_train_svm_test_newmerge_1890_2090_nonnoisy_recon_test-$trynum-nojitter-vxidelen-$vx-$vy-$vz/house_newtsdf1_trained_svm_model.svm

#input_svm=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/tsdf_feature_test_600frame_debug_hardnegmining_train_svm_3_dx-5_dy-5_dr-30_2/house_1_trained_svm_model.svm

# output_prefix
set -e
output_prefix=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/svm_test_newmerge_t1_debug_detect_$trynum-dx-0.5_dy-0.5_dr-15_linear_nonnoisy_nojitter
if [ ! -d "$output_prefix" ]; then
mkdir $output_prefix
fi
output_prefix=$output_prefix"/house_1"

echo $test_bin --in-model $input_model --out-dir-prefix $output_prefix --detected-obb-file $detected_obb_file --save_tsdf_bin  --mesh-min-weight 0.0 --svm_model $input_svm --delta_x 0.5 --delta_y 0.5 --rotate_degree 7.5 --total_thread 1 --sample_voxel_sidelengths $vx $vy $vz
$test_bin --in-model $input_model --out-dir-prefix $output_prefix --detected-obb-file $detected_obb_file --save_tsdf_bin  --mesh-min-weight 0.0 --svm_model $input_svm --delta_x 0.5 --delta_y 0.5 --rotate_degree 7.5 --total_thread 8 --sample_voxel_sidelengths $vx $vy $vz
