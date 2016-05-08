#!/bin/bash
# the binary file of reconstruction program
test_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/test_sliding_window_object_detector

# input_model
#input_model=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/output-3d-model-semantic/ply-0-600-0.5-newtest-conf-flatten-1-noclean_bin_tsdf_file.bin
input_model=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/output-3d-model-semantic2/ply-0-600-0.5-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-2.0_bin_tsdf_file.bin

# input_sample
input_samples=/home/dell/Data/results/house-sliced-res-1/h-joint-align_tsdf_sliced_0.bin" "/home/dell/Data/results/house-sliced-res-1/h-joint-align_tsdf_sliced_1.bin" "/home/dell/Data/results/house-sliced-res-1/h-joint-align_tsdf_sliced_2.bin

# input svm model
#input_svm=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/tsdf_feature_test_600frame_debug_hardnegmining_train_svm_1_dx-1_dy-2_dr-15_2/house_1_trained_svm_model.svm
#input_svm=/home/dell/link_to_results/tsdf_feature_test_600frame_debug_hardnegmining_whether_bug_version_same_as_before/house_1_trained_svm_model.svm
#input_svm=/home/dell/link_to_results/tsdf_feature_test_600frame_debug_hardnegmining_linear/house_1_trained_svm_model.svm
input_svm=/home/dell/link_to_results/tsdf_feature_test_600frame_debug_hardnegmining_linear_bug/house_1_trained_svm_model.svm

#input_svm=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/tsdf_feature_test_600frame_debug_hardnegmining_train_svm_3_dx-5_dy-5_dr-30_2/house_1_trained_svm_model.svm

# output_prefix
#set -e
for vlen in 2
do
output_prefix=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/tsdf_feature_test_600frame_debug_detect_svm_2_dx-0.5_dy-0.5_dr-15_linear_bug2
if [ ! -d "$output_prefix" ]; then
mkdir $output_prefix
fi
output_prefix=$output_prefix"/house_1"
echo "$test_bin $input_samples --in-model $input_model --out-dir-prefix $output_prefix --save_tsdf_bin  --mesh-min-weight 0.2 --voxel_length $vlen --svm_model $input_svm --delta_x 0.5 --delta_y 0.5 --rotate_degree 15 --total_thread 8"

$test_bin $input_samples --in-model $input_model --out-dir-prefix $output_prefix --save_tsdf_bin  --mesh-min-weight 0.0 --voxel_length $vlen --svm_model $input_svm --delta_x 0.5 --delta_y 0.5 --rotate_degree 15 --total_thread 8 
done
