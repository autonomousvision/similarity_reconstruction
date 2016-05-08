#!/bin/bash
# the binary file of reconstruction program
test_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/compute_precision_recall_curve

# input_model
#input_model=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/output-3d-model-semantic/ply-0-600-0.5-newtest-conf-flatten-1-noclean_bin_tsdf_file.bin
#input_model=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/output-3d-model-semantic2/ply-0-600-0.5-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-2.0_bin_tsdf_file.bin

# input_sample
input_samples=/home/dell/Data/results/house-sliced-res-1/h-joint-align_tsdf_sliced_0.bin" "/home/dell/Data/results/house-sliced-res-1/h-joint-align_tsdf_sliced_1.bin" "/home/dell/Data/results/house-sliced-res-1/h-joint-align_tsdf_sliced_2.bin

#detect_output_file=/home/dell/link_to_results/tsdf_feature_test_600frame_debug_detect_svm_1_dx-0.5_dy-0.5_dr-15_2/house_1_SlidingBoxDetectionResults.txt
#detect_output_file=/home/dell/link_to_results/tsdf_feature_test_600frame_debug_detect_svm_2_dx-0.5_dy-0.5_dr-15_vlen_thresh_-0.5-5_5_30_debug2_parallel_thread_8_NN_res/house_1_SlidingBoxDetectionResults_Parallel_Final.txt
#detect_output_file=/home/dell/link_to_results/tsdf_feature_test_600frame_debug_detect_svm_2_dx-0.5_dy-0.5_dr-15_vlen_thresh_-0.5-5_5_30_debug2_parallel_thread_8_NN_retrained_with_simplehardneg_2/house_1_SlidingBoxDetectionResults_Parallel_Final.txt
#detect_output_file=/home/dell/link_to_results/tsdf_feature_test_600frame_debug_detect_svm_2_dx-0.5_dy-0.5_dr-15_vlen_thresh_-0.5-5_5_30_debug2_parallel_thread_8_NN_retrained_bug_test/house_1_SlidingBoxDetectionResults_Parallel_Final.txt
#detect_output_file=/home/dell/link_to_results/tsdf_feature_test_600frame_debug_detect_svm_2_dx-0.5_dy-0.5_dr-15_linear_new/house_1_SlidingBoxDetectionResults_Parallel_Final.txt
detect_output_file=/home/dell/link_to_results/tsdf_feature_test_600frame_debug_detect_svm_2_dx-0.5_dy-0.5_dr-15_linear_bug2/house_1_SlidingBoxDetectionResults_Parallel_Final.txt
# input svm model
#input_svm=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/tsdf_feature_test_600frame_debug_hardnegmining_train_svm_1_dx-1_dy-2_dr-15_2/house_1_trained_svm_model.svm
#input_svm=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/tsdf_feature_test_600frame_debug_hardnegmining_train_svm_3_dx-5_dy-5_dr-30_2/house_1_trained_svm_model.svm

#input_NMS_file=/home/dell/link_to_results/NMS_result1/house_1NMS_res.txt
input_NMS_file=""

# output_prefix
#set -e
for vlen in 2
do
output_prefix=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/tsdf_feature_PRCurve_test_600frame_debug_detect_svm_2_dx-0.5_dy-0.5_dr-15_linear_bug2
if [ ! -d "$output_prefix" ]; then
mkdir $output_prefix
fi
output_prefix=$output_prefix"/house_1"
echo "$test_bin $input_samples --out-dir-prefix $output_prefix --save_tsdf_bin --mesh-min-weight 0.2 --voxel_length $vlen --detect_output_file $detect_output_file --input_nms_file $input_NMS_file"

#$test_bin $input_samples --out-dir-prefix $output_prefix --save_tsdf_bin --mesh-min-weight 0.2 --voxel_length $vlen --detect_output_file $detect_output_file --input_nms_file $input_NMS_file
$test_bin $input_samples --out-dir-prefix $output_prefix --save_tsdf_bin --mesh-min-weight 0.2 --voxel_length $vlen --detect_output_file $detect_output_file
done
