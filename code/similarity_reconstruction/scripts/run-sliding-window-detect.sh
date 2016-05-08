#!/bin/bash
detection_bin=$bin_dir/test_sliding_window_object_detector2
detect_output_dir=$detect_output_root/svm_detect_res-$detect_output_suffix/
if [ ! -d "$detect_output_dir" ]; then
    mkdir $detect_output_dir
    echo "mkdir " $detect_output_dir
#else
   # rm -r $detect_output_dir
   # mkdir $detect_output_dir
fi
detect_output_prefix=$detect_output_dir"/detect_sample"
### 2. test svm
echo "#################### test svm #####################"
input_svm_path=$trained_svm_path
#test_input_tsdf=/home/dell/results_5/reconstruction_closest_calibseq1/vri-fusing-result-s3_cam_3_with_2-st-800-ed-1300-vlen-0.2-rampsz-6-try1/recon-800-1300-vlen-0.2-rampsz-6_tsdf.bin
if [ $do_detection -gt 0 ]; then
echo $detection_bin --in-model $test_input_tsdf --out-dir-prefix $detect_output_prefix --mesh-min-weight $mesh_min_weight --svm_model $input_svm_path --delta_x $detect_delta_x --delta_y $detect_delta_y --rotate_degree $detect_delta_rot --total_thread $total_thread  --sample_voxel_sidelengths $vx $vy $vz
$detection_bin --in-model $test_input_tsdf --out-dir-prefix $detect_output_prefix --mesh-min-weight $mesh_min_weight --svm_model $input_svm_path --delta_x $detect_delta_x --delta_y $detect_delta_y --rotate_degree $detect_delta_rot --total_thread $total_thread  --sample_voxel_sidelengths $vx $vy $vz
fi
detect_res_path=$detect_output_prefix"_SlidingBoxDetectionResults_Parallel_Final.txt"
template_obb_path=$detect_output_prefix"_training_template.txt"

