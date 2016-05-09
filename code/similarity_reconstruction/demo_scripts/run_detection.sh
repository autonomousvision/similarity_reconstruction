#!/bin/bash
set -e
# initialize relavant folders
# . ./init_paths.sh

# binary file paths
train_bin=$bin_dir/train_detectors_main
detect_bin=$bin_dir/detect_main

# output root for training & detection
run_root=$result_root/run_detection
if [ ! -d $run_root ]; then
    mkdir $run_root
fi

# parameters
# step size used in training: step_x = 1m, step_y = 1m, step_rotation_angle = 1 degree
# if not specified, the default is (2, 2, 2) for large objects (min_sidelength > 8m) and (1, 1, 1) for small objects
detect_deltas=(1 1 1)
# step size used in testing: step_x, step_y, step_rotation_angle (in degree).
test_detect_deltas=(1 1 1)
# sampling size: along x, y, z axis
detect_sample_size=(9 9 6)
# thread number for parallel detection
total_thread=10
# minimum score to keep the detection
# For buildings (category 1 & 2), use a lower threshold to preserve more detected samples
# For cars (category 3), use a higher threshold to reject false positives 
# because the reconstruction is noisy at the scale of cars
min_score_to_keep=(-0.5 -0.5 0.1)

# tsdf model and anntations for training
train_scene_file=$detector_train_data_dir/scene_tsdf.bin
annotations=$detector_train_data_dir/annotated_obbs.txt

# output directory for training result
output_prefix=$run_root

# tsdf model for detection
detection_scene_file=$scene_model_bin


# training
if [ $run_train -gt 0 ]; then
    # display the data for training
    if [[ $display && $display -gt 0 ]]; then 
        for i in 0 1 2
        do
            echo "show training samples for category $i"
            cur_vis_txt=$detector_train_data_dir/visualization/category_$i/visualization.txt
            ./visualization.sh "training_data_category_$i" "$cur_vis_txt"
        done
    fi
    # echo $train_bin --scene_file $train_scene_file --annotations $annotations --output_prefix $output_prefix --sample_size ${detect_sample_size[@]} --total_thread $total_thread --svm_param_c 100 --svm_param_w1 10 --detect_deltas ${detect_deltas[@]}
    $train_bin --scene_file $train_scene_file --annotations $annotations --output_prefix $output_prefix --sample_size ${detect_sample_size[@]} --total_thread $total_thread --svm_param_c 100 --svm_param_w1 10 --detect_deltas ${detect_deltas[@]}
    # dir storing trained models
    detector_file_dir=$output_prefix
fi

if [ $run_detect -gt 0 ]; then
    # echo $detect_bin --scene_file $detection_scene_file --detector_file $detector_file_dir --output_prefix $output_prefix --total_thread $total_thread --min_score_to_keep ${min_score_to_keep[@]} --detect_deltas ${test_detect_deltas[@]}
    $detect_bin --scene_file $detection_scene_file --detector_file $detector_file_dir --output_prefix $output_prefix --total_thread $total_thread --min_score_to_keep ${min_score_to_keep[@]} #--detect_deltas ${test_detect_deltas[@]}
    if [[ $display && $display -gt 0 ]]; then 
        ./visualization.sh "detect_results" "$visualization_txt"
    fi
fi

detect_res_txt=$output_prefix/detect_res_all_obb_nmsed.txt
visualization_txt=$output_prefix/visualization/visualization.txt

