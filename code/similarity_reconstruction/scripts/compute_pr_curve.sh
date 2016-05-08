#!/bin/bash
pr_compute_bin=$bin_dir/compute_precision_recall_curve
### 3. compute pr curve
echo "#################### compute pr curve #####################"
pr_curve_output_dir=$pr_output_root/svm_detect_pr2-$detect_output_suffix/
if [ ! -d "$pr_curve_output_dir" ]; then
    mkdir $pr_curve_output_dir
fi
pr_curve_output_prefix=$pr_curve_output_dir"/sample"
nms_res=$pr_curve_output_prefix"NMS_res.txt"

if [ $do_NMS -gt 0 ]; then
    nms_option=""
else
    nms_option="--input_nms_file $nms_res"
fi
# sleep 2
if [ $do_pr -gt 0 ]; then
echo $pr_compute_bin --out-dir-prefix $pr_curve_output_prefix --mesh-min-weight $mesh_min_weight --detect_output_file $detect_res_path --sample_voxel_sidelengths $vx $vy $vz --detected-obb-file $test_detected_obb_file $nms_option --in-model $test_input_tsdf --lowest_score_to_supress $lowest_score_to_supress --delta_x $detect_delta_x --delta_y $detect_delta_y --rotate_degree $detect_delta_rot
sleep 3
$pr_compute_bin --out-dir-prefix $pr_curve_output_prefix --mesh-min-weight $mesh_min_weight --detect_output_file $detect_res_path --sample_voxel_sidelengths $vx $vy $vz --detected-obb-file $test_detected_obb_file $nms_option --in-model $test_input_tsdf --lowest_score_to_supress $lowest_score_to_supress --delta_x $detect_delta_x --delta_y $detect_delta_y --rotate_degree $detect_delta_rot
fi
detect_obb_infos=$pr_curve_output_prefix"_obb_infos.txt"
detect_obb_scores=$pr_curve_output_prefix"_obb_scores.txt"
