#!/bin/bash
adjust_bin=$bin_dir/test_adjust_detection

output_dir=$adjust_output_root/adjusted_obbs
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

in_model=$test_input_tsdf
detected_obb_file=$detect_obb_infos
detected_obb_score_file=$detect_obb_scores
sample_voxel_sidelengths="$vx $vy $vz"
svm_model=$trained_svm_path
out_dir_prefix=$output_dir/adjusted_obbs.txt

if [ $do_adjust_obbs -gt 0 ]; then
    echo $adjust_bin --in-model $in_model --detected-obb-file $detected_obb_file --detected-obb-score-file $detected_obb_score_file --sample_voxel_sidelengths $sample_voxel_sidelengths --out-dir-prefix $out_dir_prefix --svm_model $svm_model
    sleep 2
    $adjust_bin --in-model $in_model --detected-obb-file $detected_obb_file --detected-obb-score-file $detected_obb_score_file --sample_voxel_sidelengths $sample_voxel_sidelengths --out-dir-prefix $out_dir_prefix --svm_model $svm_model
fi

adjusted_obb_txt=$out_dir_prefix"_obb_infos.txt"
adjusted_score_txt=$out_dir_prefix"_obb_scores.txt"
