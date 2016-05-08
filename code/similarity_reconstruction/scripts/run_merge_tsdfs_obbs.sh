#!/bin/bash
# the binary file of reconstruction program
set -e

. ./init.sh

merge_bin=$bin_dir/merge_tsdfs_obbs

echo "################## merge tsdf/obbs for one category ###################"
test_input_tsdf="/home/dell/results_5/test_pca_data/step_house/data/models/calib_seq/recon-800-1300-vlen-0.2-rampsz-6_tsdf.bin /home/dell/results_5/test_pca_data/step_house/data/models/seq1470/recon-1470-1790-vlen-0.2-rampsz-6_tsdf.bin"

test_input_tsdf[0]="/home/dell/results_5/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-1470-ed-1790-vlen-0.2-rampsz-6-try1/recon-1470-1790-vlen-0.2-rampsz-6_tsdf.bin" 
test_input_tsdf[1]="/home/dell/results_5/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-1890-ed-2090-vlen-0.2-rampsz-6-try1/recon-1890-2090-vlen-0.2-rampsz-6_tsdf.bin" 
#test_input_tsdf[1]="/home/dell/results_5/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-3320-ed-3530-vlen-0.2-rampsz-6-try1/recon-3320-3530-vlen-0.2-rampsz-6_tsdf.bin"
test_input_tsdf=${test_input_tsdf[*]}
echo $test_input_tsdf

detect_obb_infos="/home/dell/results_5/test_pca_data/step_house/data/obbs/calib_seqmerged_model.obb_infos.txt /home/dell/results_5/test_pca_data/step_house/data/obbs/seq1470/merged_model.obb_infos.txt"
merge_output_dir=$result_root/test_pca_data/step_house/merged_model-1470-3350/
merge_output_dir=$result_root/test_pca_data/multi_houses/merged_model-1470-1890/
if [ ! -d $merge_output_dir ]; then
    mkdir $merge_output_dir
fi
merge_output_prefix=$merge_output_dir/merged_model-1470-3050
merge_output_prefix=$merge_output_dir/merged_model-1470-1890
#model_fileoption="--in-models "$(IFS=$' '; echo "${input_tsdf_arr[*]}")
model_fileoption=""
merge_score_thresh=-100

echo $merge_bin --in-models $test_input_tsdf --in-obbs $detect_obb_infos --out $merge_output_prefix --obb-min-score $merge_score_thresh
$merge_bin --in-models $test_input_tsdf --in-obbs $detect_obb_infos --out $merge_output_prefix --obb-min-score $merge_score_thresh
merged_model=$merge_output_prefix"_tsdf.bin"
merged_detect_box_txt=$merge_output_prefix".obb_infos.txt"

merged_model_arr[$test_seq_i $ci]=$merged_model
merged_detect_box_txt_arr[$test_seq_i $ci]=$merged_detect_box_txt
