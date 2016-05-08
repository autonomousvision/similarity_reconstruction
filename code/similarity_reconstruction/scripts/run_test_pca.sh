#!/bin/bash
set -e
. ./init.sh 
test_pca_bin=$bin_dir/test_pca
in_model=/home/dell/results_5/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-3320-ed-3530-vlen-0.2-rampsz-6-try1/recon-3320-3530-vlen-0.2-rampsz-6_tsdf.bin
matlab_files="/home/dell/results_5/test_pca_data/step_house/calibseq/joint-opt.ply_icompnum_0_whole_iter_2_TransformScale_EndSave_TransformScale.mat /home/dell/results_5/test_pca_data/step_house/seq3350/joint-opt_whole_iter_2_TransformScale_EndSave_TransformScale.mat"
matlab_files="/home/dell/results_5/test_pca_data/step_house/calibseq/stephouse_calibseq.mat /home/dell/results_5/test_pca_data/step_house/seq3350/joint-opt_whole_iter_2_TransformScale_EndSave_TransformScale.mat"
matlab_files="/home/dell/upload2/7-13/joint-opt.ply_icompnum_0_whole_iter_2_ModelCoeffPCA_reconstructed_withweight.mat"
#matlab_files="/home/dell/testmat.mat"
bb_size="51 51 51"
#bb_size="2 2 2"
mesh_min_weight=0
pca_num=1
pca_max_iter=50
noise_clean_counter_thresh=1
noise_connected_component_thresh=-1
output_dir=$result_root/test_pca_data/step_house/resultx/
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi
output_prefix=$output_dir/res.ply

echo $test_pca_bin --in-model $in_model --in-mat $matlab_files --bb-size $bb_size --mesh_min_weight $mesh_min_weight --pca_number $pca_num --pca-max-iter $pca_max_iter --noise_clean_counter_thresh $noise_clean_counter_thresh --noise_connected_component_thresh $noise_connected_component_thresh --out-prefix $output_prefix
$test_pca_bin --in-model $in_model --in-mat $matlab_files --bb-size $bb_size --mesh_min_weight $mesh_min_weight --pca_number $pca_num --pca-max-iter $pca_max_iter --noise_clean_counter_thresh $noise_clean_counter_thresh --noise_connected_component_thresh $noise_connected_component_thresh --out-prefix $output_prefix


