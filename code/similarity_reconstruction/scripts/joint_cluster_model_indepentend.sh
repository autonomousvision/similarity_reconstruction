#!/bin/bash
set -e
. ./init.sh

test_tsdf_align_bin=$bin_dir/test_joint_cluster_model

input_model=/home/dell/results_5/test_pca_data/step_house/merged_model/merged_model_tsdf.bin
input_model=/home/dell/results_5/test_pca_data/cars/merged_model-1470-3350/merged_model-1470-3050_tsdf.bin
input_model=/home/dell/results_5/test_pca_data/multi_houses/merged_model-1470-1890/merged_model-1470-1890_tsdf.bin
#detect_box_txt=/home/dell/Data/results/test_joint_cluster_model/test_boxes_two_c2.txt
detect_box_txt=/home/dell/results_5/test_pca_data/step_house/merged_obbs/merged_model.obb_infos.txt
detect_box_txt=/home/dell/results_5/test_pca_data/cars/obbs-1470-3350/all-1470-3350-obb.txt
detect_box_txt=/home/dell/results_5/test_pca_data/multi_houses/obbs-1470-1890/merged_all.txt

output_root=/home/dell/results_5/test_pca_data/cars/result2/
output_root=/home/dell/results_5/test_pca_data/multi_houses/result1/
if [ ! -d "$output_root" ]; then
    mkdir $output_root
fi

echo "################## initial consistency check for one category ###########"
do_consistency_check=1
joint_output_tsdf=$input_model
st_neighbor=-1
ed_neighbor=2
depthmap_check=1
skymap_check=1
filter_noise=60
consistency_check_root=$output_root/init_consistency_check-$joint_learn_suffix-$st_neighbor-$ed_neighbor
echo consistency_check_root $consistency_check_root
if [ ! -d $consistency_check_root ]; then
    mkdir $consistency_check_root
fi
consistency_tsdf=1
. ./run-sky-consistency-checking.sh
#consistent_tsdf_output=$out".tsdf_consistency_cleaned_tsdf.bin"

lambda_average_scale=1000.0 
lambda_obs=0.2000
lambda_reg_rot=50.0 
lambda_reg_scale=50.0 
lambda_reg_trans=50.0
lambda_reg_zscale=0.0
noise_counter_thresh=5
noise_counter_thresh=3
noise_comp_thresh=-1
max_iter=3

for pca_num in 1 
do
    output_dir=$output_root/joint-pca-$pca_num-lavgscale-$lambda_average_scale-lobs-$lambda_obs-lregrot-$lambda_reg_rot-lregscale-$lambda_reg_scale-lregtrans-$lambda_reg_trans-lzscale-$lambda_reg_zscale-0.01percentile
    if [ ! -d "$output_dir" ]; then
        mkdir $output_dir
    fi
    out_ply=$output_dir"/joint-opt.ply"
    echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter $max_iter --mesh_min_weight 0.00 --lambda-average-scale $lambda_average_scale --lambda-obs $lambda_obs --lambda-reg-rot $lambda_reg_rot --lambda-reg-scale $lambda_reg_scale --lambda-reg-trans $lambda_reg_trans --noise_clean_counter_thresh $noise_counter_thresh --noise_connected_component_thresh $noise_comp_thresh --lambda_reg_zscale $lambda_reg_zscale
    sleep 1
    $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter $max_iter --mesh_min_weight 0.00 --lambda-average-scale $lambda_average_scale --lambda-obs $lambda_obs --lambda-reg-rot $lambda_reg_rot --lambda-reg-scale $lambda_reg_scale --lambda-reg-trans $lambda_reg_trans --noise_clean_counter_thresh $noise_counter_thresh --noise_connected_component_thresh $noise_comp_thresh --lambda_reg_zscale $lambda_reg_zscale
    joint_learn_outdir=$output_dir
done
