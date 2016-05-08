#!/bin/bash
set -e
test_tsdf_align_bin=$bin_dir/test_joint_cluster_model

#input_model=/home/dell/link_to_results/output-3d-model-camera3-test-withcam2-no-satarate/ply-0-1000-1.2-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-1.6_bin_tsdf_file.bin
input_model=$merged_model
#input_model=$test_input_tsdf
#detect_box_txt=/home/dell/Data/results/test_joint_cluster_model/test_boxes_two_c2.txt
detect_box_txt=$merged_detect_box_txt
#out_prefix=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/joint-cluster-model-test1/test1.ply

#lambda_average_scale=50.0 
##lambda_obs=0.05000
#lambda_reg_rot=50.0 
#lambda_reg_scale=50.0 
#lambda_reg_trans=50.0
#lambda_reg_zscale=1000000.0
##lambda_obs=0.0000
##lambda_reg_rot=0.0 
##lambda_reg_scale=0.0 
##lambda_reg_trans=0.0
##lambda_reg_zscale=0.0
#max_iter=3
lambda_out=210
lambda_out=9999999

#for pca_num in 0 
#do
output_root=$joint_learn_output_root
output_dir=$output_root/joint-pca-$pca_num-lavgscale-$lambda_average_scale-lobs-$lambda_obs-lregrot-$lambda_reg_rot-lregscale-$lambda_reg_scale-lregtrans-$lambda_reg_trans-lzscale-$lambda_reg_zscale-0.01percentile-rpca-robustalign-test-onlyalignterm-ls
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi

out_ply=$output_dir"/joint-opt.ply"
#echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number 3 --optimize-max-iter 3 --mesh_min_weight 0.8 
#echo "$test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter 20 --mesh_min_weight 0.8 "
if [ $do_joint_learn -gt 0 ]; then
    echo "#'$$$$$$$$$$$$$$$$$$$$'$ do joint learn "
echo $test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter $max_iter --mesh_min_weight 0.00 --lambda-average-scale $lambda_average_scale --lambda-obs $lambda_obs --lambda-reg-rot $lambda_reg_rot --lambda-reg-scale $lambda_reg_scale --lambda-reg-trans $lambda_reg_trans --noise_clean_counter_thresh $noise_counter_thresh --noise_connected_component_thresh $noise_comp_thresh --lambda_reg_zscale $lambda_reg_zscale --lambda_out $lambda_out
sleep 1
$test_tsdf_align_bin --in-model $input_model --detect-box-txt $detect_box_txt --out-prefix $out_ply --pca_number $pca_num --optimize-max-iter $max_iter --mesh_min_weight 0.00 --lambda-average-scale $lambda_average_scale --lambda-obs $lambda_obs --lambda-reg-rot $lambda_reg_rot --lambda-reg-scale $lambda_reg_scale --lambda-reg-trans $lambda_reg_trans --noise_clean_counter_thresh $noise_counter_thresh --noise_connected_component_thresh $noise_comp_thresh --lambda_reg_zscale $lambda_reg_zscale --lambda_out $lambda_out

fi
joint_learn_outdir=$output_dir
#done
