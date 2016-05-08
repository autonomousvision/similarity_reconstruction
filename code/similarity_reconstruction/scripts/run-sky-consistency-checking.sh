#!/bin/bash
in_model=$joint_output_tsdf
in_obb=$merged_detect_box_txt
out=$consistency_check_root/out

#. ./init.sh
#in_model=/home/dell/results_2/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-1470-ed-1790-vlen-0.2-rampsz-6-try1/recon-1470-1790-vlen-0.2-rampsz-6_tsdf.bin
#in_obb=/home/dell/results_2/seperate-seq-batch-all-seq0-detector-building2-car/batch-res-1470-1790-car-svmw1-10-svmc-100/merge_tsdf_obb_output-trainingseq-0-merge_score_thresh-1/merged_model.obb_infos.txt
#out=/home/dell/results_2/consist_check_test1
#if [ ! -d $out ]; then
#    mkdir $out
#fi
#out=$out/out
#startim=1470
#endim=1790

if [ $consistency_tsdf -gt 0 ]; then
    clean_tsdf="--clean_tsdf"
else
    clean_tsdf=""
fi
data_roots="/home/dell/Data/data-4-10/2013_05_28_drive_0000_sync/image_02/rect /home/dell/Data/data-4-10/2013_05_28_drive_0000_sync/image_03/rect"
cam_info_prefix="param_scale_4"
skymap_prefix="sky_00_scale_4"
depth_prefix="depth_00_slic_cropped_scale_4_filtered"
startim=$startimg
endim=$endimg
#startim=1546
#endim=1546
mesh_min_weight=0
sky_thresh=3
apply_to_category="0 1 2"
max_cam_distance=30

consist_bin=$bin_dir/test_consistency_check

if [ $do_consistency_check -gt 0 ]; then
    echo $consist_bin --in-model $in_model --in-obb $in_obb --out $out --data-roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startim --end_image $endim --mesh_min_weight $mesh_min_weight --sky_thresh $sky_thresh --apply_to_category $apply_to_category --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --st_neighbor $st_neighbor --ed_neighbor $ed_neighbor --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise
    sleep 1
    $consist_bin --in-model $in_model --in-obb $in_obb --out $out --data-roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startim --end_image $endim --mesh_min_weight $mesh_min_weight --sky_thresh $sky_thresh --apply_to_category $apply_to_category --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --st_neighbor $st_neighbor --ed_neighbor $ed_neighbor --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise

fi
consistent_tsdf_output=$out".tsdf_consistency_cleaned_tsdf.bin"
