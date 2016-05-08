#!/bin/bash
set -e
in_model=$joint_output_tsdf
in_obb=$merged_detect_box_txt
out=$consistency_check_root/out
startim=$startimg
endim=$endimg
if [ $consistency_tsdf -gt 0 ]; then
    clean_tsdf="--clean_tsdf"
else
    clean_tsdf=""
fi

#. ./init.sh
#in_model=/home/dell/results_5/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-1470-ed-1790-vlen-0.2-rampsz-6-try1/recon-1470-1790-vlen-0.2-rampsz-6_tsdf.bin
#in_obb=/home/dell/results_5/detection_test_1470-2/detect_res_all.txt
#out=/home/dell/results_5/consist_check_test_1470-nonoisefilter
#if [ ! -d $out ]; then
#    mkdir $out
#fi
#out=$out/
#startim=1470
#endim=1790
#consistency_tsdf=1
#do_consistency_check=1
##filter_noise=100
#filter_noise=0
#depthmap_check=1
#skymap_check=1
#consistency_tsdf=1

if [ $consistency_tsdf -gt 0 ]; then
    clean_tsdf="--clean_tsdf"
else
    clean_tsdf=""
fi

data_roots="/home/dell/Data/data-4-10/2013_05_28_drive_0000_sync/image_02/rect /home/dell/Data/data-4-10/2013_05_28_drive_0000_sync/image_03/rect"
cam_info_prefix="param_scale_4"
skymap_prefix="sky_00_scale_4"
depth_prefix="depth_00_slic_cropped_scale_4_filtered"
mesh_min_weight=0
sky_thresh=2
max_cam_distance=30

consist_bin=$bin_dir/test_consistency_check

if [ $do_consistency_check -gt 0 ]; then
    echo $consist_bin --in_model $in_model --detect_obb_file $in_obb --output_filename $out --data_roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startim --end_image $endim --mesh_min_weight $mesh_min_weight --sky_map_thresh $sky_thresh --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise
    sleep 1
    $consist_bin --in_model $in_model --detect_obb_file $in_obb --output_filename $out --data_roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startim --end_image $endim --mesh_min_weight $mesh_min_weight --sky_map_thresh $sky_thresh --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise
fi
consistent_tsdf_output=$out".tsdf_consistency_cleaned_tsdf.bin"
