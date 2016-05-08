#!/bin/bash
set -e
#. ./init.sh

bin_dir=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/
test_tsdf_align_bin=$bin_dir/crop_training_tsdf
#test_tsdf_align_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build_handin/bin/crop_training_tsdf

category_arr=(building car van truck)
startimg_arr=(1470 1890 3320 6050)
endimg_arr=(1790 2090 3530 6350)

declare -A detected_obb_file_arr

#for ci in 0 1 #1
#do  
#    for i in 0 1 2 3
#    do
for ci in 0 1 2 3 #1
do  
    for i in 0 # 1 2 3
    do
        category=${category_arr[$ci]}
        startimg=${startimg_arr[$i]}
        endimg=${endimg_arr[$i]}
        input_tsdf_arr[$i]=$result_root/reconstruction_closest_3/vri-fusing-result5_cam_3_with_2-st-$startimg-ed-$endimg-vlen-0.2-rampsz-6-try1/recon-$startimg-$endimg-vlen-0.2-rampsz-6_tsdf.bin
        input_tsdf_arr[$i]=/home/dell/results_5/reconstruction_closest_test-5-29-refractored/vri-fusing-result-s3_cam_3_with_2-st-1470-ed-1790-vlen-0.2-rampsz-6-try1/recon-1470-1790-vlen-0.2-rampsz-6_tsdf.bin
        input_tsdf_arr[$i]=/home/dell/results_5/demo_1470_1790_bilinear_w0.1/reconstruction_baseline-1470-1790/vri_fusing_result_side2_with_side1-1470-1790/recon-1470-1790_tsdf.bin
        input_tsdf_arr[$i]=/home/dell/results_5/demo_1470_1790_bilinear_w0.5/reconstruction_baseline-1470-1790/vri_fusing_result_side2_with_side1-1470-1790/recon-1470-1790_tsdf.bin
        # in readxmlinfo folder and don't need to be modified
        # the crop box file
        crop_box_txts[$i]="/home/dell/Data/download_labels/label_for_cropping/readxmlinfo/gt_4_12_"$startimg-$endimg-building/gt_$startimg-$endimg-building".txt"
        #crop_box_txts[$i]="/home/dell/Data/download_labels/label_for_cropping/readxmlinfo/gt_4_12_"$startimg-$endimg-building/gt_$startimg-$endimg-building".txt"
        #detected_obb_file_arr[$i $ci]=/home/dell/Data/download_labels/label_4_12_backup/gt_4_12_$startimg-$endimg-$category/gt_$startimg-$endimg-$category".txt"
        #detected_obb_file_arr[$i $ci]=/home/dell/Data/download_labels/label_4_12/gt_4_12_$startimg-$endimg-$category/gt_$startimg-$endimg-$category".txt"
        detected_obb_file_arr[$i $ci]=/home/dell/Data/download_labels/gt_7_11_labels/gt_4_12_$startimg-$endimg-$category-new2-7-11/gt_$startimg-$endimg-$category".txt"
        #echo ${detected_obb_file_arr[$seq_i $ci]}
    done

done

#for ci in 0 1 #1
#do
#    for stedi in 2 
#    do
for ci in 0 1 2 3 #1 #1
do
    for stedi in 0 #1 2 3  
    do
        category=${category_arr[$ci]}
        startimg=${startimg_arr[$stedi]}
        endimg=${endimg_arr[$stedi]}
        echo $category
        echo $startimg
        echo $endimg

        detect_obb=${detected_obb_file_arr[$stedi $ci]}
        crop_box_txt=${crop_box_txts[$stedi]}
        input_tsdf=${input_tsdf_arr[$stedi]}

        out_dir=$result_root/cropped_tsdf_for_training_7_12_new3_bthresh0.5/
        if [ ! -d "$out_dir" ]; then
            mkdir $out_dir
        fi
        out_dir2=$out_dir/gt_cropped_$startimg-$endimg-$category
        if [ ! -d "$out_dir2" ]; then
            mkdir $out_dir2
        fi
        out_txt=$out_dir2/gt_cropped_$startimg-$endimg-$category".ply"

        echo $test_tsdf_align_bin --in_model $input_tsdf --obj_obbs $detect_obb --crop_obb $crop_box_txt --out $out_txt
        sleep 1
        $test_tsdf_align_bin --in_model $input_tsdf --obj_obbs $detect_obb --crop_obb $crop_box_txt --out $out_txt
    done
done
