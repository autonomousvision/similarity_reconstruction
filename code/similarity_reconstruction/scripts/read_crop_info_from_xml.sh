#!/bin/bash
set -e
build_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap
test_tsdf_align_bin=$build_root/bin/test_read_from_xml

category_arr=(building car van truck)
startimg_arr=(1470 1890 3320 6050)
endimg_arr=(1790 2090 3530 6350)
detect_box_xmls[0]="/home/dell/Data/download_labels/label_for_cropping/001431_001569.xml"

#detect_box_xmls[1]="/home/dell/Data/download_labels/label_xml_4_12/001822_001945.xml /home/dell/Data/download_labels/label_xml_4_12/001922_002032.xml /home/dell/Data/download_labels/label_xml_4_12/002009_002157.xml"
detect_box_xmls[1]="/home/dell/Data/download_labels/label_for_cropping/001922_002032.xml.1"

detect_box_xmls[2]="/home/dell/Data/download_labels/label_for_cropping/003262_003404.xml"

detect_box_xmls[3]="/home/dell/Data/download_labels/label_for_cropping/006124_006231.xml"

for ci in 0 1 2 3
do
    for stedi in 1
    do
        category=${category_arr[$ci]}
        startimg=${startimg_arr[$stedi]}
        endimg=${endimg_arr[$stedi]}
        echo $category
        echo $startimg
        echo $endimg

        detect_box_xml=${detect_box_xmls[$stedi]}
        echo $detect_box_xml
        out_dir=/home/dell/Data/download_labels/label_for_cropping/readxmlinfo/gt_4_12_$startimg-$endimg-$category
        if [ ! -d "$out_dir" ]; then
            mkdir $out_dir
        fi
        out_txt=$out_dir/gt_$startimg-$endimg-$category".txt"

        echo $test_tsdf_align_bin --detect-box-xml $detect_box_xml --out-txt $out_txt --category $category
        sleep 12
        $test_tsdf_align_bin --detect-box-xml $detect_box_xml --out-txt $out_txt --category $category

    done
done
