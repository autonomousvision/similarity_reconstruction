#!/bin/bash
set -e
build_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap
test_tsdf_align_bin=$build_root/bin/test_read_from_xml

category_arr=(building car van truck)
startimg_arr=(1470 1890 3320 6050)
endimg_arr=(1790 2090 3530 6350)
detect_box_xmls[0]="/home/dell/Data/download_labels/label_xml_4_12/001431_001569.xml /home/dell/Data/download_labels/label_xml_4_12/001547_001650.xml /home/dell/Data/download_labels/label_xml_4_12/001629_001738.xml /home/dell/Data/download_labels/label_xml_4_12/001712_001845.xml"

detect_box_xmls[1]="/home/dell/Data/download_labels/label_xml_4_12/001822_001945.xml /home/dell/Data/download_labels/label_xml_4_12/001922_002032.xml /home/dell/Data/download_labels/label_xml_4_12/002009_002157.xml"

detect_box_xmls[2]="/home/dell/Data/download_labels/label_xml_4_12/003262_003404.xml /home/dell/Data/download_labels/label_xml_4_12/003381_003494.xml /home/dell/Data/download_labels/label_xml_4_12/003470_003608.xml"

detect_box_xmls[3]="/home/dell/Data/download_labels/label_xml_4_12/006014_006146.xml /home/dell/Data/download_labels/label_xml_4_12/006124_006231.xml /home/dell/Data/download_labels/label_xml_4_12/006211_006313.xml /home/dell/Data/download_labels/label_xml_4_12/006290_006440.xml"

#for ci in 0 1
#do
#    for stedi in 0 1 2 3
#    do
for ci in 0 1 2 3
do
    for stedi in 0 1 2 3
    do
        category=${category_arr[$ci]}
        startimg=${startimg_arr[$stedi]}
        endimg=${endimg_arr[$stedi]}
        echo $category
        echo $startimg
        echo $endimg

        detect_box_xml=${detect_box_xmls[$stedi]}
        echo $detect_box_xml
        out_dir=/home/dell/Data/download_labels/gt_4_12_$startimg-$endimg-$category-new2-7-11
        if [ ! -d "$out_dir" ]; then
            mkdir $out_dir
        fi
        out_txt=$out_dir/gt_$startimg-$endimg-$category".txt"

        echo $test_tsdf_align_bin --detect-box-xml $detect_box_xml --out-txt $out_txt --category $category
        sleep 2
        $test_tsdf_align_bin --detect-box-xml $detect_box_xml --out-txt $out_txt --category $category

    done
done
