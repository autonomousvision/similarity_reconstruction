#!/bin/bash
set -e
merge_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/test_merge_vri

in_dir=/home/dell/link_to_results/tovri/
in_dir=/home/dell/testply/vrimerge1/
rampsize=6
# the starting and ending image numbers for reconstruction
startimg=1945
endimg=1960
vri_suffix=".depthmesh.rampsz-5.vlen-0.2.vri"
# output directory
output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
output_root=/home/dell/testply/vrimerge1/
outputdir=$output_root"merge_vri_1/"
if [ ! -d $outputdir ]; then
mkdir $outputdir
fi
outprefix=$outputdir/merge_vri_recon-$startimg-$endimg-rampsz-$rampsize".ply"

echo $merge_bin --in-dir $in_dir --rampsize $rampsize --out-prefix $outprefix  --vri-suffix $vri_suffix --start_image $startimg --end_image $endimg
$merge_bin --in-dir $in_dir --rampsize $rampsize --out-prefix $outprefix  --vri-suffix $vri_suffix --start_image $startimg --end_image $endimg
