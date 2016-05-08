#!/bin/bash
# this is without semantic...
# the root folder for all the data
set -e
data_root=/home/dell/Data/newdata_2/rect
# the binary file of reconstruction program
recon_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/fisheye_depths_fusion
# the starting and ending image numbers for reconstruction
#input_tsdf_file=/home/dell/link_to_results/output-3d-model-semantic2/ply-0-600-0.5-newtest-conf-flatten-0-noclean-voxellen-0.2-maxdist-2.0_bin_tsdf_file.bin
startimg=642
endimg=1043
param_prefix=param
image_prefix=img_00
#depth_prefix=simplefusion-spherr-supp-0.5-newdata1-keep-saturate-801-1301
depth_prefix=simplefusion-spherr-supp-0.5-newdata1-keep-satur-crop-5
# semantic folder name inside data_root
# semantic_pref=manual_annotation_1
# output directory
output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
outputdir=$output_root"output-3d-model-newdata-cam2-new3"
if [ ! -d "$outputdir" ]; then
mkdir $outputdir
fi

# weight: the minimum allowed weight for meshing
# niteration: how many times the diffusion smoothing is executed

for niteration in 0
do
for voxellen in 0.2
do
for negmaxlen in -4 -5 -6
do
for weight in 0 0.005 0.01 0.05
do
pos_maxdist=$(echo "$voxellen * 8" | bc)
neg_maxdist=$(echo "$voxellen * $negmaxlen" | bc)
echo $pos_maxdist
echo $neg_maxdist
outputfile_conf=$outputdir/ply-$startimg-$endimg-$weight-newtest-conf-flatten-$niteration-noclean-voxellen-$voxellen-posmaxdist-$pos_maxdist-negmaxdist-$neg_maxdist
#outputfile_conf_tsdfbin=$outputdir/tsdf-$startimg-$endimg-$weight-newtest-conf-flatten-$niteration-noclean-voxellen-$voxellen-maxdist-$maxdist

#$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --tsdf-filepath $outputfile_conf"_bin_tsdf_file.bin" --use_confidence --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix 
echo $recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix 
#$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --truncation_limit $maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix
#$recon_bin --in-root $data_root --out $outputfile_conf"-new.ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --truncation_limit $maxdist --voxel_length $voxellen --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix --input-tsdf-filepath $outputfile_conf"_bin_tsdf_file.bin"
$recon_bin --in-root $data_root --out $outputfile_conf".ply" --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --pos_truncation_limit $pos_maxdist --neg_truncation_limit $neg_maxdist --voxel_length $voxellen --do_fuse --param-prefix $param_prefix --image-prefix $image_prefix --depth-prefix $depth_prefix --use_confidence
done
done
done
done
