#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "visualization: usage: visualization.sh window_title visualization_txt"
    exit
fi
if [ -z $mesh_view_bin ]; then
    mesh_view_bin=/home/dell/codebase/mpi_project_git/similarity_reconstruction/code/similarity_reconstruction/visualization/trimesh2/bin.Linux64/mesh_view
fi
if [ ! -f $mesh_view_bin ]; then
    echo "visualization: The mesh_view binary file does not exist."
    return
fi
# the title of the window when displaying
window_title=$1
# set the txt file used for visualization
visualization_txt=$2
if [ ! -f $visualization_txt ]; then
    echo "visualization: The visualization_txt file does not exist."
    return
fi

#echo $mesh_view_bin $window_title $visualization_txt 
$mesh_view_bin $window_title $visualization_txt  &>/dev/null &
sleep 3
