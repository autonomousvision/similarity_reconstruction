#!/bin/bash
echo "$#"
echo "$0"
echo "$1"
echo "$2"
if [ "$#" -ne 2 ]; then
    echo "visualization: usage: visualization.sh window_title visualization_txt"
    exit
fi
if ! [ "$mesh_view_bin" ]; then
    mesh_view_bin=/home/dell/codebase/mpi_project_git/similarity_reconstruction/code/similarity_reconstruction/visualization/trimesh2/bin.Linux64/mesh_view
fi
if [ ! -f $mesh_view_bin ]; then
    echo "visualization: The mesh_view binary file does not exist."
    exit
fi
# the title of the window when displaying
window_title=$1
# set the txt file used for visualization
visualization_txt=$2
if [ ! -f $visualization_txt ]; then
    echo "visualization: The visualization_txt file does not exist."
    exit
fi

#echo $mesh_view_bin $window_title $visualization_txt 
$mesh_view_bin $window_title $visualization_txt &
sleep 3