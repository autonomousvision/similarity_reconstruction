#!/bin/bash
build_root=../../urban_reconstruction_build
build_hashmap_dir=$build_root/hashmap
mkdir $build_root
mkdir $build_hashmap_dir
cd $build_hashmap_dir
cmake ../../urban_reconstruction/hashmap
make -j 6
# make -j 2
