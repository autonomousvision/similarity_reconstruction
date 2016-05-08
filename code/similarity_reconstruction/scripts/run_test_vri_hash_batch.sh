#!/bin/bash
root_dir=/ps/geiger/czhou/cars_semi_convex_hull/plys-test-conversion/
prefix=car_
for i in {1..16}
do
    filename=$root_dir$prefix$i".ply"
    echo $filename
    ./run_test_vri_hash_conversion.sh "$filename"
done
