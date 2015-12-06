#!/bin/bash
for i in `seq 1 4`;
do
    LD_PRELOAD=/home/ubuntu/GM/Locks/POSIX_Locks/libcudart_wrapper.so chrt -f $i ../stereoDisparity $1  2>>log &
done
