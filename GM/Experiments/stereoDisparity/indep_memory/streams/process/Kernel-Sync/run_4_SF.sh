#!/bin/bash
for i in `seq 1 4`;
do
    LD_PRELOAD=/home/ubuntu/GM/Locks/Kernel_Locks/libcudart_wrapper.so chrt -f 1 ../stereoDisparity $1  2>>log &
done
