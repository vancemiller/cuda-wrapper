#!/bin/bash
    LD_PRELOAD=/home/ubuntu/GM/cuda_wrapper/Kernel_Locks/libcudart_wrapper.so ../stereoDisparity -b 1 -r -s $1 -n 4 30 2>>log

