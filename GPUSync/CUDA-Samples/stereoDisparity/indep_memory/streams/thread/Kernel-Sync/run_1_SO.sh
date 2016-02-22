#!/bin/bash
    LD_PRELOAD=/home/ubuntu/GM/cuda_wrapper/Kernel_Locks/libcudart_wrapper.so ../stereoDisparity  -f -s $1 -n 1 30 2>>log

