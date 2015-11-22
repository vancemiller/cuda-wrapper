#!/bin/bash
    LD_PRELOAD=/home/ubuntu/GM/cuda_wrapper/POSIX_Locks/libcudart_wrapper.so ../stereoDisparity  -f -s $1 -n 1 30 2>>log

