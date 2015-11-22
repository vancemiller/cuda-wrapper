#!/bin/bash
for i in `seq 1 $1`;
do
  echo -n "$i jobs,"
  insmod /home/ubuntu/GM/Locks/Kernel_Locks/rt_module/GPU_Locks.ko
  ./run_i_SO.sh $i $2;
  rmmod GPU_Locks
done
