#!/bin/bash
for i in `seq 1 $1`;
do
  LD_PRELOAD=/home/ubuntu/GM/Locks/Kernel_Locks/libcudart_wrapper.so chrt -f 1 ../stereoDisparity $2 2>>log &
  pids[${i}]=$!;
done

for pid in ${pids[*]};
do
  wait $pid;
done

echo "";
