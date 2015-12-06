#!/bin/bash
for i in `seq 1 $1`;
do
  output[${i}] = "$(LD_PRELOAD=/home/ubuntu/GM/Locks/Kernel_Locks/libcudart_wrapper.so chrt -f $i ../stereoDisparity $2 2>>log &)$"
  pids[${i}]=$!;
done

for i in `seq 1 $1`;
do
  wait $pids[${i}];
  echo "${output[${i}]}"
done

echo "";
