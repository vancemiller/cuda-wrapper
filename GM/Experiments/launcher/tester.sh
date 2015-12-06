#!/bin/bash

S=".csv"; # Filename suffix
I=4 # Number of processes

VECTOR_ADD="/home/ubuntu/GM/CUDA-Samples/vectorAdd/indep_memory/streams_pinned/process/vectorAdd"
STEREO_DISPARITY="/home/ubuntu/GM/CUDA-Samples/stereoDisparity/indep_memory/streams/process/stereoDisparity"

E=$VECTOR_ADD # Executable

for i in `seq 45 100`;
do
  P="va_results${I}/results$i"; #Filename prefix
  echo "Locks with decreasing priority"
  sudo ./run_i.sh $E 1 1 $I | tee "${P}LDP${S}"
  echo "Locks with equal priority"
  sudo ./run_i.sh $E 1 0 $I | tee "${P}LEP${S}"
  echo "No locks with decreasing priority"
  sudo ./run_i.sh $E 0 1 $I | tee "${P}NLDP${S}"
  echo "No locks with equal priority"
  sudo ./run_i.sh $E 0 0 $I | tee "${P}NLEP${S}"
done


