#!/bin/bash

S=".csv"; # Filename suffix
I=1 # Number of processes
START=1
STOP=100
R="timing_results_" # Output folder prefix

# Test programs
VECTOR_ADD="/home/ubuntu/GPUSync/CUDA-Samples/vectorAdd/indep_memory/streams_pinned/process/vectorAdd"
STEREO_DISPARITY="/home/ubuntu/GPUSync/CUDA-Samples/stereoDisparity/indep_memory/streams/process/stereoDisparity"

E=$STEREO_DISPARITY # Executable

# Build test programs
PROG_DIR=$(dirname ${E})
TEST_DIR=$PWD

cd ${PROG_DIR}
make > /dev/null
cd ${TEST_DIR}

# Locking code
KERNEL_LOCKS="/home/ubuntu/GPUSync/Locks/Kernel_Locks"
LOCK_DIR=$KERNEL_LOCKS
cd $LOCK_DIR
make > /dev/null
cd $TEST_DIR

# Make output directory
OUT_DIR=${R}${I}
mkdir $OUT_DIR

if [ $? != 0 ]
then
  echo Please specify a new output directory.
  exit
fi

# Run tests
for i in `seq $START $STOP`;
do
  P="${OUT_DIR}/results$i"; #Filename prefix
  echo "Locks with decreasing priority"
  sudo ./kern_log_i.sh $E 1 1 $I | tee "${P}LDP${S}"
  echo "Locks with equal priority"
  sudo ./kern_log_i.sh $E 1 0 $I | tee "${P}LEP${S}"
  echo "No locks with decreasing priority"
  sudo ./kern_log_i.sh $E 0 1 $I | tee "${P}NLDP${S}"
  echo "No locks with equal priority"
  sudo ./kern_log_i.sh $E 0 0 $I | tee "${P}NLEP${S}"
done

# Clean up test programs
cd ${PROG_DIR}
make clean > /dev/null

# Clean up locks
cd $LOCK_DIR
make clean > /dev/null

