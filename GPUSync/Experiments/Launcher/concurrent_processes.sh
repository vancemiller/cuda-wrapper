#!/bin/bash
echo "Concurrent process timing. (Edit this file to change configuration.)"

############## Configuration #############

S=".csv"; # Filename suffix
I=5 # Number of concurrent processes
ITERATION_COUNT=100 # Number of times to run this experiment
R="sd1_results" # Output folder prefix
L="KERNEL" # Specify "KERNEL" for kernel locks and "POSIX" for posix locks

KERNEL_LOCK_DIR="/home/ubuntu/GPUSync/Locks/Kernel_Locks"
KERNEL_MODULE_DIR="/home/ubuntu/GPUSync/Locks/Kernel_Locks/rt_module"
KERNEL_MODULE_NAME="GPU_Locks"
POSIX_LOCK_DIR="/home/ubuntu/GPUSync/Locks/POSIX_Locks"
POSIX_LOCK_SERVER_DIR="${POSIX_LOCK_DIR}/shared"
POSIX_LOCK_SERVER_NAME="GPU_Locks"

CUDA_INTERCEPTION_LIBRARY_NAME="libcudart_wrapper.so"

# Test programs
VECTOR_ADD="/home/ubuntu/GPUSync/CUDA-Samples/vectorAdd/indep_memory/streams_pinned/process/vectorAdd"
STEREO_DISPARITY="/home/ubuntu/GPUSync/CUDA-Samples/stereoDisparity/indep_memory/streams/process/stereoDisparity"

E=$STEREO_DISPARITY # Executable

############## Helper functions #########
POSIX_LOCK_SERVER_PID=0

insert_module() {
  old_dir=$PWD
  if [ "$L" == "KERNEL" ]
  then
    cd $KERNEL_MODULE_DIR
    sudo insmod ${KERNEL_MODULE_NAME}.ko
  else
    cd $POSIX_LOCK_SERVER_DIR
    sudo ./$POSIX_LOCK_SERVER_NAME >/dev/null &
    POSIX_LOCK_SERVER_PID=$!
  fi
  cd $old_dir
}

remove_module() {
  if [ "$L" == "KERNEL" ]
  then
    sudo rmmod $KERNEL_MODULE_NAME
  else 
    sudo kill -TERM $POSIX_LOCK_SERVER_PID
  fi
}

############### Testing code #############

# Build test programs
PROG_DIR=$(dirname ${E})
TEST_DIR=$PWD
cd ${PROG_DIR}
make > /dev/null
cd ${TEST_DIR}

# Build locking code
if [ "$L" == "KERNEL" ]
then
  cd $KERNEL_MODULE_DIR
  make > /dev/null
  cd $KERNEL_LOCK_DIR
  make > /dev/null
  PRELOAD="LD_PRELOAD=${KERNEL_LOCK_DIR}/${CUDA_INTERCEPTION_LIBRARY_NAME}"
elif [ "$L" == "POSIX" ]
then
  cd $POSIX_LOCK_DIR
  make > /dev/null
  PRELOAD="LD_PRELOAD=${POSIX_LOCK_DIR}/${CUDA_INTERCEPTION_LIBRARY_NAME}"
else
  echo "Invalid lock type $L"
  exit
fi

# Make output directory
cd $TEST_DIR
OUT_DIR=${R}_${I}
mkdir $OUT_DIR

if [ $? != 0 ]
then
  echo Please specify a new output directory.
  exit
fi

# Run tests
for i in `seq 1 $ITERATION_COUNT`;
do
  echo "Iteration $i of $ITERATION_COUNT:"
  P="${OUT_DIR}/results$i"; #Filename prefix
  echo "Locks with decreasing priority"
  insert_module
  sudo ./run_i.sh $E 1 $I $PRELOAD | tee "${P}LDP${S}"
  remove_module
  echo "Locks with equal priority"
  insert_module
  sudo ./run_i.sh $E 0 $I $PRELOAD | tee "${P}LEP${S}"
  remove_module
  echo "No locks with decreasing priority"
  sudo ./run_i.sh $E 1 $I | tee "${P}NLDP${S}"
  echo "No locks with equal priority"
  sudo ./run_i.sh $E 0 $I | tee "${P}NLEP${S}"
done

# Clean up test programs
cd ${PROG_DIR}
make clean > /dev/null

# Locking code
if [ "$L" == "KERNEL" ]
then
  cd $KERNEL_MODULE_DIR
  make clean > /dev/null
  cd $KERNEL_LOCK_DIR
  make clean > /dev/null
else
  cd $POSIX_LOCK_DIR
  make clean > /dev/null
fi

exit

