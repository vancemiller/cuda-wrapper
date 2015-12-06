#!/bin/bash

# Expect 4 arguments

# Argument 1: Path to program to run
# Argument 2: Locks. BOOL: 0 = no locks, 1 = locks.
# Argument 3: Priority. BOOL: 0 = equal priority, 1 = decreasing priority.
# Argument 4: Number of iterations

if [ $# != 4 ]
then
  echo "Must specify 4 arguments: Program path, locks bool, priority bool, and number of iterations."
  exit
fi

P=$1; # Program path
L=$2; # Locks boolean
R=$3; # Priority boolean
N=$4; # Number of iterations

MODDIR="/home/ubuntu/GM/Locks/Kernel_Locks/rt_module/GPU_Locks.ko"
LIBDIR="/home/ubuntu/GM/Locks/Kernel_Locks/libcudart_wrapper.so"
LOGDIR="$(pwd)/"
LOGFILE="log"
LOG="2>>${LOGDIR}${LOGFILE}"
RESULTDIR="$(pwd)/tmp/"
RESULTFILE="res"
RESULT="1>${RESULTDIR}${RESULTFILE}"

SYNC=0 #Sync = spin

if [ $L == 1 ]
then
  # Locks enabled
  F="LD_PRELOAD=${LIBDIR} "
  # Insert interception module 
  insmod ${MODDIR}
else
  F=""
fi
# Descend into the directory
DIR=$(dirname "${P}")
PROG=$(basename "${P}")
cd $DIR
mkdir $RESULTDIR

# Launch the processes
for i in `seq 1 $N`;
do
  if [ $R == 1 ]
  then
  # Decreasing Priorities enabled
    E="${F}chrt -f $i "
  else
    E="${F}chrt -f 1 "
  fi
  E="${E}./${PROG} ${SYNC} ${LOG} ${RESULT}${i} &"
  eval $E
  pids[${i}]=$!
done

echo -n "$N,"

for i in `seq 1 $N`;
do
  wait ${pids[${i}]};
  echo -n "$(cat ${RESULTDIR}${RESULTFILE}${i})"
done

if [ $L == 1 ]
then
  rmmod ${MODDIR}
fi

rm -r $RESULTDIR
echo "";
