#!/bin/bash

# Expect 3 arguments

# Argument 1: Program to run.
# Argument 2: Priority. BOOL: 0 = equal priority, 1 = decreasing priority.
# Argument 3: Number of iterations
# Argument 4: (optional) LD_PRELOAD string for shared library overrides
if [ "3" -gt "$#" ]
then
  echo "Must specify 3 arguments: Program, priority bool, and number of iterations."
  exit
fi

P=$1 # Program path
R=$2 # Priority boolean
N=$3 # Number of iterations
LD=${4:-""} #LD_PRELOAD path

LOGDIR="$PWD/"
LOGFILE="log"
LOG="2>>${LOGDIR}${LOGFILE}"
RESULTDIR="$PWD/.tmp/"
RESULTFILE="res"
RESULT="1>${RESULTDIR}${RESULTFILE}"
PROGDIR=$(dirname ${P})
SYNC=0 #Sync = spin

mkdir $RESULTDIR
cd $PROGDIR
# Launch the processes
for i in `seq 1 $N`;
do
  if [ $R == 1 ]
  then
  # Decreasing Priorities enabled
    E="chrt -f $i "
  else
    E="chrt -f 1 "
  fi
  E="${LD} ${E} ${P} ${SYNC} ${LOG} ${RESULT}${i} &"
  eval $E
  pids[${i}]=$!
done

echo -n "$N,"

for i in `seq 1 $N`;
do
  wait ${pids[${i}]};
  echo -n "$(cat ${RESULTDIR}${RESULTFILE}${i})"
done

rm -r $RESULTDIR

echo "";

