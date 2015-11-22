#!/bin/bash
for i in `seq 1 4`;
do
    chrt -f 1 ../stereoDisparity $1 2>>log &
done
