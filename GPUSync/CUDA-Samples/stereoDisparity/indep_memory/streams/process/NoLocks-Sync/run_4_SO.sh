#!/bin/bash
for i in `seq 1 4`;
do
    ../stereoDisparity $1 2>>log &
done
