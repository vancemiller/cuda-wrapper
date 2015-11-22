#!/bin/bash
for i in `seq 1 4`;
do
    ../vectorAdd $1 2>>log &
done
