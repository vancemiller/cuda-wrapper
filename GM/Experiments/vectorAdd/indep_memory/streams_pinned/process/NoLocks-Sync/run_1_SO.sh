#!/bin/bash
for i in `seq 1 1`;
do
    ../vectorAdd $1 2>>log &
done
