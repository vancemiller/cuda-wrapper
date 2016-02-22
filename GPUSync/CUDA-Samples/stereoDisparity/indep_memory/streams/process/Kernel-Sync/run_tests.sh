#!/bin/bash
# Call script with first parameter as the experiment number.
# Results are stored in files identified by the experiment number.
# Reusing a number will overwrite results.

# The second, optional parameter is the number of processes to attempt to run at once. 
# WARNING: The board will hang if too many realtime processes are launched concurrently.
# 30 is too large...

I=${2:-10}; #Iterations, defaults to 10
P="results$1"; #Filename prefix
S=".csv"; #Filename suffix
Y=0 #Sync level = spin

echo "Locks with increasing priority"
sudo ./timing_test_locks_increasing_priority.sh $I $Y | tee "${P}LIP${S}"
echo "Locks with equal priority"
sudo ./timing_test_locks.sh $I $Y | tee "${P}LEP${S}"
echo "No locks with increasing priority"
sudo ./timing_test_no_locks_increasing_priority.sh $I $Y | tee "${P}NLIP${S}"
echo "No locks with equal priority"
sudo ./timing_test_no_locks.sh $I $Y | tee "${P}NLEP${S}"

