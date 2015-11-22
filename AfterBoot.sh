#!/bin/bash -v
echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable
cat /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable
echo 1 > /sys/devices/system/cpu/cpu0/online
echo 1 > /sys/devices/system/cpu/cpu1/online
echo 1 > /sys/devices/system/cpu/cpu2/online
echo 1 > /sys/devices/system/cpu/cpu3/online
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo -1 >/proc/sys/kernel/sched_rt_runtime_us
echo "AfterBoot Done"
