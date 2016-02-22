CC := gcc
CCFLAGS := -Wall -lpthread -lrt

######################################################################################


all: build

build: GPU_Locks

GPU_Locks: GPU_Locks.c
	$(CC) GPU_Locks.c -o GPU_Locks $(CCFLAGS)

clean: 
	rm -f GPU_Locks
