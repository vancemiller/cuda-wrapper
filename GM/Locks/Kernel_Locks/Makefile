CC := gcc
CCFLAGS := -Wall -fPIC -shared
LDFLAGS := -ldl -lpthread -lrt
INCLUDES := -I/usr/local/cuda/include/

################################################################################

all: build

build: libcudart_wrapper.so

libcudart_wrapper.so: GPU_Locklib.c ../../Wrapper/libcudart_wrapper.c 
	$(CC) $(CCFLAGS) $(INCLUDES) -o libcudart_wrapper.so GPU_Locklib.c ../../Wrapper/libcudart_wrapper.c $(LDFLAGS)

clean:
	rm libcudart_wrapper.so
