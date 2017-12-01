CC := gcc
CCFLAGS := -Wall -fPIC -shared
INCLUDES := -I/usr/local/cuda/targets/x86_64-linux/include/

################################################################################

all: build

build: libcudart_wrapper.so

libcudart_wrapper.so: libcudart_wrapper.c
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ $^

clean:
	rm libcudart_wrapper.so
