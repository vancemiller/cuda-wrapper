#!/usr/bin/python3
#
# Generates a complete library of wrapper functions for all the
# CUDA functions defined in cuda_runtime_api.h. The library is a single
# C file (libcudart_wrapper.cpp) which should be compiled to a shared
# object (.so) to form a dynamically loaded library
# libcudart_wrapper.so. (see Makefile)
# This library is loaded at execution time for
# a CUDA program using the LD_PRELOAD function before the real
# libcudart is loaded. This sequence causes calls from the CUDA program to
# be linked to libcudart_wrapper.so which is then linked to the real
# libcudart. The result is that CUDA calls from the program go first
# to the wrapper library where they can be acted on before being forwarded to
# the real CUDA library. For example,
#
# LD_PRELOAD=<complete path to>/libcudart_wrapper.so ./<cuda program>
#
# Note that the Makefile for the CUDA program should specify dynamic linkage using
# the flags "-cudart shared" on the nvcc command to generate the executable
# Example from a Makefile in the CUDA sample programs:
# $(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -cudart shared -o $@ $+ $(LIBRARIES)
#
# Written by Vance Miller and Forrest Li, Department of Computer Science,
# University of North Carolina at Chapel Hill.
# 2015.

import CppHeaderParser
import sys

# Config vars

OUTPUT_FILE = "libcudart_wrapper.cpp"  # relative to python program
STUB_LOCATION = "./stubs/"  # relative to python program
SOURCE_HEADER = "/usr/local/cuda/include/cuda_runtime_api.h"  # absolute path

# library header
LIB_HEADER = '''\
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>
#include "cuda_runtime_api.h"

'''

# Intercept function template
FUNC_TEMPLATE = '''\
static {ret} (*orig_{name})({parameters}) = NULL;

{stub}

'''

# Library init function header
INIT_HEADER = '''\
__attribute__((constructor)) static void init() {
'''

# Init template for each function to intercept
INIT_TEMPLATE = '''\
  // clear dl error
  dlerror();
  if (orig_{name} == NULL) {{
    typedef {ret} (*{name}_t)({parameters});
    orig_{name} = reinterpret_cast<{name}_t>(dlsym(RTLD_NEXT, "{name}"));
  }}
  if (orig_{name} == NULL)
  {{
    fprintf(stderr, ">>>>>>> %s\\n", dlerror());
  }}

'''

# End init function
INIT_FOOTER = '''}'''


def generate(output_file, stub_location, header_file):
  '''
    Generate the libcudart.cpp shim library
  '''
  header = CppHeaderParser.CppHeader(header_file)

  # open file to write to
  with open(OUTPUT_FILE, "w") as ofh:
    # write header
    ofh.write(LIB_HEADER)

    index = 0

    # write proto wrappers
    for function in header.functions:
      name = function['name']
      stub_file = '{}/{}_{}.cpp'.format(stub_location, index, name)
      index += 1
      parameters = ', '.join(x['type'] for x in function['parameters'])
      with open(stub_file) as stub:
        ofh.write(FUNC_TEMPLATE.format(name=name, ret=function['rtnType'], parameters=parameters, stub=stub.read()))
    
    # write dlsym initialization
    ofh.write(INIT_HEADER)
    index = 0
    for function in header.functions:
      parameters = ', '.join(x['type'] for x in function['parameters'])
      ofh.write(INIT_TEMPLATE.format(
            name=function['name'], ret=function['rtnType'], parameters=parameters))
      index +=1
    ofh.write(INIT_FOOTER)


if __name__ == "__main__":
  if (len(sys.argv) >= 2):
    OUTPUT_FILE = sys.argv[1]
  if (len(sys.argv) >= 3):
    STUB_LOCATION = sys.argv[2]
  if (len(sys.argv) >= 4):
    SOURCE_HEADER = sys.argv[3]
  generate(OUTPUT_FILE, STUB_LOCATION, SOURCE_HEADER)
