#!/usr/bin/env python3
#
# Generates stub C code fragments for each Cuda function defined in
# the cuda_runtime_api.h file.  There is one such C fragment generated
# for each Cuda function (stored in the ./stubs directory with the
# file name of the Cuda function).  Each stub consists of the lines of
# C code generated from the function template defined below.  The
# template code simply generates a fragment to call the original
# Cuda function and return to the caller.  These fragments are
# integrated into a complete wrapper for each call by the Python
# program wrap_generate.py.
#
# The wrapper for a Cuda function can be customized with GPU locking
# calls by editing the stub code for a function after it has been generated
# by this program and before it is integrated into wrappers by the
# wrap_generate.py program,  For example, the stub generated for the
# Cuda cudaMemcpy() function is:
# __host__ cudaError_t CUDARTAPI cudaMemcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
# {
#   __host__ cudaError_t CUDARTAPI ret;
#   // Write your own code here
#   ret = orig_cudaMemcpy(dst, src, count, kind);
#   return ret;
# }
#
# Written by Vance Miller and Forrest Li, Department of Computer Science,
# University of North Carolina at Chapel Hill.
# 2015.

import pathlib
import sys
import CppHeaderParser
import re

# Config vars

STUB_LOCATION = "./stubs/"
SOURCE_HEADER = "/usr/local/cuda/include/cuda_runtime_api.h"

### Function Template ###
# By modifying this template, other wrapper executions can be
# achieved.  For example, the following template will enable
# logging of all Cuda calls in a program.

FUNC_WRAPPER = """{ret} {name}({parameters})
{{
    {ret} ret;
    // Write your own code here
    ret = orig_{name}({args});
    return ret;
}}
"""
# end of function template


def generate(header_file, stub_location):
  header = CppHeaderParser.CppHeader(header_file)
  index = 0
  for function in header.functions:
    parameters = ', '.join([x['type'] + ' ' + x['name']
                for x in function['parameters']])
    args = ', '.join([x['name'] for x in function['parameters']])

    pathlib.Path(stub_location).mkdir(parents=True, exist_ok=True)
    with open('{}/{}_{}.cpp'.format(stub_location, index, function['name']), 'w') as stub:
      stub.write(FUNC_WRAPPER.format(name=function['name'],
                       ret=function['rtnType'], parameters=parameters, args=args))
    index += 1


if __name__ == "__main__":
  if (len(sys.argv) >= 2):
    STUB_LOCATION = sys.argv[1]
  if (len(sys.argv) >= 3):
    SOURCE_HEADER = sys.argv[2]
  generate(SOURCE_HEADER, STUB_LOCATION)
