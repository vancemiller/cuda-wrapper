#!/usr/bin/python
#
# Generates CUDA wrapper functions for defined functions

import re
import pdb

### Config vars

OUTPUT_FILE = "libcudart_wrapper.c";
STUB_LOCATION = "./stubs/";
SOURCE_HEADER = "/usr/local/cuda/include/cuda_runtime_api.h";

###

findFuncNameRE = re.compile("(\w+)\(");
findPrototypeRE = re.compile("^extern\s+__host__.+?\(.*?\);$", flags=(re.DOTALL | re.MULTILINE))
finddvRE = re.compile("__dv\(.+?\)")

### function template ###
FUNC_BODY = """\t{func_ret} ret;
\t// Write your own custom c code in the {func_name}.c file
\tfprintf(stderr, ">>>>>>>> {func_name} called\\n");
\tret = orig_{func_name}({func_args});
\tfprintf(stderr, ">>>>>>>> {func_name} returned\\n");
\treturn ret;
""";

def generate():
    # open file
    prototypes = findPrototypes(SOURCE_HEADER);

    for prototype in prototypes:
        # remove newlines
        prototype = prototype.replace('\n', '');
        func_name, func_args, func_ret = parse_proto(prototype);

        stub = open(STUB_LOCATION + func_name + '.c', 'w');

        stub.write(FUNC_BODY.format(func_name=func_name, func_ret=func_ret, func_args=func_args));
        stub.close();

def parse_proto(prototype):
    # remove __dv(0)
    prototype = re.sub(finddvRE, "", prototype);
    # remove extern
    prototype = func_format_prototype(prototype);
    # strip everything but the name
    func_name = func_format_name(prototype);
    # get args
    func_args = func_format_args(prototype);
    # get ret
    func_ret = func_ret_type(prototype);

    return func_name, func_args, func_ret;

def findPrototypes(file_name):
    fd = open(file_name, "r");
    contents = fd.read();
    fd.close();
    return re.findall(findPrototypeRE, contents)

def func_format_prototype(prototype):
    # remove extern
    if (prototype[0:7] == "extern "):
        prototype = prototype[7:];
    # remove semicolon
    if (prototype[-1] == ";"):
        prototype = prototype[0: -1];
    return prototype.replace("__dv(0)", "");

def func_format_name(prototype):
    return re.search(findFuncNameRE, prototype).group(1);

def func_format_args(prototype):
    args = [ param.strip().split(" ")[-1].replace("*", "") \
            if param.strip().find(" ") != -1 \
            else "" \
            for param in re.findall("\(.*\)", prototype)[0][1:-1].split(",") ];
    return re.sub("['\[\]]", "", str(args));

def func_ret_type(prototype):
    ret = re.sub("__.*?__", "", prototype);
    ret = re.sub("CUDARTAPI", "", ret);
    return re.sub("[A-Za-z0-9\$]+\(.*?\)", "", ret).strip();

if __name__ == "__main__":
    generate()

