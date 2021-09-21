cmake_minimum_required(VERSION 3.16)
project(cuda_wrapper LANGUAGES CXX)
set(_CUDA_WRAPPER_LIST_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")
set(_CUDA_WRAPPER_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")

# Write the output into the current source directory so we can commit the generated result.
function(generate_cuda_wrapper NAME STUB_DIR HEADER)
	find_package(Python COMPONENTS Interpreter)
	execute_process(COMMAND "${Python_EXECUTABLE}" -m virtualenv "${_CUDA_WRAPPER_BIN_DIR}/${PROJECT_NAME}_venv")

	# update the environment with VIRTUAL_ENV variable (mimic the activate script)
	set(ENV{VIRTUAL_ENV} "${_CUDA_WRAPPER_BIN_DIR}/${PROJECT_NAME}_venv")
	# change the context of the search
	set(Python_FIND_VIRTUALENV FIRST)
	# unset Python3_EXECUTABLE because it is also an input variable (see documentation, Artifacts Specification section)
	unset(Python_EXECUTABLE)
	# Launch a new search
	find_package(Python COMPONENTS Interpreter)
	# Install the requirements
	execute_process(COMMAND "${Python_EXECUTABLE}" -m pip install -r ${_CUDA_WRAPPER_LIST_DIR}/requirements.txt)

	find_package(CUDA REQUIRED)
	add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${NAME}_header.preprocessed
		# cmake has trouble finding pcpp in the virtual environment
		# remove __align__, __location__, CUDARTAPI because CppHeaderParser can't handle it
		COMMAND ${_CUDA_WRAPPER_BIN_DIR}/${PROJECT_NAME}_venv/bin/pcpp
			-D__location__\\\(x\\\)= -DCUDARTAPI= -D__align__\\\(x\\\)= ${HEADER} >
			${CMAKE_CURRENT_BINARY_DIR}/${NAME}_header.preprocessed || (exit 0)
		DEPENDS ${HEADER}
		COMMENT "Preprocessing header")
	add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${STUB_DIR}
		COMMAND ${Python_EXECUTABLE}
		ARGS ${_CUDA_WRAPPER_LIST_DIR}/make_stubs.py ${STUB_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${NAME}_header.preprocessed
		DEPENDS ${_CUDA_WRAPPER_LIST_DIR}/make_stubs.py ${CMAKE_CURRENT_BINARY_DIR}/${NAME}_header.preprocessed
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMENT "Generating stubs")
	add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.cpp
		COMMAND ${Python_EXECUTABLE}
		ARGS ${_CUDA_WRAPPER_LIST_DIR}/wrap_generate.py ${NAME}.cpp  ${STUB_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${NAME}_header.preprocessed
		DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${STUB_DIR} ${_CUDA_WRAPPER_LIST_DIR}/wrap_generate.py ${CMAKE_CURRENT_BINARY_DIR}/${NAME}_header.preprocessed
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMENT "Generating library source")

	cuda_add_library(${NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.cpp SHARED)
endfunction()
