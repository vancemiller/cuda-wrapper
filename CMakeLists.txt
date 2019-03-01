cmake_minimum_required(VERSION 3.5)
project(cuda_wrapper LANGUAGES CXX)

find_package(CUDA REQUIRED)

add_custom_command(OUTPUT stubs COMMAND ${CMAKE_SOURCE_DIR}/make_stubs.py COMMENT "Generating stubs")
add_custom_command(OUTPUT ${PROJECT_NAME}.cpp
	COMMAND ${CMAKE_SOURCE_DIR}/wrap_generate.py ARGS ${PROJECT_NAME}.cpp
	DEPENDS stubs ${CMAKE_SOURCE_DIR}/wrap_generate.py COMMENT "Generating library source")
cuda_add_library(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.cpp SHARED)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
