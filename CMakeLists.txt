cmake_minimum_required(VERSION 3.0)

# project name
project(segment 
    VERSION 1.0
    DESCRIPTION "Utility for in situ plankton segmentation"
    LANGUAGES CXX)

# check the compiler
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # require at least g++ 8
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
        message(FATAL_ERROR "GCC version must be at least 8!")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # require at least clang 5
    ## if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5)
    ##     message(FATAL_ERROR "Clang version must be at least 5!")
    ## endif()
else()
    message(WARNING "You are using an unsupported compiler! Compilation has only been tested with Clang and GCC.")
endif()

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add the executable
add_executable(segment 
    main.cpp 
    imageProcessing.hpp
    imageProcessing.cpp)

target_link_libraries(segment ${OpenCV_LIBS})
target_compile_features(segment PRIVATE cxx_std_17)
target_link_libraries(segment stdc++fs)

# add the binary tree to the search path for include files
target_include_directories(segment PUBLIC "${PROJECT_BINARY_DIR}")

# NOTE: Enabling visual mode will disable the use of OpenMP
option(WITH_OPENMP "Compile the project with OpenMP" ON)
option(WITH_VISUAL "Enable visual mode, this will give previews of 
    segments to be cropped. NOTE: This will disable OpenMP" OFF)

if (WITH_VISUAL)
    target_compile_definitions(segment PRIVATE "WITH_VISUAL")
    set(WITH_OPENMP OFF)
endif()
if (WITH_OPENMP)
    target_compile_definitions(segment PRIVATE "WITH_OPENMP")
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
    set(WITH_VISUAL OFF)
endif()

# TODO: add testing functions
