cmake_minimum_required(VERSION 3.5)

project(CS759_project LANGUAGES CUDA CXX)

set(SOURCES_cfd
    src/cfd.cpp
    src/field.cpp 
    src/splinear.cpp
)

set(SOURCES_cfd_omp
    src/cfd_omp.cpp
    src/field.cpp 
    src/splinear_omp.cpp
)

set(SOURCES_cfd_cuda
    src/cfd_cuda.cu
    src/field.cpp
    src/splinear_cuda.cu   
)

add_library(cfd SHARED
    ${SOURCES_cfd}
)

add_library(cfd_omp SHARED
    ${SOURCES_cfd_omp}
)

add_library(cfd_cuda SHARED
    ${SOURCES_cfd_cuda}
)

target_include_directories(cfd
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(cfd_omp
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(cfd_cuda
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)   

target_compile_options(cfd_omp
    PUBLIC
    "-fopenmp"
)

set_target_properties(cfd_cuda 
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

set(CMAKE_CXX_FLAGS "-fopenmp")

add_executable(${PROJECT_NAME}
    src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    cfd
    cfd_omp
    cfd_cuda
)


