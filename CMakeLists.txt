cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME gpu-compute-driver-bench)
set(COMMON_LIB_NAME common)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_MCC "Enable MCC Compiler, Default use g++" OFF)
option(ENABLE_NVCC "Enable NVCC Compiler, Default use g++" OFF)
option(ENABLE_DEBUG "Enable Debug Mode, Default Release Mode" OFF)

if(ENABLE_NVCC)
    project(${PROJECT_NAME} LANGUAGES CUDA C CXX)
    include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    /usr/local/cuda/include
    /usr/include/libcpuid
    /usr/include/eigen3
    )
    link_directories(/usr/local/cuda/targets/x86_64-linux/lib/stubs)
    link_directories(/usr/local/cuda/targets/x86_64-linux/lib)
    link_directories(/usr/local/cuda/lib64)
    add_definitions(-DTEST_ON_NVIDIA)
else()
    project(${PROJECT_NAME} LANGUAGES C CXX)
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/common/include
        /usr/local/musa/include
        /usr/include/libcpuid
        /usr/include/eigen3
        )
    link_directories(/usr/local/musa/lib)
endif()

if(ENABLE_DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
  message("Enable Debug Mode")
endif()

if(ENABLE_MCC)
  message("USE MCC COMPILER")
  set (MTCC_COMPILER /usr/local/musa/bin/clang++ CACHE STRING "set MTCC Compiler path")
  set(CMAKE_CXX_COMPILER ${MTCC_COMPILER} CACHE STRING "C++ Compiler" FORCE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=${NVCC_COMPILER}")
endif()

if(ENABLE_NVCC)
    message("USE NVCC COMPILER")
    find_program(NVCC_EXECUTABLE nvcc)

    if(NVCC_EXECUTABLE)
        message("Found nvcc: ${NVCC_EXECUTABLE}")
        set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE} CACHE FILEPATH "Path to nvcc compiler" FORCE)
    else()
        message(FATAL_ERROR "nvcc not found. Please specify the path manually.")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif()
################################################################
# COMPUTE ARCH
################################################################\

if(ENABLE_MCC AND NOT DEFINED ENV{MTGPU_ARCH})
set(MTGPU_ARCH_MAPPING
    "0111=mp_10"
    "0106=mp_10"
    "0105=mp_10"
    "0103=mp_10"
    "0102=mp_10"
    "0101=mp_10"
    "0100=mp_10"
    "0123=mp_10"
    "0122=mp_10"
    "0121=mp_10"
    "0201=mp_21"
    "0200=mp_21"
    "0222=mp_21"
    "0203=mp_21"
    "0202=mp_21"
    "0301=mp_22"
    "0300=mp_22"
    "0323=mp_22"
    "0327=mp_22"
    "a101=mp_22"
    "0313=mp_22"
    "0400=mp_31"
    "0500=mp_32"
    "0600=mp_41"
    "0610=mp_42"
    "0700=mp_43"
)

execute_process(
    COMMAND bash -c "lspci -nn | grep -E '1ed5|0202'"
    OUTPUT_VARIABLE MTGPU_ARCH_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message("MTGPU_ARCH_RAW: ${MTGPU_ARCH_RAW}")

set(START_INDEX 0)
set(END_INDEX 0)

while(END_INDEX GREATER -1)
    string(SUBSTRING "${MTGPU_ARCH_RAW}" ${END_INDEX} -1 MTGPU_ARCH_RAW)
    string(FIND "${MTGPU_ARCH_RAW}" "1ed5:" INDEX_1ED5)
    string(FIND "${MTGPU_ARCH_RAW}" "0202:" INDEX_0202)

    if(INDEX_1ED5 GREATER -1)
        set(START_INDEX ${INDEX_1ED5})
    elseif(INDEX_0202 GREATER -1)
    	set(START_INDEX ${INDEX_0202})
    else()
        message(WARNING "Neither 1ed5 nor 0202 found in PCI ID list.")
        set(START_INDEX -1)
    endif()

    if(START_INDEX EQUAL -1)
        break()
    endif()

    set(END_INDEX ${START_INDEX})

    math(EXPR END_INDEX "${START_INDEX} + 5")
    string(SUBSTRING "${MTGPU_ARCH_RAW}" "${END_INDEX}" 4 SUBSTRING_RESULT)
    # 检查 SUBSTRING_RESULT 是否在 MTGPU_ARCH_MAPPING 中
    set(MTGPU_PCI_FOUND FALSE)
    foreach(mapping ${MTGPU_ARCH_MAPPING})
   	string(REPLACE "=" ";" item ${mapping})
       	list(GET item 0 arch)
       	list(GET item 1 mp)
      	if("${arch}" STREQUAL "${SUBSTRING_RESULT}")
            set(MTGPU_PCI_FOUND TRUE)
	    set(ENV{MTGPU_ARCH} ${mp})
            message("MTGPU_ARCH auto detected: ${mp}")
            break()
        endif()
    endforeach()

    # 如果找到则跳出循环
    if(MTGPU_PCI_FOUND)
        break()
    else()
        message("Not GPU PCI ID value: ${SUBSTRING_RESULT}")
    endif()

endwhile()
endif()

if (ENABLE_MCC)
    if (DEFINED ENV{MTGPU_ARCH} )
        set(COMPILE_ARCH $ENV{MTGPU_ARCH})
    else()
        message(WARNING "Unsupported MTGPU_ARCH value: ${MTGPU_ARCH}")
    endif()
    message(STATUS "--cuda-gpu-arch=${COMPILE_ARCH}")
endif()

add_subdirectory(common)
add_subdirectory(multicards)
add_subdirectory(schedule)
add_subdirectory(memory)
add_subdirectory(resource)

include_directories(common/include)
include_directories(${CMAKE_SOURCE_DIR}/bigModule/)
