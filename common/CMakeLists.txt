#set library name
set(LIBNAME benchmark_common)

#set src files
file(GLOB SOURCE_FILES
    src/*.cpp
)

#set header files
file(GLOB HEADER_FILES
    include/*.h
)

#build a static library
add_library(${LIBNAME} STATIC ${SOURCE_FILES} ${HEADER_FILES})