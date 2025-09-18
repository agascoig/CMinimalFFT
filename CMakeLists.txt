# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version
project(CMinimalFFT C)          # Specify project name and language

# Set the C standard to C23
set(SDKROOT /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "-march=native -O3 -fno-builtin -fPIC -funroll-loops -std=c23")

add_library(minimalfft
    direct.c
    bluestein.c
    pfa.c
    plan.c
    stockham.c
    wheel.c
    indexer.c
    cminimalfft.h
    plan.h
)

find_library(SYSTEM_LIB System)

# Add OpenLibm library path
set(OPENLIBM_PATH "$ENV{HOME}/github/openlibm/build")

# Link with OpenLibm instead of default math library
target_link_libraries(minimalfft
${OPENLIBM_PATH}/libopenlibm.dylib
${SYSTEM_LIB})

# Add OpenLibm include directory if needed
target_include_directories(minimalfft PUBLIC
${OPENLIBM_PATH}/../include
/usr/local/include
)

# Disable automatic linking with system math library
set_target_properties(minimalfft PROPERTIES
LINK_FLAGS "-nostdlib++ -Wl,-force_load")

add_subdirectory(test)
add_subdirectory(assembly)

