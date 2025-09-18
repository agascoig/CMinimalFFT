# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version
project(test17 C)          # Specify project name and language

add_executable(test17
    test17.c
    ../direct.c
    ../bluestein.c
    ../pfa.c
    ../plan.c
    ../stockham.c
    ../wheel.c
    ../indexer.c
    ../cminimalfft.h
    ../plan.h
    hashmap.h
    MT_random.h
    MT_random.c
)

set(ASAN_FLAGS
#   -fsanitize=address
#   -fno-omit-frame-pointer
)

target_compile_options(test17 PRIVATE ${ASAN_FLAGS})
target_link_options(test17 PRIVATE ${ASAN_FLAGS})

# Link with OpenLibm instead of default math library
target_link_libraries(test17
${OPENLIBM_PATH}/libopenlibm.dylib
/opt/homebrew/lib/libfftw3.dylib
${SYSTEM_LIB})

# Add OpenLibm include directory if needed
target_include_directories(test17 PRIVATE
/usr/local/include
${OPENLIBM_PATH}/../include
..
)

