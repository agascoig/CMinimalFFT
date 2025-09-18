# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version
project(asssembly C)          # Specify project name and language

set(OPENLIBM_PATH "$ENV{HOME}/github/openlibm")

set(ASM_FLAGS
    -march=native
    -O3
    -nodefaultlibs
    -fno-builtin
    -fPIC
    -funroll-loops
    -std=c23
    -fverbose-asm
    -g
    -I$ENV{HOME}/github/openlibm/include
    -I/usr/local/include
)

set(SRC_FILES
    ../direct.c
    ../bluestein.c
    ../pfa.c
    ../plan.c
    ../stockham.c
    ../wheel.c
    ../indexer.c
)

foreach(src ${SRC_FILES})
    get_filename_component(src_name ${src} NAME_WE)
    get_filename_component(src_abs ${src} ABSOLUTE)
    add_custom_command(
        OUTPUT ${src_name}.s
        COMMAND ${CMAKE_C_COMPILER} ${ASM_FLAGS} -S ${src_abs} -o ${src_name}.s
        DEPENDS ${src_abs}
        COMMENT "Generating assembly for ${src_abs}"
        VERBATIM
    )
    list(APPEND ASM_OUTPUTS ${src_name}.s)
endforeach()

add_custom_target(assembly ALL DEPENDS ${ASM_OUTPUTS})



