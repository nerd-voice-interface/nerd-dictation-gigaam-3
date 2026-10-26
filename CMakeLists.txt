cmake_minimum_required(VERSION 3.20)
project(nerd-dictation-gigaam VERSION 3.0.0 LANGUAGES ASM C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Model and resource paths (GigaAM v3 CTC with embedded compressed volumes)
# ============================================================================

set(GIGAAM_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/gigaam-v3-ctc")
set(GIGAAM_VOLUMES_DIR "${GIGAAM_MODEL_DIR}/volumes")
set(GIGAAM_TOKENS_H "${GIGAAM_MODEL_DIR}/tokens.h")

# The repository keeps the UPSTREAM model in these volumes. The model the
# library embeds is derived from it at build time by scripts/build_model.py
# (exact graph rewrites, see scripts/optimize_model.py), into the build tree.
file(GLOB GIGAAM_UPSTREAM_VOLUMES "${GIGAAM_VOLUMES_DIR}/part_*.zst")
if(NOT GIGAAM_UPSTREAM_VOLUMES)
    message(FATAL_ERROR "GigaAM v3 upstream volumes not found in ${GIGAAM_VOLUMES_DIR}")
endif()
set(GIGAAM_NUM_VOLUMES 17)
set(GIGAAM_BUILT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gigaam-model")

if(NOT EXISTS "${GIGAAM_TOKENS_H}")
    message(FATAL_ERROR "GigaAM tokens header not found: ${GIGAAM_TOKENS_H}")
endif()

# ============================================================================
# Find dependencies
# ============================================================================

find_package(PkgConfig REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(ZSTD_EXECUTABLE zstd REQUIRED)

# zstd for decompression
pkg_check_modules(ZSTD REQUIRED libzstd)
message(STATUS "GigaAM: Found zstd ${ZSTD_VERSION}")

# onnxruntime for inference. Arch's onnxruntime(-cuda) ships a CMake package
# (onnxruntime::onnxruntime, headers in include/onnxruntime); builds without
# one (onnxruntime-gpu-sm61) install the headers in the source tree layout,
# include/onnxruntime/core/session.
find_package(onnxruntime CONFIG QUIET)
if(TARGET onnxruntime::onnxruntime)
    set(ONNXRUNTIME_LIB onnxruntime::onnxruntime)
    set(ONNXRUNTIME_INCLUDE_DIR "")
    message(STATUS "GigaAM: onnxruntime ${onnxruntime_VERSION} CMake package from ${onnxruntime_DIR}")
else()
    find_library(ONNXRUNTIME_LIB onnxruntime REQUIRED)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATH_SUFFIXES onnxruntime onnxruntime/core/session
        REQUIRED
    )
    message(STATUS "GigaAM: Found onnxruntime at ${ONNXRUNTIME_LIB}")
    message(STATUS "GigaAM: onnxruntime headers at ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

# ============================================================================
# Include res_embed for resource embedding
# ============================================================================

set(RES_EMBED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/res_embed")

if(NOT EXISTS "${RES_EMBED_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "\n"
        "res_embed submodule not found!\n"
        "\n"
        "Please initialize the git submodule:\n"
        "  git submodule update --init --recursive\n"
    )
endif()

# Build res_embed as static library
set(BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(${RES_EMBED_DIR} ${CMAKE_CURRENT_BINARY_DIR}/res_embed EXCLUDE_FROM_ALL)

# Reset BUILD_SHARED_LIBS for main library
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# ============================================================================
# Include kaldi-native-fbank for feature extraction
# ============================================================================

set(KALDI_NATIVE_FBANK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/kaldi-native-fbank")

if(NOT EXISTS "${KALDI_NATIVE_FBANK_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "\n"
        "kaldi-native-fbank submodule not found!\n"
        "\n"
        "Please initialize the git submodule:\n"
        "  git submodule update --init --recursive\n"
    )
endif()

# Build kaldi-native-fbank as static library with PIC for shared library linking
set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${KALDI_NATIVE_FBANK_DIR} ${CMAKE_CURRENT_BINARY_DIR}/kaldi-native-fbank EXCLUDE_FROM_ALL)

message(STATUS "GigaAM: Using kaldi-native-fbank from ThirdParty/kaldi-native-fbank")

# ============================================================================
# Include nerd-dictation-api (C API header)
# ============================================================================

set(NERD_DICTATION_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/nerd-dictation-api")

if(NOT EXISTS "${NERD_DICTATION_API_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "\n"
        "nerd-dictation-api submodule not found!\n"
        "\n"
        "Please initialize the git submodule:\n"
        "  git submodule update --init --recursive\n"
    )
endif()

add_subdirectory(${NERD_DICTATION_API_DIR} ${CMAKE_CURRENT_BINARY_DIR}/nerd-dictation-api EXCLUDE_FROM_ALL)

# ============================================================================
# Source files
# ============================================================================

set(GIGAAM_SOURCES
    src/backend.cpp
    src/model_loader.cpp
    src/fbank.cpp
    src/onnx_model.cpp
    src/ctc_decoder.cpp
)

# ============================================================================
# Library
# ============================================================================

add_library(nerd-dictation-gigaam SHARED
    ${GIGAAM_SOURCES}
)

target_include_directories(nerd-dictation-gigaam
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${KALDI_NATIVE_FBANK_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${RES_EMBED_DIR}/include
    ${GIGAAM_BUILT_DIR}
    ${ZSTD_INCLUDE_DIRS}
)

target_link_libraries(nerd-dictation-gigaam
    PUBLIC
    nerd-dictation-api
    kaldi-native-fbank-core
    ${ONNXRUNTIME_LIB}
    res_embed_static
    ${ZSTD_LIBRARIES}
)

target_compile_options(nerd-dictation-gigaam PRIVATE ${ZSTD_CFLAGS_OTHER})

# Define to use embedded model
target_compile_definitions(nerd-dictation-gigaam PRIVATE
    GIGAAM_USE_EMBEDDED_MODEL=1
)

# ============================================================================
# Build the embedded model from the upstream one, then embed its volumes
# ============================================================================

math(EXPR GIGAAM_LAST_VOLUME "${GIGAAM_NUM_VOLUMES} - 1")
set(GIGAAM_BUILT_VOLUMES "")
foreach(i RANGE 0 ${GIGAAM_LAST_VOLUME})
    if(i LESS 10)
        list(APPEND GIGAAM_BUILT_VOLUMES "${GIGAAM_BUILT_DIR}/part_0${i}.zst")
    else()
        list(APPEND GIGAAM_BUILT_VOLUMES "${GIGAAM_BUILT_DIR}/part_${i}.zst")
    endif()
endforeach()

add_custom_command(
    OUTPUT ${GIGAAM_BUILT_VOLUMES} "${GIGAAM_BUILT_DIR}/model_volumes.h"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_model.py"
            "${GIGAAM_VOLUMES_DIR}" "${GIGAAM_BUILT_DIR}" ${GIGAAM_NUM_VOLUMES}
    DEPENDS ${GIGAAM_UPSTREAM_VOLUMES}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_model.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/optimize_model.py"
    COMMENT "GigaAM: building the embedded model (exact rewrites, zstd -19)"
    VERBATIM)
add_custom_target(gigaam-model DEPENDS ${GIGAAM_BUILT_VOLUMES} "${GIGAAM_BUILT_DIR}/model_volumes.h")
add_dependencies(nerd-dictation-gigaam gigaam-model)

message(STATUS "GigaAM: Embedding ${GIGAAM_NUM_VOLUMES} compressed model volumes...")

# Volume names for res_embed and whole-archive linking
set(GIGAAM_VOLUME_LIBS "")

foreach(i RANGE 0 ${GIGAAM_LAST_VOLUME})
    if(i LESS 10)
        set(vol_num "0${i}")
    else()
        set(vol_num "${i}")
    endif()

    set(vol_name "model_vol_${vol_num}")
    set(vol_path "${GIGAAM_BUILT_DIR}/part_${vol_num}.zst")

    res_embed(TARGET nerd-dictation-gigaam NAME "${vol_name}" PATH "${vol_path}" KEYWORD)
    # The volume is an output of gigaam-model: build it first, once (in
    # parallel, Make would otherwise run build_model.py for each consumer).
    add_dependencies(nerd-dictation-gigaam_${vol_name} gigaam-model)

    list(APPEND GIGAAM_VOLUME_LIBS "${CMAKE_CURRENT_BINARY_DIR}/libnerd-dictation-gigaam_${vol_name}.a")
endforeach()

message(STATUS "GigaAM: Embedded volumes ready for linking")

# ============================================================================
# Force whole-archive linking for embedded resources
# ============================================================================

if(UNIX AND NOT APPLE)
    # Build the whole-archive link flags string
    set(WHOLE_ARCHIVE_FLAGS "-Wl,--whole-archive")
    foreach(lib ${GIGAAM_VOLUME_LIBS})
        set(WHOLE_ARCHIVE_FLAGS "${WHOLE_ARCHIVE_FLAGS} ${lib}")
    endforeach()
    set(WHOLE_ARCHIVE_FLAGS "${WHOLE_ARCHIVE_FLAGS} -Wl,--no-whole-archive")

    set_target_properties(nerd-dictation-gigaam PROPERTIES
        LINK_FLAGS "${WHOLE_ARCHIVE_FLAGS}"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS nerd-dictation-gigaam
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Export symbols for shared library
set_target_properties(nerd-dictation-gigaam PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Test (optional)
# ============================================================================

option(BUILD_TESTS "Build test executable" OFF)

if(BUILD_TESTS)
    add_executable(test_transcribe tests/test_transcribe.cpp)

    target_include_directories(test_transcribe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Build volume library list for test
    set(TEST_VOLUME_LIBS "")
    foreach(i RANGE 0 ${GIGAAM_LAST_VOLUME})
        if(i LESS 10)
            set(vol_num "0${i}")
        else()
            set(vol_num "${i}")
        endif()
        list(APPEND TEST_VOLUME_LIBS "nerd-dictation-gigaam_model_vol_${vol_num}")
    endforeach()

    target_link_libraries(test_transcribe PRIVATE
        nerd-dictation-gigaam
        ${TEST_VOLUME_LIBS}
    )
endif()
