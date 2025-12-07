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
set(GIGAAM_TOKENS "${GIGAAM_MODEL_DIR}/tokens.txt")

# Check model files exist for build
if(NOT EXISTS "${GIGAAM_VOLUMES_DIR}/part_00.zst")
    message(FATAL_ERROR "\n"
        "GigaAM v3 compressed volumes not found: ${GIGAAM_VOLUMES_DIR}/part_00.zst\n"
        "\n"
        "Please run the compression script first:\n"
        "  python scripts/compress_model.py models/gigaam-v3-ctc/model.onnx models/gigaam-v3-ctc/volumes\n"
    )
endif()

if(NOT EXISTS "${GIGAAM_TOKENS}")
    message(FATAL_ERROR "GigaAM tokens not found: ${GIGAAM_TOKENS}")
endif()

# ============================================================================
# Find zstd for decompression
# ============================================================================

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)

message(STATUS "GigaAM: Found zstd ${ZSTD_VERSION}")

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

# Disable res_embed example build
set(BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)

add_subdirectory(${RES_EMBED_DIR} ${CMAKE_CURRENT_BINARY_DIR}/res_embed)

# ============================================================================
# Include sherpa-onnx as submodule
# ============================================================================

set(SHERPA_ONNX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/sherpa-onnx")

if(NOT EXISTS "${SHERPA_ONNX_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "\n"
        "sherpa-onnx submodule not found!\n"
        "\n"
        "Please initialize the git submodule:\n"
        "  git submodule update --init --recursive\n"
    )
endif()

# sherpa-onnx settings: C API only, minimal build
set(SHERPA_ONNX_ENABLE_C_API ON CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_JNI OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_WEBSOCKET OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_TTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# Use system onnxruntime
set(SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE ON CACHE BOOL "" FORCE)
set(ENV{SHERPA_ONNXRUNTIME_INCLUDE_DIR} "/usr/include/onnxruntime/core/session")
set(ENV{SHERPA_ONNXRUNTIME_LIB_DIR} "/usr/lib")

add_subdirectory(${SHERPA_ONNX_DIR} ${CMAKE_CURRENT_BINARY_DIR}/sherpa-onnx)

message(STATUS "GigaAM backend: using sherpa-onnx from ThirdParty/sherpa-onnx")

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

add_subdirectory(${NERD_DICTATION_API_DIR} ${CMAKE_CURRENT_BINARY_DIR}/nerd-dictation-api)

# ============================================================================
# Source files
# ============================================================================

set(GIGAAM_SOURCES
    src/backend.cpp
    src/model_loader.cpp
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
    ${SHERPA_ONNX_DIR}/sherpa-onnx/c-api
    ${RES_EMBED_DIR}/include
    ${ZSTD_INCLUDE_DIRS}
)

target_link_libraries(nerd-dictation-gigaam
    PUBLIC
    nerd-dictation-api
    sherpa-onnx-c-api
    res_embed
    ${ZSTD_LIBRARIES}
)

target_compile_options(nerd-dictation-gigaam PRIVATE ${ZSTD_CFLAGS_OTHER})

# Define to use embedded model
target_compile_definitions(nerd-dictation-gigaam PRIVATE
    GIGAAM_USE_EMBEDDED_MODEL=1
)

# ============================================================================
# Embed compressed model volumes (17 volumes × ~26 MB each = ~428 MB total)
# ============================================================================

message(STATUS "GigaAM: Embedding 17 compressed model volumes...")

# Volume names for res_embed and whole-archive linking
set(GIGAAM_VOLUME_LIBS "")

foreach(i RANGE 0 16)
    if(i LESS 10)
        set(vol_num "0${i}")
    else()
        set(vol_num "${i}")
    endif()

    set(vol_name "model_vol_${vol_num}")
    set(vol_path "${GIGAAM_VOLUMES_DIR}/part_${vol_num}.zst")

    if(NOT EXISTS "${vol_path}")
        message(FATAL_ERROR "Volume not found: ${vol_path}")
    endif()

    res_embed(TARGET nerd-dictation-gigaam NAME "${vol_name}" PATH "${vol_path}" KEYWORD)

    list(APPEND GIGAAM_VOLUME_LIBS "${CMAKE_CURRENT_BINARY_DIR}/libnerd-dictation-gigaam_${vol_name}.a")
endforeach()

# Note: tokens.txt is installed as a file (sherpa-onnx requires file path for tokens)

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

# Install tokens.txt (required file - model is embedded but tokens need file path)
install(FILES ${GIGAAM_TOKENS}
    DESTINATION share/nerd-dictation-gigaam/model
)

# Export symbols for shared library
set_target_properties(nerd-dictation-gigaam PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Test
# ============================================================================

add_executable(test_transcribe tests/test_transcribe.cpp)

target_include_directories(test_transcribe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Build volume library list for test
set(TEST_VOLUME_LIBS "")
foreach(i RANGE 0 16)
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

# Make tokens.txt available for testing
configure_file(${GIGAAM_TOKENS}
    ${CMAKE_CURRENT_BINARY_DIR}/tokens.txt COPYONLY)
