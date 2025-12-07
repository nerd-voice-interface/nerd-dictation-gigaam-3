/**
 * GigaAM v3 Model Loader with Parallel Zstd Decompression
 *
 * Decompresses 17 embedded zstd volumes in parallel using 8 threads
 * into a single mmap'd memory region.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace gigaam {

/**
 * Load embedded model by decompressing all volumes in parallel.
 *
 * @param out_data Pointer to receive the decompressed model data
 * @param out_size Pointer to receive the total size
 * @param num_threads Number of decompression threads (default 8)
 * @return 0 on success, negative on error
 */
int load_embedded_model(const uint8_t** out_data, size_t* out_size, int num_threads = 8);

/**
 * Free the decompressed model memory (munmap).
 *
 * @param data Pointer returned by load_embedded_model
 * @param size Size returned by load_embedded_model
 */
void free_embedded_model(const uint8_t* data, size_t size);

/**
 * Check if embedded model is available.
 *
 * @return true if model volumes are embedded, false otherwise
 */
bool has_embedded_model();

} /* namespace gigaam */
