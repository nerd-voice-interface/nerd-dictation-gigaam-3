/**
 * GigaAM v3 Model Loader with Parallel Zstd Decompression
 *
 * Decompresses 17 embedded zstd volumes in parallel using threads
 * into a single mmap'd memory region.
 *
 * Uses res_embed API to access embedded compressed data.
 */

#include "model_loader.h"
#include "model_volumes.h"

#include <res_embed.h>
#include <zstd.h>
#include <sys/mman.h>
#include <thread>
#include <vector>
#include <atomic>
#include <iostream>
#include <cstring>

namespace gigaam {

/* Worker function for decompressing a single volume */
static void decompress_volume_worker(
    size_t volume_idx,
    uint8_t* output_base,
    std::atomic<int>* error_flag,
    std::atomic<size_t>* completed_count
) {
    if (error_flag->load() != 0) {
        return;  /* Another thread already failed */
    }

    const VolumeInfo& vol = MODEL_VOLUMES[volume_idx];

    /* Get compressed data from res_embed */
    size_t compressed_size = 0;
    const char* compressed = res::embed::get(vol.name, &compressed_size);

    if (!compressed || compressed_size == 0) {
        std::cerr << "[GigaAM] Volume " << volume_idx
                  << " (" << vol.name << ") not found in embedded resources" << std::endl;
        error_flag->store(1);
        return;
    }

    /* Output location in mmap'd buffer */
    uint8_t* output = output_base + vol.decompressed_offset;

    /* Decompress using zstd */
    size_t result = ZSTD_decompress(
        output, vol.decompressed_size,
        compressed, compressed_size
    );

    if (ZSTD_isError(result)) {
        std::cerr << "[GigaAM] Decompression error for volume " << volume_idx
                  << ": " << ZSTD_getErrorName(result) << std::endl;
        error_flag->store(1);
        return;
    }

    if (result != vol.decompressed_size) {
        std::cerr << "[GigaAM] Volume " << volume_idx
                  << " decompressed size mismatch: expected " << vol.decompressed_size
                  << ", got " << result << std::endl;
        error_flag->store(1);
        return;
    }

    (*completed_count)++;
}

int load_embedded_model(const uint8_t** out_data, size_t* out_size, int num_threads) {
    if (!out_data || !out_size) {
        return -1;
    }

    if (!has_embedded_model()) {
        std::cerr << "[GigaAM] No embedded model available" << std::endl;
        return -2;
    }

    /* Allocate mmap'd buffer for decompressed model */
    void* buffer = mmap(
        nullptr,
        MODEL_TOTAL_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0
    );

    if (buffer == MAP_FAILED) {
        std::cerr << "[GigaAM] mmap failed for " << MODEL_TOTAL_SIZE << " bytes" << std::endl;
        return -3;
    }

    std::cerr << "[GigaAM] Allocated " << MODEL_TOTAL_SIZE / (1024 * 1024)
              << " MB for model decompression" << std::endl;

    /* Parallel decompression */
    std::atomic<int> error_flag{0};
    std::atomic<size_t> completed_count{0};

    /* Limit threads to number of volumes */
    if (num_threads > static_cast<int>(MODEL_NUM_VOLUMES)) {
        num_threads = static_cast<int>(MODEL_NUM_VOLUMES);
    }
    if (num_threads < 1) {
        num_threads = 1;
    }

    std::cerr << "[GigaAM] Decompressing " << MODEL_NUM_VOLUMES
              << " volumes using " << num_threads << " threads..." << std::endl;

    std::vector<std::thread> workers;
    workers.reserve(MODEL_NUM_VOLUMES);

    /* Use a simple thread pool approach: launch workers in batches */
    size_t volumes_launched = 0;

    while (volumes_launched < MODEL_NUM_VOLUMES && error_flag.load() == 0) {
        /* Launch batch of threads */
        size_t batch_size = std::min(
            static_cast<size_t>(num_threads),
            MODEL_NUM_VOLUMES - volumes_launched
        );

        workers.clear();
        for (size_t i = 0; i < batch_size; i++) {
            workers.emplace_back(
                decompress_volume_worker,
                volumes_launched + i,
                static_cast<uint8_t*>(buffer),
                &error_flag,
                &completed_count
            );
        }

        /* Wait for batch to complete */
        for (auto& t : workers) {
            t.join();
        }

        volumes_launched += batch_size;
    }

    if (error_flag.load() != 0) {
        std::cerr << "[GigaAM] Decompression failed" << std::endl;
        munmap(buffer, MODEL_TOTAL_SIZE);
        return -4;
    }

    std::cerr << "[GigaAM] Decompression complete: " << completed_count.load()
              << " volumes processed" << std::endl;

    /* Make buffer read-only for safety */
    mprotect(buffer, MODEL_TOTAL_SIZE, PROT_READ);

    *out_data = static_cast<const uint8_t*>(buffer);
    *out_size = MODEL_TOTAL_SIZE;

    return 0;
}

void free_embedded_model(const uint8_t* data, size_t size) {
    if (data && size > 0) {
        munmap(const_cast<uint8_t*>(data), size);
    }
}

bool has_embedded_model() {
    /* Check if first volume is available via res_embed */
    size_t size = 0;
    const char* data = res::embed::get(MODEL_VOLUMES[0].name, &size);
    return data != nullptr && size > 0;
}

} /* namespace gigaam */
