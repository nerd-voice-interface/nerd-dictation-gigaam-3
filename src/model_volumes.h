/**
 * GigaAM v3 Model Volumes Metadata
 *
 * Model split into 17 volumes, compressed with zstd -19
 * Total decompressed size: 885,236,194 bytes (~844 MB)
 * Total compressed size: ~428 MB
 *
 * Uses res_embed API to access embedded data at runtime.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace gigaam {

struct VolumeInfo {
    size_t compressed_size;
    size_t decompressed_size;
    size_t decompressed_offset;  /* Offset in final buffer */
    const char* name;  /* res_embed resource name */
};

constexpr size_t MODEL_NUM_VOLUMES = 17;
constexpr size_t MODEL_TOTAL_SIZE = 885236194;

/* Volume metadata with res_embed resource names */
const VolumeInfo MODEL_VOLUMES[MODEL_NUM_VOLUMES] = {
    { 26579740, 52428800,         0, "model_vol_00" },
    { 26267667, 52428800,  52428800, "model_vol_01" },
    { 26871474, 52428800, 104857600, "model_vol_02" },
    { 26910690, 52428800, 157286400, "model_vol_03" },
    { 26361165, 52428800, 209715200, "model_vol_04" },
    { 26719311, 52428800, 262144000, "model_vol_05" },
    { 26690236, 52428800, 314572800, "model_vol_06" },
    { 26347334, 52428800, 367001600, "model_vol_07" },
    { 27044503, 52428800, 419430400, "model_vol_08" },
    { 26318227, 52428800, 471859200, "model_vol_09" },
    { 26750304, 52428800, 524288000, "model_vol_10" },
    { 26652478, 52428800, 576716800, "model_vol_11" },
    { 26345465, 52428800, 629145600, "model_vol_12" },
    { 26358697, 52428800, 681574400, "model_vol_13" },
    { 26364098, 52428800, 734003200, "model_vol_14" },
    { 26351557, 52428800, 786432000, "model_vol_15" },
    { 23215617, 46375394, 838860800, "model_vol_16" },  /* last, smaller */
};

} /* namespace gigaam */
