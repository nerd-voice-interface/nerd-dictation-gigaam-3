/**
 * FBank feature extractor for GigaAM
 *
 * Extracts 64-dimensional mel-filterbank features using kaldi-native-fbank.
 * Parameters are tuned for GigaAM v3 model.
 */

#ifndef GIGAAM_FBANK_H
#define GIGAAM_FBANK_H

#include <memory>
#include <vector>
#include <cstdint>

namespace gigaam {

/**
 * GigaAM-specific FBank parameters:
 * - sample_rate: 16000 Hz
 * - num_mel_bins: 64
 * - frame_length: 25ms (400 samples)
 * - frame_shift: 10ms (160 samples)
 * - low_freq: 0 Hz
 * - high_freq: 8000 Hz
 * - window: povey
 * - remove_dc_offset: false
 * - dither: 0.0
 */

class FbankExtractor {
public:
    FbankExtractor();
    ~FbankExtractor();

    // Non-copyable
    FbankExtractor(const FbankExtractor&) = delete;
    FbankExtractor& operator=(const FbankExtractor&) = delete;

    /**
     * Extract fbank features from audio samples.
     *
     * @param samples Audio samples normalized to [-1, 1]
     * @param num_samples Number of samples
     * @return Features as flattened array (num_frames * 64)
     *         Features are per_feature normalized (zero mean, unit variance)
     */
    std::vector<float> Extract(const float* samples, size_t num_samples);

    /**
     * Get number of mel bins (always 64 for GigaAM)
     */
    int32_t NumMelBins() const { return 64; }

    /**
     * Get expected sample rate (always 16000 for GigaAM)
     */
    int32_t SampleRate() const { return 16000; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gigaam

#endif  // GIGAAM_FBANK_H
