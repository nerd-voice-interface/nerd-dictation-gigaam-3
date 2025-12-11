/**
 * FBank feature extractor implementation for GigaAM
 */

#include "fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include <cmath>
#include <algorithm>

namespace gigaam {

class FbankExtractor::Impl {
public:
    Impl() {
        // GigaAM-specific parameters
        opts_.frame_opts.dither = 0.0f;
        opts_.frame_opts.snip_edges = false;
        opts_.frame_opts.samp_freq = 16000;
        opts_.frame_opts.frame_shift_ms = 10.0f;
        opts_.frame_opts.frame_length_ms = 25.0f;
        opts_.frame_opts.remove_dc_offset = false;  // GigaAM specific
        opts_.frame_opts.preemph_coeff = 0.97f;
        opts_.frame_opts.window_type = "povey";
        opts_.frame_opts.round_to_power_of_two = true;

        opts_.mel_opts.num_bins = 64;  // GigaAM uses 64 mel bins
        opts_.mel_opts.low_freq = 0.0f;  // GigaAM specific
        opts_.mel_opts.high_freq = 8000.0f;  // GigaAM specific
        opts_.mel_opts.is_librosa = false;
    }

    std::vector<float> Extract(const float* samples, size_t num_samples) {
        // Create fresh fbank instance for each extraction
        knf::OnlineFbank fbank(opts_);

        // Accept waveform (samples should be in [-1, 1] range)
        fbank.AcceptWaveform(16000, samples, static_cast<int32_t>(num_samples));
        fbank.InputFinished();

        int32_t num_frames = fbank.NumFramesReady();
        if (num_frames == 0) {
            return {};
        }

        // Extract features
        std::vector<float> features(num_frames * 64);
        for (int32_t i = 0; i < num_frames; ++i) {
            const float* frame = fbank.GetFrame(i);
            std::copy(frame, frame + 64, features.data() + i * 64);
        }

        // Apply per_feature normalization (NeMo style)
        // Normalize each feature dimension to zero mean and unit variance
        NormalizePerFeature(features.data(), num_frames, 64);

        return features;
    }

private:
    void NormalizePerFeature(float* features, int32_t num_frames, int32_t num_bins) {
        // For each mel bin, compute mean and std, then normalize
        for (int32_t bin = 0; bin < num_bins; ++bin) {
            // Compute mean
            float sum = 0.0f;
            for (int32_t t = 0; t < num_frames; ++t) {
                sum += features[t * num_bins + bin];
            }
            float mean = sum / num_frames;

            // Compute variance
            float var_sum = 0.0f;
            for (int32_t t = 0; t < num_frames; ++t) {
                float diff = features[t * num_bins + bin] - mean;
                var_sum += diff * diff;
            }
            float std_dev = std::sqrt(var_sum / num_frames + 1e-5f);

            // Normalize
            for (int32_t t = 0; t < num_frames; ++t) {
                features[t * num_bins + bin] = (features[t * num_bins + bin] - mean) / std_dev;
            }
        }
    }

    knf::FbankOptions opts_;
};

FbankExtractor::FbankExtractor() : impl_(std::make_unique<Impl>()) {}
FbankExtractor::~FbankExtractor() = default;

std::vector<float> FbankExtractor::Extract(const float* samples, size_t num_samples) {
    return impl_->Extract(samples, num_samples);
}

}  // namespace gigaam
