/**
 * FBank feature extractor implementation for GigaAM
 */

#include "fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include <algorithm>

namespace gigaam {

class FbankExtractor::Impl {
public:
    Impl() {
        // GigaAM-specific parameters
        // See: https://github.com/salute-developers/GigaAM/blob/main/gigaam/preprocess.py
        opts_.frame_opts.dither = 0.0f;
        opts_.frame_opts.snip_edges = false;
        opts_.frame_opts.samp_freq = 16000;
        opts_.frame_opts.frame_shift_ms = 10.0f;
        opts_.frame_opts.frame_length_ms = 25.0f;
        opts_.frame_opts.remove_dc_offset = false;
        opts_.frame_opts.preemph_coeff = 0.0f;  // GigaAM: no preemphasis
        opts_.frame_opts.window_type = "hann";  // GigaAM uses Hann window
        opts_.frame_opts.round_to_power_of_two = false;  // GigaAM uses n_fft=400

        opts_.mel_opts.num_bins = 64;
        opts_.mel_opts.low_freq = 0.0f;
        opts_.mel_opts.high_freq = 8000.0f;
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

        // Note: GigaAM model has normalize_type="" (empty) in metadata,
        // so no per_feature normalization is applied - model expects raw fbank

        return features;
    }

private:
    knf::FbankOptions opts_;
};

FbankExtractor::FbankExtractor() : impl_(std::make_unique<Impl>()) {}
FbankExtractor::~FbankExtractor() = default;

std::vector<float> FbankExtractor::Extract(const float* samples, size_t num_samples) {
    return impl_->Extract(samples, num_samples);
}

}  // namespace gigaam
