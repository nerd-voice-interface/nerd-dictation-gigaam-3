/**
 * ONNX model implementation for GigaAM
 */

#include "onnx_model.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace gigaam {

// cuBLAS SGEMM algorithm per GEMM shape of the embedded model, tuned in place
// on the Quadro P500 (gigaam-profile: tune_table.py). onnxruntime-gpu-sm61
// uses it through its sgemm_algo_table option; shapes it does not list, and
// other GPUs, keep cuBLAS's default choice. Must be re-tuned when the
// embedded graph changes (scripts/optimize_model.py).
static const char* const kSgemmAlgoTable =
    "/usr/share/nerd-dictation-gigaam/sgemm-p500-gigaam.tsv";

class OnnxModel::Impl {
public:
    Impl() : env_(ORT_LOGGING_LEVEL_WARNING, "gigaam") {}

    bool Load(const void* data, size_t size, bool use_gpu) {
        try {
            Ort::SessionOptions session_opts = MakeSessionOptions(use_gpu);

            session_ = std::make_unique<Ort::Session>(
                env_,
                data,
                size,
                session_opts
            );

            // Get input/output names
            Ort::AllocatorWithDefaultOptions allocator;

            auto input_name = session_->GetInputNameAllocated(0, allocator);
            input_name_ = input_name.get();

            auto output_name = session_->GetOutputNameAllocated(0, allocator);
            output_name_ = output_name.get();

            // Read metadata
            ReadMetadata();

            return true;
        } catch (const Ort::Exception& e) {
            error_ = std::string("ONNX error: ") + e.what();
            std::cerr << "[GigaAM] " << error_ << std::endl;
            return false;
        }
    }

    bool LoadFromFile(const std::string& path, bool use_gpu) {
        try {
            Ort::SessionOptions session_opts = MakeSessionOptions(use_gpu);

            session_ = std::make_unique<Ort::Session>(
                env_,
                path.c_str(),
                session_opts
            );

            // Get input/output names
            Ort::AllocatorWithDefaultOptions allocator;

            auto input_name = session_->GetInputNameAllocated(0, allocator);
            input_name_ = input_name.get();

            auto output_name = session_->GetOutputNameAllocated(0, allocator);
            output_name_ = output_name.get();

            // Read metadata
            ReadMetadata();

            return true;
        } catch (const Ort::Exception& e) {
            error_ = std::string("ONNX error: ") + e.what();
            std::cerr << "[GigaAM] " << error_ << std::endl;
            return false;
        }
    }

    std::vector<float> Run(const float* features, int32_t num_frames) {
        if (!session_) {
            error_ = "Model not loaded";
            return {};
        }

        try {
            // GigaAM expects input shape: (batch=1, channels=64, time)
            // Features are in shape (time, 64), need to transpose to (64, time)
            std::vector<float> transposed(num_frames * 64);
            for (int32_t t = 0; t < num_frames; ++t) {
                for (int32_t c = 0; c < 64; ++c) {
                    transposed[c * num_frames + t] = features[t * 64 + c];
                }
            }

            // Create input tensor
            std::array<int64_t, 3> input_shape = {1, 64, num_frames};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault
            );

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                transposed.data(),
                transposed.size(),
                input_shape.data(),
                input_shape.size()
            );

            // Create feature_lengths tensor (required by GigaAM)
            std::array<int64_t, 1> length_shape = {1};
            int64_t feature_length = num_frames;
            Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info,
                &feature_length,
                1,
                length_shape.data(),
                length_shape.size()
            );

            // Run inference with two inputs: features and feature_lengths
            const char* input_names[] = {input_name_.c_str(), "feature_lengths"};
            const char* output_names[] = {output_name_.c_str()};
            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(input_tensor));
            inputs.push_back(std::move(length_tensor));

            auto outputs = session_->Run(
                Ort::RunOptions{nullptr},
                input_names,
                inputs.data(),
                inputs.size(),
                output_names,
                1
            );

            // Extract output
            auto& output = outputs[0];
            auto output_shape = output.GetTensorTypeAndShapeInfo().GetShape();
            size_t output_size = 1;
            for (auto dim : output_shape) {
                output_size *= dim;
            }

            const float* output_data = output.GetTensorData<float>();
            return std::vector<float>(output_data, output_data + output_size);

        } catch (const Ort::Exception& e) {
            error_ = std::string("Inference error: ") + e.what();
            std::cerr << "[GigaAM] " << error_ << std::endl;
            return {};
        }
    }

    bool IsLoaded() const { return session_ != nullptr; }
    const std::string& GetError() const { return error_; }
    int32_t VocabSize() const { return vocab_size_; }
    int32_t SubsamplingFactor() const { return subsampling_factor_; }

private:
    static Ort::SessionOptions MakeSessionOptions(bool use_gpu) {
        Ort::SessionOptions session_opts;
        session_opts.SetIntraOpNumThreads(4);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (use_gpu) {
            try {
                std::unordered_map<std::string, std::string> opts{
                    {"device_id", "0"},
                    // Pinned, not left to the default: on the Quadro P500 the
                    // DEFAULT search is 3.5x slower than EXHAUSTIVE/HEURISTIC,
                    // which run equally fast; EXHAUSTIVE costs ~2 ms on the
                    // first segment of each new length (measured, see
                    // gigaam-profile), as HEURISTIC does.
                    {"cudnn_conv_algo_search", "EXHAUSTIVE"},
                };
                if (std::filesystem::exists(kSgemmAlgoTable))
                    opts["sgemm_algo_table"] = kSgemmAlgoTable;
                Ort::CUDAProviderOptions cuda_opts;
                cuda_opts.Update(opts);
                session_opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
                std::cerr << "[GigaAM] Using CUDA execution provider"
                          << (opts.count("sgemm_algo_table")
                                  ? std::string(", SGEMM algorithms from ") + kSgemmAlgoTable
                                  : std::string())
                          << std::endl;
            } catch (const Ort::Exception& e) {
                std::cerr << "[GigaAM] CUDA not available: " << e.what()
                          << ", falling back to CPU" << std::endl;
            }
        }
        return session_opts;
    }

    void ReadMetadata() {
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            auto metadata = session_->GetModelMetadata();

            // Try to read vocab_size
            try {
                auto value = metadata.LookupCustomMetadataMapAllocated("vocab_size", allocator);
                if (value) {
                    vocab_size_ = std::stoi(value.get());
                }
            } catch (...) {}

            // Try to read subsampling_factor
            try {
                auto value = metadata.LookupCustomMetadataMapAllocated("subsampling_factor", allocator);
                if (value) {
                    subsampling_factor_ = std::stoi(value.get());
                }
            } catch (...) {}

        } catch (const Ort::Exception& e) {
            std::cerr << "[GigaAM] Failed to read metadata: " << e.what() << std::endl;
        }
    }

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    std::string error_;
    int32_t vocab_size_ = 34;
    int32_t subsampling_factor_ = 4;
};

OnnxModel::OnnxModel() : impl_(std::make_unique<Impl>()) {}
OnnxModel::~OnnxModel() = default;

bool OnnxModel::Load(const void* data, size_t size, bool use_gpu) {
    bool result = impl_->Load(data, size, use_gpu);
    if (result) {
        vocab_size_ = impl_->VocabSize();
        subsampling_factor_ = impl_->SubsamplingFactor();
    }
    error_ = impl_->GetError();
    return result;
}

bool OnnxModel::LoadFromFile(const std::string& path, bool use_gpu) {
    bool result = impl_->LoadFromFile(path, use_gpu);
    if (result) {
        vocab_size_ = impl_->VocabSize();
        subsampling_factor_ = impl_->SubsamplingFactor();
    }
    error_ = impl_->GetError();
    return result;
}

std::vector<float> OnnxModel::Run(const float* features, int32_t num_frames) {
    auto result = impl_->Run(features, num_frames);
    error_ = impl_->GetError();
    return result;
}

bool OnnxModel::IsLoaded() const {
    return impl_->IsLoaded();
}

}  // namespace gigaam
