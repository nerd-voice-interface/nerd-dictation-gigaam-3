/**
 * ONNX model implementation for GigaAM
 */

#include "onnx_model.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <dlfcn.h>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace gigaam {

// cuBLAS SGEMM algorithm per GEMM shape of the embedded model, tuned in place
// for one GPU model (gigaam-profile: tune_table.py) and named after it:
// <dir>/<gpu name, lower case, non-alphanumerics as '-'>.tsv, e.g.
// quadro-p500.tsv. onnxruntime-gpu-sm61 takes it through its sgemm_algo_table
// CUDA option; a stock onnxruntime does not know the option, and then the
// model simply runs with cuBLAS's default choice. Tables must be re-tuned when
// the embedded graph changes (scripts/optimize_model.py).
static const char* const kSgemmTableDir = "/usr/share/nerd-dictation-gigaam/sgemm";

// The name of CUDA device 0 (as onnxruntime's device_id 0 sees it), from the
// driver API, loaded at run time so the backend links no CUDA library.
static std::string CudaDeviceName() {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib)
        return {};
    using Init = int (*)(unsigned);
    using Get = int (*)(int*, int);
    using Name = int (*)(char*, int, int);
    auto init = reinterpret_cast<Init>(dlsym(lib, "cuInit"));
    auto get = reinterpret_cast<Get>(dlsym(lib, "cuDeviceGet"));
    auto name = reinterpret_cast<Name>(dlsym(lib, "cuDeviceGetName"));
    char buf[256] = {};
    int dev = 0;
    std::string result;
    if (init && get && name && init(0) == 0 && get(&dev, 0) == 0 &&
        name(buf, sizeof buf, dev) == 0)
        result = buf;
    dlclose(lib);
    return result;
}

// The table for this GPU, or "" if there is none.
static std::string SgemmTableForThisGpu() {
    std::string gpu = CudaDeviceName();
    if (gpu.empty())
        return {};
    std::string slug;
    for (unsigned char c : gpu) {
        if (std::isalnum(c))
            slug += static_cast<char>(std::tolower(c));
        else if (!slug.empty() && slug.back() != '-')
            slug += '-';
    }
    while (!slug.empty() && slug.back() == '-')
        slug.pop_back();
    std::string path = std::string(kSgemmTableDir) + "/" + slug + ".tsv";
    std::cerr << "[GigaAM] GPU \"" << gpu << "\": "
              << (std::filesystem::exists(path) ? "SGEMM table " + path
                                                : "no SGEMM table (" + path + ")")
              << std::endl;
    return std::filesystem::exists(path) ? path : std::string();
}

class OnnxModel::Impl {
public:
    Impl() : env_(ORT_LOGGING_LEVEL_WARNING, "gigaam") {}

    bool Load(const void* data, size_t size, bool use_gpu) {
        return Create(use_gpu, [&](const Ort::SessionOptions& opts) {
            return std::make_unique<Ort::Session>(env_, data, size, opts);
        });
    }

    bool LoadFromFile(const std::string& path, bool use_gpu) {
        return Create(use_gpu, [&](const Ort::SessionOptions& opts) {
            return std::make_unique<Ort::Session>(env_, path.c_str(), opts);
        });
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
    bool UsesCuda() const { return cuda_; }
    const std::string& GetError() const { return error_; }
    int32_t VocabSize() const { return vocab_size_; }
    int32_t SubsamplingFactor() const { return subsampling_factor_; }

private:
    // Creates the session, with the CUDA EP if use_gpu and it can be had.
    // onnxruntime loads the CUDA provider library when the EP is appended
    // (an unloadable library fails there, see MakeSessionOptions), and builds
    // the CUDA EP (device, cuBLAS/cuDNN handles) at session creation, where a
    // failure throws instead of falling back. So cuda_ is true exactly when a
    // session with the CUDA EP was created; on a creation failure with CUDA
    // the session is retried on the CPU alone.
    template <typename MakeSession>
    bool Create(bool use_gpu, MakeSession make) {
        try {
            Ort::SessionOptions session_opts = MakeSessionOptions(use_gpu, cuda_);
            try {
                session_ = make(session_opts);
            } catch (const Ort::Exception& e) {
                if (!cuda_)
                    throw;
                std::cerr << "[GigaAM] WARNING: session creation with the CUDA "
                             "execution provider failed: " << e.what()
                          << "; retrying on the CPU" << std::endl;
                session_opts = MakeSessionOptions(false, cuda_);
                session_ = make(session_opts);
            }

            Ort::AllocatorWithDefaultOptions allocator;
            input_name_ = session_->GetInputNameAllocated(0, allocator).get();
            output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
            ReadMetadata();
            return true;
        } catch (const Ort::Exception& e) {
            error_ = std::string("ONNX error: ") + e.what();
            std::cerr << "[GigaAM] " << error_ << std::endl;
            return false;
        }
    }

    // Makes libcudnn.so.9's symbols global before onnxruntime dlopens its CUDA
    // provider library. Arch's onnxruntime-cuda 1.29.0-3 ships a
    // libonnxruntime_providers_cuda.so without a NEEDED entry for
    // libcudnn.so.9, so loading it fails with an undefined cudnn* symbol (see
    // docs/arch-onnxruntime-cuda-cudnn-needed.md). Where the provider links
    // cuDNN properly this only loads early what it would load anyway.
    static void PreloadCudnn() {
        static const bool done = [] {
            void* h = dlopen("libcudnn.so.9", RTLD_NOW | RTLD_GLOBAL);
            if (h)
                std::cerr << "[GigaAM] Preloaded libcudnn.so.9 (RTLD_GLOBAL) for the "
                             "CUDA execution provider" << std::endl;
            else
                std::cerr << "[GigaAM] libcudnn.so.9 not preloaded: " << dlerror()
                          << std::endl;
            return true;
        }();
        (void)done;
    }

    static Ort::SessionOptions MakeSessionOptions(bool use_gpu, bool& cuda) {
        cuda = false;
        Ort::SessionOptions session_opts;
        session_opts.SetIntraOpNumThreads(4);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (use_gpu) {
            PreloadCudnn();
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
                // The first Update loads the CUDA provider library, so a
                // provider that cannot be loaded fails here, before the
                // sgemm_algo_table probe below.
                Ort::CUDAProviderOptions cuda_opts;
                cuda_opts.Update(opts);
                std::string table = SgemmTableForThisGpu();
                if (!table.empty()) {
                    auto with_table = opts;
                    with_table["sgemm_algo_table"] = table;
                    Ort::CUDAProviderOptions probe;
                    try {
                        probe.Update(with_table);
                        cuda_opts = std::move(probe);
                        opts = with_table;
                    } catch (const Ort::Exception& e) {
                        // A stock onnxruntime has no such option: run without.
                        std::cerr << "[GigaAM] this onnxruntime does not take "
                                     "sgemm_algo_table (" << e.what()
                                  << "); using cuBLAS's default algorithms" << std::endl;
                    }
                }
                session_opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
                cuda = true;
                std::cerr << "[GigaAM] Appended CUDA execution provider"
                          << (opts.count("sgemm_algo_table")
                                  ? ", SGEMM algorithms from " + opts["sgemm_algo_table"]
                                  : std::string())
                          << std::endl;
            } catch (const Ort::Exception& e) {
                std::cerr << "[GigaAM] WARNING: CUDA execution provider not available: "
                          << e.what() << "; falling back to the CPU" << std::endl;
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
    bool cuda_ = false;
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

bool OnnxModel::UsesCuda() const {
    return impl_->UsesCuda();
}

bool OnnxModel::IsLoaded() const {
    return impl_->IsLoaded();
}

}  // namespace gigaam
