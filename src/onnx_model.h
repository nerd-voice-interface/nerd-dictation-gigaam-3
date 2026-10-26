/**
 * ONNX model wrapper for GigaAM
 *
 * Simple wrapper over onnxruntime C++ API for running inference
 * on the GigaAM NeMo CTC model.
 */

#ifndef GIGAAM_ONNX_MODEL_H
#define GIGAAM_ONNX_MODEL_H

#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace gigaam {

/**
 * GigaAM model:
 * - Input: (batch=1, channels=64, time) - transposed fbank features
 * - Output: (batch=1, time/4, vocab_size=34) - log probabilities
 */
class OnnxModel {
public:
    OnnxModel();
    ~OnnxModel();

    // Non-copyable
    OnnxModel(const OnnxModel&) = delete;
    OnnxModel& operator=(const OnnxModel&) = delete;

    /**
     * Load model from memory buffer.
     *
     * @param data Pointer to ONNX model data
     * @param size Size of model data in bytes
     * @param use_gpu If true, try to use CUDA provider
     * @return true on success
     */
    bool Load(const void* data, size_t size, bool use_gpu = true);

    /**
     * Load model from file.
     *
     * @param path Path to ONNX model file
     * @param use_gpu If true, try to use CUDA provider
     * @return true on success
     */
    bool LoadFromFile(const std::string& path, bool use_gpu = true);

    /**
     * Run inference on features.
     *
     * @param features Fbank features (num_frames * 64), row-major
     * @param num_frames Number of frames
     * @return Log probabilities (output_frames * vocab_size), row-major
     *         output_frames = num_frames / subsampling_factor
     */
    std::vector<float> Run(const float* features, int32_t num_frames);

    /**
     * Get vocabulary size (34 for GigaAM: space + 32 letters + blank)
     */
    int32_t VocabSize() const { return vocab_size_; }

    /**
     * Get subsampling factor (4 for GigaAM)
     */
    int32_t SubsamplingFactor() const { return subsampling_factor_; }

    /**
     * Check if model is loaded
     */
    bool IsLoaded() const;

    /**
     * True if the loaded session has the CUDA execution provider (nodes it
     * does not support still run on the CPU provider); false if it runs on
     * the CPU provider alone.
     */
    bool UsesCuda() const;

    /**
     * Get last error message
     */
    const std::string& GetError() const { return error_; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    std::string error_;
    int32_t vocab_size_ = 34;
    int32_t subsampling_factor_ = 4;
};

}  // namespace gigaam

#endif  // GIGAAM_ONNX_MODEL_H
