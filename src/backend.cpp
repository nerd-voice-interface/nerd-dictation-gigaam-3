/**
 * GigaAM ASR Backend - C API Implementation
 *
 * Uses direct ONNX inference with GigaAM NeMo CTC model for Russian language.
 * The model can be loaded from:
 *   1. Embedded compressed volumes (decompressed at runtime)
 *   2. External file system path
 *
 * Note: GigaAM is an offline (non-streaming) model. Audio is accumulated
 * and processed on finalize() call.
 */

#include "nerd_dictation/backend_api.h"
#include "fbank.h"
#include "onnx_model.h"
#include "ctc_decoder.h"
#include "model_loader.h"
#include "models/gigaam-v3-ctc/tokens.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <mutex>
#include <memory>

#ifndef GIGAAM_MODEL_DIR
#define GIGAAM_MODEL_DIR "/usr/share/nerd-dictation-gigaam/model"
#endif

/* Get model directory from environment or default */
static std::string get_model_dir() {
    const char* env_dir = std::getenv("GIGAAM_MODEL_DIR");
    if (env_dir && env_dir[0] != '\0') {
        return std::string(env_dir);
    }
    return GIGAAM_MODEL_DIR;
}

/* Check if GPU should be used */
static bool should_use_gpu() {
    const char* env = std::getenv("GIGAAM_USE_GPU");
    if (env) {
        return strcmp(env, "0") != 0 && strcmp(env, "false") != 0;
    }
    return true;  /* Default: try GPU */
}

/* Internal backend structure */
struct nerd_dictation_backend {
    std::unique_ptr<gigaam::FbankExtractor> fbank;
    std::unique_ptr<gigaam::OnnxModel> model;

    int sample_rate;

    /* Audio accumulator buffer (offline model processes all at once) */
    std::vector<float> audio_buffer;

    /* Result buffer (owned, reused between calls) */
    std::string result_text;
    std::string error_message;

    /* Model paths (for file-based loading) */
    std::string model_path;

    /* Embedded model data (for buffer-based loading) */
    const uint8_t* embedded_model_data;
    size_t embedded_model_size;

    /* Thread safety */
    std::mutex buffer_mutex;
};

/* ============================================================================
 * Lifecycle functions
 * ============================================================================ */

extern "C" nerd_dictation_t nerd_dictation_create(void) {
    auto* backend = new (std::nothrow) nerd_dictation_backend();
    if (!backend) {
        return nullptr;
    }

    backend->sample_rate = 16000;  /* GigaAM uses 16kHz */
    backend->embedded_model_data = nullptr;
    backend->embedded_model_size = 0;

    /* Create feature extractor */
    backend->fbank = std::make_unique<gigaam::FbankExtractor>();

    /* Create ONNX model */
    backend->model = std::make_unique<gigaam::OnnxModel>();

    bool use_embedded = gigaam::has_embedded_model();
    bool use_gpu = should_use_gpu();

    if (use_embedded) {
        /* Load embedded model via parallel decompression */
        std::cerr << "[GigaAM] Loading embedded model (parallel zstd decompression)..." << std::endl;

        int ret = gigaam::load_embedded_model(
            &backend->embedded_model_data,
            &backend->embedded_model_size,
            8  /* 8 threads */
        );

        if (ret != 0 || !backend->embedded_model_data) {
            backend->error_message = "Failed to decompress embedded model";
            std::cerr << "[GigaAM] ERROR: " << backend->error_message << std::endl;
            delete backend;
            return nullptr;
        }

        std::cerr << "[GigaAM] Decompressed " << (backend->embedded_model_size / 1024 / 1024)
                  << " MB model into memory" << std::endl;

        /* Load model from buffer */
        if (!backend->model->Load(backend->embedded_model_data, backend->embedded_model_size, use_gpu)) {
            backend->error_message = "Failed to load ONNX model: " + backend->model->GetError();
            std::cerr << "[GigaAM] ERROR: " << backend->error_message << std::endl;
            gigaam::free_embedded_model(backend->embedded_model_data, backend->embedded_model_size);
            delete backend;
            return nullptr;
        }
    } else {
        /* Fall back to file-based loading */
        std::string model_dir = get_model_dir();

        /* Try model.onnx first (fp32), then model.int8.onnx (quantized) */
        backend->model_path = model_dir + "/model.onnx";
        FILE* f_check = fopen(backend->model_path.c_str(), "r");
        if (!f_check) {
            backend->model_path = model_dir + "/model.int8.onnx";
        } else {
            fclose(f_check);
        }

        std::cerr << "[GigaAM] Loading model from file: " << backend->model_path << std::endl;

        /* Load model from file */
        if (!backend->model->LoadFromFile(backend->model_path, use_gpu)) {
            backend->error_message = "Failed to load model: " + backend->model->GetError();
            std::cerr << "[GigaAM] ERROR: " << backend->error_message << std::endl;
            delete backend;
            return nullptr;
        }
    }

    std::cerr << "[GigaAM] Model loaded successfully (sample rate: "
              << backend->sample_rate << " Hz, offline mode, "
              << (use_embedded ? "embedded" : "file") << ", "
              << (use_gpu ? "GPU" : "CPU") << ")" << std::endl;

    return backend;
}

extern "C" void nerd_dictation_destroy(nerd_dictation_t backend) {
    if (!backend) return;

    /* Free embedded model memory if allocated */
    if (backend->embedded_model_data) {
        gigaam::free_embedded_model(backend->embedded_model_data, backend->embedded_model_size);
    }

    delete backend;
}

extern "C" int nerd_dictation_get_info(nerd_dictation_t backend, struct nerd_dictation_info* info) {
    if (!backend || !info) return -1;
    if (info->struct_size < sizeof(struct nerd_dictation_info)) return -2;

    info->name = "gigaam";
    info->streaming = 0;  /* Offline model - no true streaming */
    info->sample_rate = backend->sample_rate;
    info->channels = 1;
    info->bits_per_sample = 16;

    return 0;
}

/* ============================================================================
 * Recognition functions
 * ============================================================================ */

extern "C" int nerd_dictation_feed(nerd_dictation_t backend, const int16_t* samples, size_t num_samples) {
    if (!backend || !backend->model || !backend->model->IsLoaded()) return -1;
    if (!samples || num_samples == 0) return 0;

    std::lock_guard<std::mutex> lock(backend->buffer_mutex);

    /* Convert int16 to float [-1, 1] and append to buffer */
    size_t old_size = backend->audio_buffer.size();
    backend->audio_buffer.resize(old_size + num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        backend->audio_buffer[old_size + i] = samples[i] / 32768.0f;
    }

    return 0;
}

extern "C" int nerd_dictation_result(nerd_dictation_t backend, struct nerd_dictation_result* result) {
    if (!backend || !result) return -1;
    if (!backend->model || !backend->model->IsLoaded()) return -2;

    /* For offline model, we can't provide intermediate results */
    result->text = "";
    result->is_final = 0;
    result->confidence = -1.0f;

    return 0;
}

extern "C" int nerd_dictation_finalize(nerd_dictation_t backend, struct nerd_dictation_result* result) {
    if (!backend || !result) return -1;
    if (!backend->model || !backend->model->IsLoaded()) return -2;

    std::lock_guard<std::mutex> lock(backend->buffer_mutex);

    if (backend->audio_buffer.empty()) {
        backend->result_text.clear();
        result->text = backend->result_text.c_str();
        result->is_final = 1;
        result->confidence = -1.0f;
        return 0;
    }

    float duration = static_cast<float>(backend->audio_buffer.size()) / backend->sample_rate;
    std::cerr << "[GigaAM] Processing " << backend->audio_buffer.size()
              << " samples (" << duration << " seconds)" << std::endl;

    /* Extract fbank features */
    std::vector<float> features = backend->fbank->Extract(
        backend->audio_buffer.data(),
        backend->audio_buffer.size()
    );

    if (features.empty()) {
        backend->error_message = "Failed to extract features";
        std::cerr << "[GigaAM] ERROR: " << backend->error_message << std::endl;
        return -3;
    }

    int32_t num_frames = static_cast<int32_t>(features.size() / 64);
    std::cerr << "[GigaAM] Extracted " << num_frames << " frames" << std::endl;

    /* Run inference */
    std::vector<float> log_probs = backend->model->Run(features.data(), num_frames);

    if (log_probs.empty()) {
        backend->error_message = "Inference failed: " + backend->model->GetError();
        std::cerr << "[GigaAM] ERROR: " << backend->error_message << std::endl;
        return -4;
    }

    /* Calculate output frames (subsampling factor = 4) */
    int32_t output_frames = static_cast<int32_t>(log_probs.size() / backend->model->VocabSize());
    std::cerr << "[GigaAM] Got " << output_frames << " output frames" << std::endl;

    /* Decode */
    backend->result_text = gigaam::CtcGreedyDecode(
        log_probs.data(),
        output_frames,
        backend->model->VocabSize()
    );

    if (!backend->result_text.empty()) {
        std::cerr << "[GigaAM] Result: '" << backend->result_text << "'" << std::endl;
    }

    result->text = backend->result_text.c_str();
    result->is_final = 1;
    result->confidence = -1.0f;

    /* Clear buffer for next utterance */
    backend->audio_buffer.clear();

    return 0;
}

extern "C" int nerd_dictation_reset(nerd_dictation_t backend) {
    if (!backend) return -1;

    std::lock_guard<std::mutex> lock(backend->buffer_mutex);
    backend->audio_buffer.clear();
    backend->result_text.clear();

    return 0;
}

/* ============================================================================
 * Streaming API
 * ============================================================================ */

extern "C" int nerd_dictation_transcribe_stream(
    nerd_dictation_t backend,
    nerd_dictation_audio_callback audio_cb,
    nerd_dictation_result_callback result_cb,
    void* user_data
) {
    if (!backend || !audio_cb) return -1;
    if (!backend->model || !backend->model->IsLoaded()) return -2;

    std::cerr << "[GigaAM] Starting batch recognition (offline mode)" << std::endl;

    /* Audio buffer for callback */
    std::vector<int16_t> audio_chunk(backend->sample_rate / 10);  /* 100ms chunks */
    size_t chunk_count = 0;

    /* Accumulate all audio first */
    while (true) {
        int samples_read = audio_cb(audio_chunk.data(), audio_chunk.size(), user_data);

        if (samples_read < 0) {
            backend->error_message = "Audio callback returned error";
            return -3;
        }

        if (samples_read == 0) {
            break;
        }

        chunk_count++;

        /* Feed audio to accumulator */
        int ret = nerd_dictation_feed(backend, audio_chunk.data(), static_cast<size_t>(samples_read));
        if (ret < 0) {
            return ret;
        }
    }

    /* Process all accumulated audio */
    struct nerd_dictation_result final_result;
    int ret = nerd_dictation_finalize(backend, &final_result);

    if (ret == 0 && result_cb && final_result.text && strlen(final_result.text) > 0) {
        result_cb(&final_result, user_data);
    }

    std::cerr << "[GigaAM] Batch recognition finished (processed "
              << chunk_count << " chunks)" << std::endl;

    return 0;
}

/* ============================================================================
 * Utilities
 * ============================================================================ */

extern "C" const char* nerd_dictation_get_error(nerd_dictation_t backend) {
    if (!backend) return nullptr;
    return backend->error_message.empty() ? nullptr : backend->error_message.c_str();
}

extern "C" int nerd_dictation_api_version(void) {
    return NERD_DICTATION_API_VERSION;
}
