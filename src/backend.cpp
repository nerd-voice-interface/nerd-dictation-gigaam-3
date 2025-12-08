/**
 * GigaAM ASR Backend - C API Implementation
 *
 * Uses sherpa-onnx with GigaAM NeMo CTC model for Russian language.
 * The model can be loaded from:
 *   1. Embedded compressed volumes (decompressed at runtime)
 *   2. External file system path
 *
 * Note: GigaAM is an offline (non-streaming) model. Audio is accumulated
 * and processed on finalize() call.
 */

#include "nerd_dictation/backend_api.h"
#include <sherpa-onnx/c-api/c-api.h>
#include "model_loader.h"
#include "models/gigaam-v3-ctc/tokens.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <mutex>

#ifndef GIGAAM_MODEL_DIR
#define GIGAAM_MODEL_DIR "/usr/share/nerd-dictation-gigaam/model"
#endif

#ifndef GIGAAM_TOKENS_PATH
#define GIGAAM_TOKENS_PATH GIGAAM_MODEL_DIR "/tokens.txt"
#endif

/* Get model directory from environment or default */
static std::string get_model_dir() {
    const char* env_dir = std::getenv("GIGAAM_MODEL_DIR");
    if (env_dir && env_dir[0] != '\0') {
        return std::string(env_dir);
    }
    return GIGAAM_MODEL_DIR;
}

/* Internal backend structure */
struct nerd_dictation_backend {
    const SherpaOnnxOfflineRecognizer* recognizer;

    int sample_rate;

    /* Audio accumulator buffer (offline model processes all at once) */
    std::vector<float> audio_buffer;

    /* Result buffer (owned, reused between calls) */
    std::string result_text;
    std::string error_message;

    /* Model paths (for file-based loading) */
    std::string model_path;
    std::string tokens_path;

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

    backend->recognizer = nullptr;
    backend->sample_rate = 16000;  /* GigaAM uses 16kHz */
    backend->embedded_model_data = nullptr;
    backend->embedded_model_size = 0;

    /* Configure sherpa-onnx offline recognizer */
    SherpaOnnxOfflineRecognizerConfig config;
    memset(&config, 0, sizeof(config));

    /* Feature config */
    config.feat_config.sample_rate = backend->sample_rate;
    config.feat_config.feature_dim = 64;  /* GigaAM v3 uses 64-dim features */

    bool use_embedded = gigaam::has_embedded_model();

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

        /* Use model_buf for buffer-based loading */
        config.model_config.nemo_ctc.model = "";  /* No file path */
        config.model_config.nemo_ctc.model_buf = reinterpret_cast<const char*>(backend->embedded_model_data);
        config.model_config.nemo_ctc.model_buf_size = static_cast<int32_t>(backend->embedded_model_size);

        /* Use embedded tokens */
        config.model_config.tokens = "";  /* No file path */
        config.model_config.tokens_buf = gigaam::TOKENS_DATA;
        config.model_config.tokens_buf_size = gigaam::TOKENS_SIZE;
    } else {
        /* Fall back to file-based loading */
        std::string model_dir = get_model_dir();
        backend->tokens_path = model_dir + "/tokens.txt";

        /* Try model.onnx first (fp32), then model.int8.onnx (quantized) */
        backend->model_path = model_dir + "/model.onnx";
        FILE* f_check = fopen(backend->model_path.c_str(), "r");
        if (!f_check) {
            backend->model_path = model_dir + "/model.int8.onnx";
        } else {
            fclose(f_check);
        }

        std::cerr << "[GigaAM] Loading model from file: " << backend->model_path << std::endl;

        /* Check if model file exists */
        FILE* f = fopen(backend->model_path.c_str(), "r");
        if (!f) {
            backend->error_message = "GigaAM: model file not found: " + backend->model_path;
            std::cerr << "[GigaAM] ERROR: " << backend->error_message << std::endl;
            delete backend;
            return nullptr;
        }
        fclose(f);

        config.model_config.nemo_ctc.model = backend->model_path.c_str();
    }

    /* Check if tokens file exists (only for file-based loading) */
    if (!use_embedded) {
        FILE* f = fopen(backend->tokens_path.c_str(), "r");
        if (!f) {
            backend->error_message = "GigaAM: tokens file not found: " + backend->tokens_path;
            std::cerr << "[GigaAM] ERROR: " << backend->error_message << std::endl;
            delete backend;
            return nullptr;
        }
        fclose(f);

        /* Tokens from file */
        config.model_config.tokens = backend->tokens_path.c_str();
    }

    /* General model config */
    config.model_config.num_threads = 4;
    config.model_config.provider = "cuda";
    config.model_config.debug = 0;
    config.model_config.model_type = "nemo_ctc";  /* Skip auto-detection */

    /* Decoding config */
    config.decoding_method = "greedy_search";
    config.max_active_paths = 4;

    /* Create recognizer */
    backend->recognizer = SherpaOnnxCreateOfflineRecognizer(&config);
    if (!backend->recognizer) {
        backend->error_message = "Failed to create GigaAM recognizer";
        std::cerr << "[GigaAM] ERROR: " << backend->error_message << std::endl;
        if (backend->embedded_model_data) {
            gigaam::free_embedded_model(backend->embedded_model_data, backend->embedded_model_size);
        }
        delete backend;
        return nullptr;
    }

    std::cerr << "[GigaAM] Model loaded successfully (sample rate: "
              << backend->sample_rate << " Hz, offline mode, "
              << (use_embedded ? "embedded" : "file") << ")" << std::endl;

    return backend;
}

extern "C" void nerd_dictation_destroy(nerd_dictation_t backend) {
    if (!backend) return;

    if (backend->recognizer) {
        SherpaOnnxDestroyOfflineRecognizer(backend->recognizer);
    }

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
    if (!backend || !backend->recognizer) return -1;
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
    if (!backend->recognizer) return -2;

    /* For offline model, we can't provide intermediate results */
    result->text = "";
    result->is_final = 0;
    result->confidence = -1.0f;

    return 0;
}

extern "C" int nerd_dictation_finalize(nerd_dictation_t backend, struct nerd_dictation_result* result) {
    if (!backend || !result) return -1;
    if (!backend->recognizer) return -2;

    std::lock_guard<std::mutex> lock(backend->buffer_mutex);

    if (backend->audio_buffer.empty()) {
        backend->result_text.clear();
        result->text = backend->result_text.c_str();
        result->is_final = 1;
        result->confidence = -1.0f;
        return 0;
    }

    std::cerr << "[GigaAM] Processing " << backend->audio_buffer.size()
              << " samples (" << (float)backend->audio_buffer.size() / backend->sample_rate
              << " seconds)" << std::endl;

    /* Create stream for this utterance */
    const SherpaOnnxOfflineStream* stream = SherpaOnnxCreateOfflineStream(backend->recognizer);
    if (!stream) {
        backend->error_message = "Failed to create offline stream";
        std::cerr << "[GigaAM] ERROR: " << backend->error_message << std::endl;
        return -3;
    }

    /* Accept all accumulated audio */
    SherpaOnnxAcceptWaveformOffline(
        stream,
        backend->sample_rate,
        backend->audio_buffer.data(),
        static_cast<int32_t>(backend->audio_buffer.size())
    );

    /* Decode */
    SherpaOnnxDecodeOfflineStream(backend->recognizer, stream);

    /* Get result */
    const SherpaOnnxOfflineRecognizerResult* sherpa_result =
        SherpaOnnxGetOfflineStreamResult(stream);

    if (sherpa_result && sherpa_result->text && strlen(sherpa_result->text) > 0) {
        backend->result_text = sherpa_result->text;
        std::cerr << "[GigaAM] Result: '" << backend->result_text << "'" << std::endl;
    } else {
        backend->result_text.clear();
    }

    result->text = backend->result_text.c_str();
    result->is_final = 1;
    result->confidence = -1.0f;

    /* Cleanup */
    if (sherpa_result) {
        SherpaOnnxDestroyOfflineRecognizerResult(sherpa_result);
    }
    SherpaOnnxDestroyOfflineStream(stream);

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
    if (!backend->recognizer) return -2;

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
