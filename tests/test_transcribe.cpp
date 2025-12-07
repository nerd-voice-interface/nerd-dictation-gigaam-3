/**
 * GigaAM Backend Test - C API
 */

#include "nerd_dictation/backend_api.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cctype>

/* Trim whitespace from both ends of string */
static std::string trim(const std::string& s) {
    auto start = std::find_if_not(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
    auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) { return std::isspace(c); }).base();
    return (start < end) ? std::string(start, end) : std::string();
}

/* Combined context for both callbacks */
struct StreamContext {
    /* Audio state */
    const int16_t* samples;
    size_t num_samples;
    size_t offset;
    size_t chunk_size;

    /* Result state */
    std::string final_text;
    std::string last_partial;
};

/* Audio callback for streaming API */
static int audio_callback(int16_t* buffer, size_t max_samples, void* user_data) {
    auto* ctx = static_cast<StreamContext*>(user_data);

    if (ctx->offset >= ctx->num_samples) {
        return 0;  /* End of stream */
    }

    size_t remaining = ctx->num_samples - ctx->offset;
    size_t to_read = std::min(std::min(ctx->chunk_size, remaining), max_samples);

    memcpy(buffer, ctx->samples + ctx->offset, to_read * sizeof(int16_t));
    ctx->offset += to_read;

    return static_cast<int>(to_read);
}

/* Result callback */
static void result_callback(const struct nerd_dictation_result* result, void* user_data) {
    auto* ctx = static_cast<StreamContext*>(user_data);

    if (result->is_final) {
        ctx->final_text = result->text ? result->text : "";
        std::cout << "[FINAL] " << ctx->final_text << std::endl;
    } else if (result->text) {
        ctx->last_partial = result->text;
        std::cout << "[partial] " << result->text << std::endl;
    }
}

int main(int argc, char* argv[]) {
    const char* wav_path = nullptr;

    /* Default test file path */
    std::string default_path;
#ifdef GIGAAM_MODEL_DIR
    default_path = std::string(GIGAAM_MODEL_DIR) + "/test_wavs/example.wav";
#else
    default_path = "/usr/share/nerd-dictation-gigaam/model/test_wavs/example.wav";
#endif

    if (argc < 2) {
        wav_path = default_path.c_str();
        std::cout << "Using default test file: " << wav_path << std::endl;
    } else {
        wav_path = argv[1];
    }

    std::cout << "=== GigaAM Backend Test (C API) ===" << std::endl;
    std::cout << "Input file: " << wav_path << std::endl;

    /* Read WAV file */
    std::ifstream file(wav_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << wav_path << std::endl;
        return 1;
    }

    std::vector<char> buffer((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    file.close();

    if (buffer.size() < 44) {
        std::cerr << "Error: Invalid WAV file (too small)" << std::endl;
        return 1;
    }

    /* Parse WAV header */
    uint16_t num_channels = *reinterpret_cast<uint16_t*>(&buffer[22]);
    uint32_t sample_rate = *reinterpret_cast<uint32_t*>(&buffer[24]);
    uint16_t bits_per_sample = *reinterpret_cast<uint16_t*>(&buffer[34]);

    std::cout << "WAV info: " << sample_rate << " Hz, " << bits_per_sample << " bit, " << num_channels << " ch" << std::endl;

    if (bits_per_sample != 16 || num_channels != 1) {
        std::cerr << "Error: Only 16-bit mono WAV supported" << std::endl;
        return 1;
    }

    /* Extract audio data */
    size_t data_start = 44;
    const int16_t* audio_data = reinterpret_cast<const int16_t*>(buffer.data() + data_start);
    size_t num_samples = (buffer.size() - data_start) / sizeof(int16_t);

    std::cout << "Loaded " << num_samples << " samples" << std::endl;

    /* GigaAM model uses 16kHz */
    if (sample_rate != 16000) {
        std::cerr << "Warning: GigaAM model expects 16kHz audio, got " << sample_rate << " Hz" << std::endl;
    }

    /* Create backend using C API */
    std::cout << "\nCreating GigaAM backend..." << std::endl;

    nerd_dictation_t backend = nerd_dictation_create();
    if (!backend) {
        std::cerr << "Error: Failed to create backend" << std::endl;
        return 1;
    }

    /* Get backend info */
    struct nerd_dictation_info info;
    info.struct_size = sizeof(info);

    if (nerd_dictation_get_info(backend, &info) == 0) {
        std::cout << "Backend: " << info.name << std::endl;
        std::cout << "Streaming: " << (info.streaming ? "yes" : "no (offline)") << std::endl;
        std::cout << "Sample rate: " << info.sample_rate << " Hz" << std::endl;
    }

    /* Transcribe using streaming API (actually batch for offline model) */
    std::cout << "\nTranscribing..." << std::endl;

    StreamContext stream_ctx;
    stream_ctx.samples = audio_data;
    stream_ctx.num_samples = num_samples;
    stream_ctx.offset = 0;
    stream_ctx.chunk_size = 1600;  /* 100ms at 16kHz */

    int ret = nerd_dictation_transcribe_stream(
        backend,
        audio_callback,
        result_callback,
        &stream_ctx
    );

    if (ret < 0) {
        std::cerr << "Error: nerd_dictation_transcribe_stream returned " << ret << std::endl;
    }

    /* Get result */
    std::string result = stream_ctx.final_text;

    /* Cleanup */
    nerd_dictation_destroy(backend);

    /* Output result */
    std::cout << "\n=== Result ===" << std::endl;
    std::cout << "Transcription: \"" << result << "\"" << std::endl;

    /* Basic validation */
    if (result.empty()) {
        std::cerr << "\n[FAIL] No transcription result" << std::endl;
        return 1;
    }

    std::cout << "\n[PASS] Got transcription result" << std::endl;
    return 0;
}
