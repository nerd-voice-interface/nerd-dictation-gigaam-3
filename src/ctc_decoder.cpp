/**
 * CTC Greedy Decoder implementation for GigaAM
 */

#include "ctc_decoder.h"
#include <cmath>

namespace gigaam {

// GigaAM vocabulary: space + 32 Russian letters + blank
// Index 0: space
// Index 1-32: а-я (Russian lowercase letters)
// Index 33: blank token
static const char* kTokens[] = {
    " ",   // 0: space
    "а",   // 1
    "б",   // 2
    "в",   // 3
    "г",   // 4
    "д",   // 5
    "е",   // 6
    "ж",   // 7
    "з",   // 8
    "и",   // 9
    "й",   // 10
    "к",   // 11
    "л",   // 12
    "м",   // 13
    "н",   // 14
    "о",   // 15
    "п",   // 16
    "р",   // 17
    "с",   // 18
    "т",   // 19
    "у",   // 20
    "ф",   // 21
    "х",   // 22
    "ц",   // 23
    "ч",   // 24
    "ш",   // 25
    "щ",   // 26
    "ъ",   // 27
    "ы",   // 28
    "ь",   // 29
    "э",   // 30
    "ю",   // 31
    "я",   // 32
    ""     // 33: blank (empty string for easy concatenation)
};

const char* GetToken(int32_t token_id) {
    if (token_id < 0 || token_id > 33) {
        return "";
    }
    return kTokens[token_id];
}

std::string CtcGreedyDecode(
    const float* log_probs,
    int32_t num_frames,
    int32_t vocab_size,
    int32_t blank_id
) {
    if (num_frames <= 0 || log_probs == nullptr) {
        return "";
    }

    std::string result;
    int32_t prev_token = -1;

    for (int32_t t = 0; t < num_frames; ++t) {
        // Find argmax for this frame
        const float* frame_probs = log_probs + t * vocab_size;
        int32_t best_token = 0;
        float best_prob = frame_probs[0];

        for (int32_t v = 1; v < vocab_size; ++v) {
            if (frame_probs[v] > best_prob) {
                best_prob = frame_probs[v];
                best_token = v;
            }
        }

        // CTC collapse: skip if same as previous or if blank
        if (best_token != prev_token && best_token != blank_id) {
            result += GetToken(best_token);
        }

        prev_token = best_token;
    }

    return result;
}

}  // namespace gigaam
