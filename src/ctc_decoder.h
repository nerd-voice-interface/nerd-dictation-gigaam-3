/**
 * CTC Greedy Decoder for GigaAM
 *
 * Simple greedy decoding: argmax per frame, collapse repeated, remove blank.
 */

#ifndef GIGAAM_CTC_DECODER_H
#define GIGAAM_CTC_DECODER_H

#include <string>
#include <vector>
#include <cstdint>

namespace gigaam {

/**
 * GigaAM vocabulary (34 tokens):
 * - 0: space " "
 * - 1-32: Russian letters а-я
 * - 33: blank <blk>
 */

/**
 * Decode log probabilities to text using greedy CTC decoding.
 *
 * @param log_probs Log probabilities, shape (num_frames, vocab_size)
 * @param num_frames Number of output frames
 * @param vocab_size Vocabulary size (34 for GigaAM)
 * @param blank_id Blank token ID (33 for GigaAM)
 * @return Decoded Russian text
 */
std::string CtcGreedyDecode(
    const float* log_probs,
    int32_t num_frames,
    int32_t vocab_size = 34,
    int32_t blank_id = 33
);

/**
 * Get token string for given token ID.
 * Used internally by decoder.
 */
const char* GetToken(int32_t token_id);

}  // namespace gigaam

#endif  // GIGAAM_CTC_DECODER_H
