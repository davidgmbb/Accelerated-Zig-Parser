// A C port of the 512-byte-chunk compacting Zig tokenizer in src/CompactingTokenizer.zig.
// Requires Clang and AVX-512 F/BW/VL/CD/VBMI/VBMI2/VPOPCNTDQ plus BMI2 (e.g. -march=znver4, znver5 or tigerlake).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "compacting_tokenizer_tags.h"

#ifdef __cplusplus
extern "C" {
#endif

enum azp_error {
    AZP_ERROR_INVALID_CARRIAGE_RETURN = -1, // A '\r' that isn't followed by '\n'.
    AZP_ERROR_INVALID_UTF8 = -2,
    AZP_ERROR_FILE_TOO_BIG = -3, // Token positions are 32-bit.
};

/// Tokenizes Zig source code. `kinds`, `starts` and `ends` must each have room for `len + 64` entries.
/// Token `i` is `source[starts[i] .. ends[i]]` with kind `kinds[i]` (an `enum azp_tag`). Whitespace isn't emitted;
/// comments are (`AZP_TAG_COMMENT`). Returns the number of tokens, or a negative `enum azp_error`.
ptrdiff_t azp_tokenize(const uint8_t *source, size_t len, uint8_t *kinds, uint32_t *starts, uint32_t *ends);

/// The name of a tag as in the Zig implementation, e.g. "identifier" or "+=".
const char *azp_tag_name(uint8_t tag);

#ifdef __cplusplus
}
#endif
