// A C port of the 512-byte-chunk compacting Zig tokenizer in src/CompactingTokenizer.zig. The structure, names and
// comments follow the Zig version; see it for the full explanations.
//
// It processes 512-byte chunks as eight 64-byte vectors, with a 512-bit bitstring per character class (`u64x8`,
// lane 0 holding the lowest bits). There is one implementation for every target: it only needs the AVX-512 subset
// that Zen 4, Zen 5 and Tiger Lake all have.
#include "compacting_tokenizer.h"

#if !defined(__clang__)
#error "compacting_tokenizer.c uses Clang vector extensions"
#endif
#if !(defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512CD__) && \
      defined(__AVX512VBMI__) && defined(__AVX512VBMI2__) && defined(__AVX512VPOPCNTDQ__) && defined(__BMI2__))
#error "compacting_tokenizer.c needs AVX-512 F/BW/VL/CD/VBMI/VBMI2/VPOPCNTDQ and BMI2 (e.g. -march=znver4, znver5 or tigerlake)"
#endif

#include <immintrin.h>
#include <string.h>

#include "compacting_tokenizer_tables.h"

#define INLINE static inline __attribute__((always_inline))

typedef uint64_t u64x8 __attribute__((ext_vector_type(8)));
typedef uint8_t u8x64 __attribute__((ext_vector_type(64)));
typedef uint8_t u8x16 __attribute__((ext_vector_type(16)));

static const uint8_t iota_bytes[64] __attribute__((aligned(64))) = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
};

// ---------------------------------------------------------------------------------------------------------------------
// 512-bit bitstring helpers (`shl`, `shr`, `shld`, `sub`, ... in the Zig version)

// The 512-bit shifts `a << i` and `a >> i`, for a constant 0 < i < 64.
INLINE u64x8 shl(u64x8 a, int i) { return (__builtin_shufflevector(a, (u64x8){0}, 8, 0, 1, 2, 3, 4, 5, 6) >> (64 - i)) | (a << i); }
INLINE u64x8 shr(u64x8 a, int i) { return (__builtin_shufflevector(a, (u64x8){0}, 1, 2, 3, 4, 5, 6, 7, 8) << (64 - i)) | (a >> i); }

// `a << 1`, shifting in the top bit of `b`.
INLINE u64x8 shld1(u64x8 a, u64x8 b) { return (__builtin_shufflevector(a, b, 15, 0, 1, 2, 3, 4, 5, 6) >> 63) | (a << 1); }

// Each lane gets the previous lane of `cur`, and lane 0 gets the last lane of `prev` (`std.simd.mergeShift(prev, cur, 7)`).
INLINE u64x8 prev_lanes(u64x8 prev, u64x8 cur) { return __builtin_shufflevector(cur, prev, 15, 0, 1, 2, 3, 4, 5, 6); }

// 512-bit subtraction: a lane borrows when the lane below it borrowed out (Kogge-Stone, propagated in a mask register).
INLINE u64x8 sub(u64x8 a, u64x8 b) {
    const uint8_t carries = _mm512_cmplt_epu64_mask((__m512i)a, (__m512i)b);
    const uint8_t equal = _mm512_cmpeq_epi64_mask((__m512i)a, (__m512i)b);
    const uint8_t borrows = equal ^ (uint8_t)((carries << 1) + equal);
    const u64x8 s = a - b;
    return (u64x8)_mm512_mask_add_epi64((__m512i)s, borrows, (__m512i)s, _mm512_set1_epi64(-1));
}

INLINE int is_zero(u64x8 x) { return _mm512_test_epi64_mask((__m512i)x, (__m512i)x) == 0; }

// The last bit of a 512-bit bitstring, moved to bit 0.
INLINE u64x8 top_bit(u64x8 x) { return (u64x8){x[7] >> 63, 0, 0, 0, 0, 0, 0, 0}; }

// The highest set bit of a 512-bit bitstring (or 0).
INLINE u64x8 highest_bit(u64x8 x) {
    const uint8_t nonzero_lanes = _mm512_test_epi64_mask((__m512i)x, (__m512i)x);
    u64x8 result = {0};
    if (nonzero_lanes == 0) return result;
    const int lane = 31 - __builtin_clz(nonzero_lanes);
    result[lane] = 1ull << (63 - __builtin_clzll(x[lane]));
    return result;
}

// ---------------------------------------------------------------------------------------------------------------------
// Vector helpers

INLINE __m512i splat8(uint8_t x) { return _mm512_set1_epi8((char)x); }

// Looks up 64 bytes in a 16-byte table (repeated across the vector). Indices >= 0x80 give 0.
INLINE __m512i vpshufb(const uint8_t table[16], __m512i indices) {
    return _mm512_shuffle_epi8(_mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)table)), indices);
}

// Looks up 64 bytes in a 128-byte table, using the low 7 bits of each index.
INLINE __m512i perm2b(const uint8_t table[128], __m512i indices) {
    return _mm512_permutex2var_epi8(_mm512_load_si512(table), indices, _mm512_load_si512(table + 64));
}

// Looks up 64 bytes in a 256-byte table.
INLINE __m512i perm4b(const uint8_t table[256], __m512i indices) {
    return _mm512_mask_blend_epi8(_mm512_cmplt_epu8_mask(indices, splat8(0x80)), perm2b(table + 128, indices), perm2b(table, indices));
}

// Looks up bytes in `cur` (indices 0-63) followed by `prev` (indices 64-127, i.e. negative indices).
INLINE __m512i lookup_cur_prev(__m512i cur, __m512i prev, __m512i indices) { return _mm512_permutex2var_epi8(cur, indices, prev); }

// The indices of the set bits of `mask`, in order (`bitsToIndices`).
INLINE __m512i bits_to_indices(uint64_t mask) { return _mm512_maskz_compress_epi8(mask, _mm512_load_si512(iota_bytes)); }

// `cur` shifted up by 1 or 2 bytes, shifting in the last bytes of `prev` (`shiftInPrevious`).
INLINE __m512i shift_in_previous_1(__m512i cur, __m512i prev) {
    return (__m512i)__builtin_shufflevector((u8x64)cur, (u8x64)prev, 127, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62);
}
INLINE __m512i shift_in_previous_2(__m512i cur, __m512i prev) {
    return (__m512i)__builtin_shufflevector((u8x64)cur, (u8x64)prev, 126, 127, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61);
}

// Loads `len` bytes (all 64 if `len >= 64`) without touching memory past them, padding the rest with `padding`.
INLINE __m512i safe_load(const uint8_t *ptr, size_t len, uint8_t padding) {
    const uint64_t mask = len >= 64 ? ~0ull : _bzhi_u64(~0ull, (uint32_t)len);
    return _mm512_mask_loadu_epi8(splat8(padding), mask, ptr);
}

// Hashes up to 16 multi-char symbols that end in this 64-byte chunk to their tag (`getMultiCharKinds`).
INLINE __m128i get_multi_char_kinds(__m512i prev_chunk, __m512i chunk, __m128i end_indices, uint16_t triples) {
    // Per symbol, the offsets of its characters before its end: { 2, 1, 0 } or { 1, 0 }; 0x80 zeroes the byte.
    const __m512i adjustment = _mm512_mask_blend_epi32(triples, _mm512_set1_epi32(0x80800001), _mm512_set1_epi32(0x80000102));
    const u8x16 e = (u8x16)end_indices;
    const u8x64 repeated = __builtin_shufflevector(e, e, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6,
                                                   6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12,
                                                   12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15);
    const __m512i successive_indices = _mm512_sub_epi8((__m512i)repeated, adjustment);
    const __m512i strs = _mm512_maskz_permutex2var_epi8(_mm512_cmplt_epu8_mask(adjustment, splat8(0x80)), chunk, successive_indices, prev_chunk);

    __m128i hash = _mm512_cvtepi32_epi8(_mm512_srli_epi32(_mm512_mullo_epi32(strs, _mm512_set1_epi32(698839068)), 25));
    hash = _mm_and_si128(_mm_sub_epi8(hash, _mm_set1_epi8(29)), _mm_set1_epi8(0x7F));
    return _mm_mask_add_epi8(hash, _mm_cmplt_epu8_mask(hash, _mm_set1_epi8(16)), hash, _mm_set1_epi8(18));
}

// The same check as std.unicode.utf8ValidateSlice.
static int utf8_is_valid(const uint8_t *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        const uint8_t c = s[i];
        if (c < 0x80) {
            i += 1;
            continue;
        }
        size_t len;
        uint32_t cp;
        if (c >= 0xC2 && c <= 0xDF) len = 2, cp = c & 0x1F;
        else if ((c & 0xF0) == 0xE0) len = 3, cp = c & 0x0F;
        else if (c >= 0xF0 && c <= 0xF4) len = 4, cp = c & 0x07;
        else return 0;
        if (n - i < len) return 0;
        for (size_t k = 1; k < len; k++) {
            if ((s[i + k] & 0xC0) != 0x80) return 0;
            cp = (cp << 6) | (s[i + k] & 0x3F);
        }
        if (len == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) return 0;
        if (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)) return 0;
        i += len;
    }
    return 1;
}

const char *azp_tag_name(uint8_t tag) { return azp_tag_names[tag] ? azp_tag_names[tag] : "<invalid kind>"; }

// ---------------------------------------------------------------------------------------------------------------------

enum {
    CARRIAGES,
    SLASHES,
    BACKSLASHES,
    NEWLINES_OR_EOF,
    QUOTES,
    APOSTROPHES,
    BAD_CONTROLS,
    ATS,
    WHITESPACES,
    ALPHA_UNDERSCORES,
    ALPHA_NUMERIC_UNDERSCORES,
    POSSIBLE_ONE_CHAR_SYMBOLS,
    PRELIM_DOUBLE_CHAR_ENDS,
    PRELIM_TRIPLE_CHAR_ENDS,
    NUM_MASKS,
};

// The bytes of `Channels{ .self = 1 }` and `Channels{ .self = 0 }`: every channel, with or without `self`.
#define ALL_CHANNELS 0xFF
#define ALL_CHANNELS_BUT_SELF 0x7F

// Phase III picks 8 keyword candidates at a time: the `k`th group of 8 even-indexed bytes, each repeated 8 times.
#define REPEAT8(x) x, x, x, x, x, x, x, x
#define GROUP_OF_8(k) \
    REPEAT8(16 * (k) + 0), REPEAT8(16 * (k) + 2), REPEAT8(16 * (k) + 4), REPEAT8(16 * (k) + 6), \
    REPEAT8(16 * (k) + 8), REPEAT8(16 * (k) + 10), REPEAT8(16 * (k) + 12), REPEAT8(16 * (k) + 14)

ptrdiff_t azp_tokenize(const uint8_t *source, size_t source_len, uint8_t *restrict kinds_out, uint32_t *restrict starts_out,
                       uint32_t *restrict ends_out) {
    if (source_len > UINT32_MAX - 64) return AZP_ERROR_FILE_TOO_BIG;

    const u64x8 zero = {0};
    const u64x8 MSb = {0, 0, 0, 0, 0, 0, 0, 1ull << 63};
    const u64x8 _1 = {1, 0, 0, 0, 0, 0, 0, 0};
    const u64x8 lane_bit_0 = 1;
    const u64x8 ODD_BITS = 0xaaaaaaaaaaaaaaaaull;

    // The previous chunk's bitstrings (`carried` in the Zig version).
    u64x8 carried_carriages = zero, carried_slashes = zero, carried_backslashes = zero, carried_ats = zero;
    u64x8 carried_next_is_escaped = zero, carried_alpha_numeric_underscores = zero;
    u64x8 carried_identifier_or_number_or_builtin_ends = zero, carried_double_char = zero;
    u64x8 carried_delete_triple_char_pos = zero, carried_local_word_ends = zero, carried_all_starts = zero;
    // Which construct the previous chunk ended inside of, at bit 0.
    u64x8 carried_inside_quote_string = zero, carried_inside_char_literal = zero;
    u64x8 carried_inside_comment = zero, carried_inside_line_string = zero;

    // The masks of the 8 64-byte chunks: `o[mask][lane]`.
    uint64_t o[NUM_MASKS][8] __attribute__((aligned(64)));

    uint8_t *cur_kind = kinds_out;
    uint32_t *cur_start = starts_out;
    uint32_t *cur_end = ends_out;

    const uint8_t *maybe_aligned_cur_in = source;
    size_t maybe_aligned_len = source_len;
    if (source_len >= 3 && source[0] == 0xEF && source[1] == 0xBB && source[2] == 0xBF) {
        maybe_aligned_cur_in += 3;
        maybe_aligned_len -= 3;
    }

    const unsigned misalignment = (uintptr_t)maybe_aligned_cur_in & 63;
    const uint64_t load_mask = (maybe_aligned_len > 64 ? ~0ull : _bzhi_u64(~0ull, (uint32_t)maybe_aligned_len)) << misalignment;
    const uint8_t *cur_in = (const uint8_t *)((uintptr_t)maybe_aligned_cur_in & ~(uintptr_t)63);
    size_t cur_len = maybe_aligned_len + misalignment;

    // Bytes past the end of the file read as newlines, which terminate comments and strings.
    __m512i chunk = _mm512_mask_loadu_epi8(splat8('\n'), load_mask, cur_in);
    __m512i prev_chunk_for_pt3 = _mm512_setzero_si512();
    __m512i prev_classified = _mm512_setzero_si512();
    __m512i prev_double_char_ends = _mm512_setzero_si512();

    // The span of 64-byte chunks with non-ASCII bytes, validated as UTF-8 after the loop.
    size_t non_ascii_start = SIZE_MAX, non_ascii_end = 0;

    // This loop has 3 phases.
    //   I. Vectorized Classification: Read input, then produce bitstrings that tell us where different character classes exist in a chunk.
    //  II. Analysis: Do bit-manipulation to figure out where all tokens start and end
    // III. Write-out: Use the information produced in Phase II to produce the output `kinds`, `starts` and `ends`.
    for (;;) {
        const uint32_t chunk_pos = (uint32_t)((uintptr_t)cur_in - (uintptr_t)source);
        int reached_eof = 0;

        __m512i chunks[8], classified_chunks[8];

        for (unsigned i = 0;;) {
            chunks[i] = chunk;
            const uint64_t non_ascii = _mm512_movepi8_mask(chunk);

            // See the documentation on `classifier`
            __m512i classified = perm2b(azp_classifier, chunk);

            if (__builtin_expect(non_ascii != 0, 0)) {
                classified = _mm512_mask_blend_epi8(non_ascii, classified, splat8(AZP_TAG_UNKNOWN));
                const size_t pos = (uintptr_t)cur_in > (uintptr_t)source ? (size_t)(cur_in - source) : 0;
                if (pos < non_ascii_start) non_ascii_start = pos;
                non_ascii_end = pos + 64;
            }

            classified_chunks[i] = classified;
            const __m512i prev1_classified = shift_in_previous_1(classified, prev_classified);
            const __m512i prev2_classified = shift_in_previous_2(classified, prev_classified);

            const __m512i double_char_ends =
                vpshufb(azp_second_char_data, classified) & vpshufb(azp_first_char_data, prev1_classified) &
                _mm512_mask_blend_epi8(_mm512_cmpeq_epi8_mask(prev1_classified, classified), splat8(ALL_CHANNELS_BUT_SELF), splat8(ALL_CHANNELS));

            const __m512i triple_char_ends =
                vpshufb(azp_third_char_data, classified) & shift_in_previous_1(double_char_ends, prev_double_char_ends) &
                _mm512_mask_blend_epi8(_mm512_cmpeq_epi8_mask(prev2_classified, prev1_classified), splat8(ALL_CHANNELS_BUT_SELF), splat8(ALL_CHANNELS));

            o[CARRIAGES][i] = _mm512_cmpeq_epi8_mask(chunk, splat8('\r'));
            o[NEWLINES_OR_EOF][i] = _mm512_cmpeq_epi8_mask(chunk, splat8('\n'));
            o[SLASHES][i] = _mm512_cmpeq_epi8_mask(chunk, splat8('/'));
            o[BACKSLASHES][i] = _mm512_cmpeq_epi8_mask(chunk, splat8('\\'));
            o[QUOTES][i] = _mm512_cmpeq_epi8_mask(chunk, splat8('"'));
            o[APOSTROPHES][i] = _mm512_cmpeq_epi8_mask(chunk, splat8('\''));
            o[ATS][i] = _mm512_cmpeq_epi8_mask(chunk, splat8('@'));
            o[BAD_CONTROLS][i] = _mm512_cmpeq_epi8_mask(classified, splat8(AZP_TAG_INVALID));
            o[WHITESPACES][i] = _mm512_cmpeq_epi8_mask(classified, splat8(AZP_TAG_WHITESPACE));
            o[ALPHA_UNDERSCORES][i] = _mm512_cmpeq_epi8_mask(classified, splat8(AZP_TAG_IDENTIFIER));
            o[ALPHA_NUMERIC_UNDERSCORES][i] = _mm512_cmpge_epu8_mask(classified, splat8(AZP_TAG_IDENTIFIER));
            o[POSSIBLE_ONE_CHAR_SYMBOLS][i] = _mm512_cmplt_epu8_mask(classified, splat8(AZP_TAG_STRING_IDENTIFIER));
            o[PRELIM_DOUBLE_CHAR_ENDS][i] = _mm512_test_epi8_mask(double_char_ends, double_char_ends);
            o[PRELIM_TRIPLE_CHAR_ENDS][i] = _mm512_test_epi8_mask(triple_char_ends, triple_char_ends);

            prev_classified = classified;
            prev_double_char_ends = double_char_ends;

            i += 1;

            if (cur_len <= 64) {
                reached_eof = 1;
                // The rest of the 512-byte chunk is past the end of the file, so it looks like newlines.
                for (; i < 8; i++) {
                    chunks[i] = splat8('\n');
                    classified_chunks[i] = splat8(AZP_TAG_WHITESPACE);
                    for (int m = 0; m < NUM_MASKS; m++) o[m][i] = 0;
                    o[NEWLINES_OR_EOF][i] = ~0ull;
                    o[WHITESPACES][i] = ~0ull;
                }
                break;
            }

            cur_in += 64;
            cur_len -= 64;
            chunk = cur_len < 64 ? safe_load(cur_in, cur_len, '\n') : _mm512_load_si512(cur_in);
            if (i >= 8) break;
        }

        const u64x8 o_carriages = *(const u64x8 *)o[CARRIAGES];
        const u64x8 o_slashes = *(const u64x8 *)o[SLASHES];
        const u64x8 o_backslashes = *(const u64x8 *)o[BACKSLASHES];
        const u64x8 o_newlines_or_eof = *(const u64x8 *)o[NEWLINES_OR_EOF];
        const u64x8 o_quotes = *(const u64x8 *)o[QUOTES];
        const u64x8 o_apostrophes = *(const u64x8 *)o[APOSTROPHES];
        const u64x8 o_ats = *(const u64x8 *)o[ATS];
        const u64x8 o_whitespaces = *(const u64x8 *)o[WHITESPACES];
        const u64x8 o_alpha_underscores = *(const u64x8 *)o[ALPHA_UNDERSCORES];
        const u64x8 o_alpha_numeric_underscores = *(const u64x8 *)o[ALPHA_NUMERIC_UNDERSCORES];
        const u64x8 o_possible_one_char_symbols = *(const u64x8 *)o[POSSIBLE_ONE_CHAR_SYMBOLS];
        const u64x8 o_prelim_double_char_ends = *(const u64x8 *)o[PRELIM_DOUBLE_CHAR_ENDS];
        const u64x8 o_prelim_triple_char_ends = *(const u64x8 *)o[PRELIM_TRIPLE_CHAR_ENDS];

        // Matches `(?<=\r)([^\n]|$)`: a '\r' must be followed by '\n'.
        const u64x8 invalid_carriage_followers = ~o_newlines_or_eof & shld1(o_carriages, carried_carriages);
        if (!is_zero(invalid_carriage_followers)) return AZP_ERROR_INVALID_CARRIAGE_RETURN;

        // Matches the character after any odd number of backslashes (simdjson's algorithm by John Keiser, see the Zig version).
        u64x8 escaped_characters;
        {
            const u64x8 next_is_escaped = top_bit(carried_next_is_escaped);
            const u64x8 potential_escape = o_backslashes & ~next_is_escaped;
            const u64x8 maybe_escaped = shl(potential_escape, 1);
            const u64x8 even_series_codes_and_odd_bits = sub(maybe_escaped | ODD_BITS, potential_escape);
            const u64x8 escape_and_terminal_code = even_series_codes_and_odd_bits ^ ODD_BITS;
            escaped_characters = escape_and_terminal_code ^ (o_backslashes | next_is_escaped);
            carried_next_is_escaped = escape_and_terminal_code & o_backslashes;
        }

        const u64x8 unescaped_quotes = o_quotes & ~escaped_characters;
        const u64x8 unescaped_apostrophes = o_apostrophes & ~escaped_characters;
        // Matches `\/(?=\/)` and `\\(?=\\)`
        const u64x8 double_slash_starts = o_slashes & (shr(o_slashes, 1) | top_bit(carried_slashes));
        const u64x8 double_backslash_starts = o_backslashes & (shr(o_backslashes, 1) | top_bit(carried_backslashes));

        const u64x8 ended_inside_quote_string = carried_inside_quote_string;
        const u64x8 ended_inside_char_literal = carried_inside_char_literal;
        const u64x8 ended_inside_comment = carried_inside_comment;
        const u64x8 ended_inside_line_string = carried_inside_line_string;
        const u64x8 ended_inside_string_or_comment_or_char_literal =
            ended_inside_quote_string | ended_inside_char_literal | ended_inside_comment | ended_inside_line_string;

        // If we start inside of something, the first character can only continue (or close) that thing.
        const u64x8 not_carried_into_quote = ended_inside_string_or_comment_or_char_literal & ~ended_inside_quote_string;
        const u64x8 not_carried_into_char_literal = ended_inside_string_or_comment_or_char_literal & ~ended_inside_char_literal;
        const u64x8 not_carried_into_comment = ended_inside_string_or_comment_or_char_literal & ~ended_inside_comment;
        const u64x8 not_carried_into_line_string = ended_inside_string_or_comment_or_char_literal & ~ended_inside_line_string;

        const u64x8 quote_bounds_incl_carry = (unescaped_quotes & ~not_carried_into_quote) ^ ended_inside_quote_string;
        const u64x8 apostrophe_bounds_incl_carry = (unescaped_apostrophes & ~not_carried_into_char_literal) ^ ended_inside_char_literal;
        const u64x8 comment_bounds_incl_carry = (double_slash_starts & ~not_carried_into_comment) | ended_inside_comment;
        const u64x8 line_string_bounds_incl_carry = (double_backslash_starts & ~not_carried_into_line_string) | ended_inside_line_string;

        // A string/char_literal from the previous chunk may end at the first character: its (pseudo) start and end are the same.
        const u64x8 first_char_ends_string =
            (unescaped_quotes & ended_inside_quote_string) | (unescaped_apostrophes & ended_inside_char_literal);

        // Bitstrings that mark where each token begins and ends
        u64x8 all_starts = first_char_ends_string;
        u64x8 all_ends = first_char_ends_string;

        const u64x8 carriages_or_newlines_or_eof = o_carriages | o_newlines_or_eof;

        // Find the strings, character literals, comments and line strings, line by line in parallel.
        {
            u64x8 all_bounds_incl_carry = quote_bounds_incl_carry | apostrophe_bounds_incl_carry | comment_bounds_incl_carry | line_string_bounds_incl_carry;
            u64x8 iter = ~carriages_or_newlines_or_eof & shld1(carriages_or_newlines_or_eof, MSb);

            for (;;) {
                // Advance each cursor in `iter` to next 1 bit in `all_bounds_incl_carry | carriages_or_newlines_or_eof`,
                // but don't count `carriages_or_newlines_or_eof` as a valid start
                const u64x8 starts = all_bounds_incl_carry & ~sub(all_bounds_incl_carry | carriages_or_newlines_or_eof, iter);
                all_starts |= starts;

                // From each start to the next newline, fold in the characters that end what we started.
                u64x8 interleaved = carriages_or_newlines_or_eof;
                interleaved |= quote_bounds_incl_carry & sub(carriages_or_newlines_or_eof, starts & quote_bounds_incl_carry);
                interleaved |= apostrophe_bounds_incl_carry & sub(carriages_or_newlines_or_eof, starts & apostrophe_bounds_incl_carry);
                interleaved &= ~starts;

                // Advance cursor within interleaved bitstring
                const u64x8 ends = interleaved & ~sub(interleaved, starts);
                all_ends |= ends;

                // Make `ends` the new iters, but if we ended on a newline, delete that bit.
                iter = ends & ~carriages_or_newlines_or_eof;
                all_bounds_incl_carry &= ~ends;

                if (is_zero(iter)) break;
            }
        }

        const u64x8 inside_strings_and_comments_including_start = sub(all_ends, all_starts);

        // Record which construct we end inside of, if any: whichever one the last start belongs to.
        carried_inside_quote_string = zero;
        carried_inside_char_literal = zero;
        carried_inside_comment = zero;
        carried_inside_line_string = zero;
        if (!is_zero(inside_strings_and_comments_including_start & MSb)) {
            const u64x8 last_start = highest_bit(all_starts);
            if (!is_zero(last_start & quote_bounds_incl_carry)) carried_inside_quote_string = _1;
            else if (!is_zero(last_start & apostrophe_bounds_incl_carry)) carried_inside_char_literal = _1;
            else if (!is_zero(last_start & comment_bounds_incl_carry)) carried_inside_comment = _1;
            else carried_inside_line_string = _1;
        }

        // Revise `all_ends` such that tokens do not include the carriages_or_newlines_or_eof
        all_ends = (all_ends & ~carriages_or_newlines_or_eof) | shr(all_ends & carriages_or_newlines_or_eof, 1);

        // Add a dummy end if we ended inside a string/comment/line_string/char_literal
        all_ends |= inside_strings_and_comments_including_start & MSb;

        const u64x8 is_first_char_inside_string_or_comment =
            ended_inside_string_or_comment_or_char_literal & ~first_char_ends_string & ~carriages_or_newlines_or_eof;

        const u64x8 ats = ~inside_strings_and_comments_including_start & o_ats;
        const u64x8 alpha_underscores = ~inside_strings_and_comments_including_start & o_alpha_underscores;
        const u64x8 alpha_numeric_underscores = ~inside_strings_and_comments_including_start & o_alpha_numeric_underscores;

        const u64x8 string_identifier_started_before_ext_chunk = unescaped_quotes & top_bit(carried_ats);
        const u64x8 builtin_started_before_ext_chunk = alpha_underscores & top_bit(carried_ats);
        const u64x8 is_comment_opener_split_between_ext_chunks = o_slashes & top_bit(carried_slashes) & ~ended_inside_string_or_comment_or_char_literal;
        const u64x8 is_line_string_opener_split_between_ext_chunks =
            o_backslashes & top_bit(carried_backslashes) & ~ended_inside_string_or_comment_or_char_literal;
        const u64x8 identifier_or_number_or_builtin_with_letters_started_before_ext_chunk =
            top_bit(carried_identifier_or_number_or_builtin_ends) & alpha_numeric_underscores;

        // We shift in a zero bit to get a (pseudo) start at the beginning of a chunk for a token that crosses into it.
        const u64x8 follows_at = shl(ats, 1);
        const u64x8 identifier_or_number_starts = alpha_numeric_underscores & ~shl(alpha_numeric_underscores, 1) & ~follows_at;
        const u64x8 builtin_starts = (ats & shr(alpha_underscores, 1)) | builtin_started_before_ext_chunk;
        const u64x8 string_identifier_starts = (ats & shr(unescaped_quotes, 1)) | string_identifier_started_before_ext_chunk;

        const u64x8 identifier_or_number_or_builtin_ends = alpha_numeric_underscores & ~shr(alpha_numeric_underscores, 1);

        all_starts &= ~(follows_at & unescaped_quotes);
        all_starts |= identifier_or_number_starts | builtin_starts | string_identifier_starts;
        all_ends |= identifier_or_number_or_builtin_ends;

        // Multi-char symbols: always match the longest possible symbol.
        u64x8 double_char_ends, triple_char_ends;
        {
            const u64x8 prelim_triple_char_ends = ~inside_strings_and_comments_including_start & o_prelim_triple_char_ends;
            const u64x8 prelim_double_char_ends =
                ~inside_strings_and_comments_including_start & o_prelim_double_char_ends & ~shr(prelim_triple_char_ends, 1);

            u64x8 s = prelim_double_char_ends | prelim_triple_char_ends;
            u64x8 ends = zero;

            s &= ~(carried_delete_triple_char_pos & ~prelim_double_char_ends);
            s &= ~(top_bit(carried_double_char) & ~prelim_triple_char_ends);

            for (;;) {
                // Iterate over all positions that have at least two non-symbol chars before them at once.
                const u64x8 iter = s & ~shl(s, 1) & ~shl(s, 2);
                ends |= iter;
                s &= ~(iter | shl(iter, 1) | (shl(iter, 2) & ~prelim_double_char_ends));
                if (__builtin_expect(is_zero(s), 1)) break;
            }

            // 3-char sequences take precedence over two-char sequences, except when they are just two characters apart.
            double_char_ends = (ends & ~prelim_triple_char_ends) | (ends & shl(ends, 2));
            triple_char_ends = ends ^ double_char_ends;

            carried_delete_triple_char_pos = (u64x8){ends[7] >> 62, 0, 0, 0, 0, 0, 0, 0};
            carried_double_char = ends & prelim_double_char_ends;
        }

        const u64x8 single_char_ends = o_possible_one_char_symbols & ~inside_strings_and_comments_including_start &
                                       ~(double_char_ends | shr(double_char_ends, 1)) &
                                       ~(triple_char_ends | shr(triple_char_ends, 1) | shr(triple_char_ends, 2));

        const u64x8 multi_char_symbol_ends = double_char_ends | triple_char_ends;
        const u64x8 all_symbol_ends = single_char_ends | multi_char_symbol_ends;

        const u64x8 multi_char_symbol_starts = shr(double_char_ends, 1) | shr(triple_char_ends, 2);
        const u64x8 multi_char_has_two_chars_in_prev_chunk = _1 & triple_char_ends;
        const u64x8 multi_char_has_one_char_in_prev_chunk = _1 & (double_char_ends | shr(triple_char_ends, 1));
        const u64x8 multi_char_starts_before_chunk = multi_char_has_one_char_in_prev_chunk | multi_char_has_two_chars_in_prev_chunk;
        const u64x8 all_multi_char_starts = multi_char_symbol_starts | multi_char_starts_before_chunk;

        // Per 64-byte chunk, the starts of the multi-char symbols that end in it, with a pseudo-start at bit 0 for one
        // that started in the previous 64-byte chunk.
        const u64x8 multi_char_crosses_into_lane = lane_bit_0 & (double_char_ends | triple_char_ends | (triple_char_ends >> 1));
        const u64x8 multi_char_starts_by_end_lane = (double_char_ends >> 1) | (triple_char_ends >> 2) | multi_char_crosses_into_lane;

        all_starts |= single_char_ends | all_multi_char_starts;
        all_ends |= all_symbol_ends;

        // Every character that is neither whitespace nor part of a token becomes a 1-character `invalid` token.
        const u64x8 invalids = ~(sub(all_ends, all_starts) | all_ends | o_whitespaces);
        all_starts |= invalids;
        all_ends |= invalids;

        // Keyword candidates, found per 64-byte chunk. A candidate can't follow an `@`, and bit 0 of a 64-byte chunk only
        // starts one if it really starts a word, or if it continues a candidate from the previous 64-byte chunk.
        const u64x8 prev_chunk_ended_on_alpha_numeric_underscore = prev_lanes(carried_alpha_numeric_underscores, alpha_numeric_underscores) >> 63;
        const u64x8 local_word_starts = alpha_underscores & ~(alpha_numeric_underscores << 1) & ~shld1(ats, carried_ats) &
                                        ~prev_chunk_ended_on_alpha_numeric_underscore;

        const u64x8 local_identifier_or_number_or_builtin_ends = alpha_numeric_underscores & ~(alpha_numeric_underscores >> 1);
        const u64x8 local_word_ends = local_identifier_or_number_or_builtin_ends &
                                      ~(local_identifier_or_number_or_builtin_ends - local_word_starts);

        const u64x8 prev_chunk_ended_on_possible_kw = prev_lanes(carried_local_word_ends, local_word_ends) >> 63;
        const u64x8 possible_kw_started_before_chunk = prev_chunk_ended_on_possible_kw & alpha_numeric_underscores;
        const u64x8 possible_kw_starts = local_word_starts | possible_kw_started_before_chunk;
        const u64x8 possible_kw_ends = local_identifier_or_number_or_builtin_ends &
                                       ~(local_identifier_or_number_or_builtin_ends - possible_kw_starts);

        const u64x8 possible_kw_starts_longer_than_1 = possible_kw_starts & (~possible_kw_ends | prev_chunk_ended_on_possible_kw);
        // `first_two_chars_mask` can grab a dummy bit for an identifier split between chunks with one character in this one.
        const u64x8 first_two_chars_mask = possible_kw_starts_longer_than_1 | (possible_kw_starts_longer_than_1 << 1);

        // Add dummy bit to second bit position if we have only one character of a keyword started in the previous chunk
        const u64x8 possible_kw_with_one_char_in_chunk = prev_chunk_ended_on_possible_kw & possible_kw_ends;
        const u64x8 possible_kw_ends_longer_than_1 = (possible_kw_ends & ~possible_kw_starts) | (possible_kw_with_one_char_in_chunk << 1);
        const u64x8 last_two_chars_mask = possible_kw_ends_longer_than_1 | (possible_kw_ends_longer_than_1 >> 1);

        const u64x8 prev_carried_lens = (u64x8)_mm512_lzcnt_epi64((__m512i)prev_lanes(carried_all_starts, all_starts)) + 1;
        const u64x8 possible_kw_carried_lens = prev_carried_lens & -possible_kw_started_before_chunk;

        const u64x8 first_token_starts_before_ext_chunk =
            is_first_char_inside_string_or_comment | first_char_ends_string | string_identifier_started_before_ext_chunk |
            builtin_started_before_ext_chunk | is_comment_opener_split_between_ext_chunks |
            is_line_string_opener_split_between_ext_chunks | (_1 & multi_char_starts_before_chunk) |
            identifier_or_number_or_builtin_with_letters_started_before_ext_chunk;

        // If the first token started before the current chunk, go back to it and save its start and kind.
        const unsigned first_token_crosses = (unsigned)first_token_starts_before_ext_chunk[0];
        cur_kind -= first_token_crosses;
        cur_start -= first_token_crosses;
        cur_end -= first_token_crosses;
        const uint32_t carried_start = first_token_crosses ? cur_start[0] : 0;
        const uint8_t carried_kind = first_token_crosses ? cur_kind[0] : 0;

        // Lane 0 keeps its previous kind if the first token crossed over (but a split `\\` gets the classified kind).
        uint64_t carry_kind_mask = ~((first_token_starts_before_ext_chunk & ~is_line_string_opener_split_between_ext_chunks)[0] & 1);

        // In "reasonable" code, at most 13 multi-char symbols fit in a 64-byte chunk, so more than 16 is rare.
        const int contains_multi_char_spam =
            _mm512_cmpge_epu64_mask(_mm512_popcnt_epi64((__m512i)multi_char_symbol_ends), _mm512_set1_epi64(16)) != 0;

        // Which 64-byte chunks must overwrite the kind of the token that crossed into them from the previous one: a keyword
        // candidate or multi-char symbol. The first 64-byte chunk instead rewinds the whole token (`first_token_crosses`).
        const u64x8 kind_rewrites = (possible_kw_started_before_chunk | multi_char_crosses_into_lane) & ~_1;

        uint64_t all_starts_lanes[8], all_ends_lanes[8], multi_char_symbol_ends_lanes[8], triple_char_ends_lanes[8];
        uint64_t multi_char_starts_by_end_lane_lanes[8], first_two_chars_mask_lanes[8], last_two_chars_mask_lanes[8];
        uint64_t possible_kw_starts_longer_than_1_lanes[8], possible_kw_carried_lens_lanes[8], possible_kw_with_one_char_lanes[8];
        uint64_t kind_rewrites_lanes[8], builtin_starts_lanes[8], string_identifier_starts_lanes[8];
        uint64_t comment_bounds_incl_carry_lanes[8], invalids_lanes[8];
        memcpy(all_starts_lanes, &all_starts, 64);
        memcpy(all_ends_lanes, &all_ends, 64);
        memcpy(multi_char_symbol_ends_lanes, &multi_char_symbol_ends, 64);
        memcpy(triple_char_ends_lanes, &triple_char_ends, 64);
        memcpy(multi_char_starts_by_end_lane_lanes, &multi_char_starts_by_end_lane, 64);
        memcpy(first_two_chars_mask_lanes, &first_two_chars_mask, 64);
        memcpy(last_two_chars_mask_lanes, &last_two_chars_mask, 64);
        memcpy(possible_kw_starts_longer_than_1_lanes, &possible_kw_starts_longer_than_1, 64);
        memcpy(possible_kw_carried_lens_lanes, &possible_kw_carried_lens, 64);
        memcpy(possible_kw_with_one_char_lanes, &possible_kw_with_one_char_in_chunk, 64);
        memcpy(kind_rewrites_lanes, &kind_rewrites, 64);
        memcpy(builtin_starts_lanes, &builtin_starts, 64);
        memcpy(string_identifier_starts_lanes, &string_identifier_starts, 64);
        memcpy(comment_bounds_incl_carry_lanes, &comment_bounds_incl_carry, 64);
        memcpy(invalids_lanes, &invalids, 64);

        for (unsigned j = 0; j < 8; j++) {
            const __m512i loaded_chunk = chunks[j];

            // This is a vector that holds the `kind` of each token.
            // It's okay to have tags broadcasted to places that will not be extracted in the end
            __m512i kinds = _mm512_mask_blend_epi8(carry_kind_mask, splat8(carried_kind), classified_chunks[j]);
            kinds = _mm512_mask_mov_epi8(kinds, builtin_starts_lanes[j], splat8(AZP_TAG_BUILTIN));
            kinds = _mm512_mask_mov_epi8(kinds, string_identifier_starts_lanes[j], splat8(AZP_TAG_STRING_IDENTIFIER));
            kinds = _mm512_mask_mov_epi8(kinds, comment_bounds_incl_carry_lanes[j], splat8(AZP_TAG_COMMENT));
            kinds = _mm512_mask_mov_epi8(kinds, invalids_lanes[j], splat8(AZP_TAG_INVALID));

            // Multi-char symbols, classified from where they end.
            {
                const __m512i multi_char_end_indices = bits_to_indices(multi_char_symbol_ends_lanes[j]);
                // Maximally, we can have 32 2 character symbols in a 64-byte chunk.
                const uint64_t which_are_triple = _pext_u64(triple_char_ends_lanes[j], multi_char_symbol_ends_lanes[j]);
                const __m128i low = get_multi_char_kinds(prev_chunk_for_pt3, loaded_chunk, _mm512_castsi512_si128(multi_char_end_indices),
                                                         (uint16_t)which_are_triple);
                __m128i high = _mm_undefined_si128();
                if (__builtin_expect(contains_multi_char_spam, 0))
                    high = get_multi_char_kinds(_mm512_setzero_si512(), loaded_chunk, _mm512_extracti32x4_epi32(multi_char_end_indices, 1),
                                                (uint16_t)(which_are_triple >> 16));
                const __m256i multi_char_kinds_compressed = _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
                kinds = _mm512_mask_expand_epi8(kinds, multi_char_starts_by_end_lane_lanes[j], _mm512_castsi256_si512(multi_char_kinds_compressed));
            }

            // Keywords: hash the first two and last two characters (and the length) of up to 24 candidates, look up the
            // keyword that could match, and compare 16 bytes of each, 8 candidates at a time.
            {
                const __m512i start_indices = _mm512_sub_epi8(bits_to_indices(first_two_chars_mask_lanes[j]),
                                                              _mm512_maskz_set1_epi8(3, (char)possible_kw_carried_lens_lanes[j]));
                const __m512i end_indices = _mm512_sub_epi8(bits_to_indices(last_two_chars_mask_lanes[j]),
                                                            _mm512_maskz_set1_epi8(3, (char)possible_kw_with_one_char_lanes[j]));
                const __m512i identifier_lens = _mm512_add_epi8(_mm512_sub_epi8(end_indices, start_indices), splat8(2));

                const __m512i first_two_chars = lookup_cur_prev(loaded_chunk, prev_chunk_for_pt3, start_indices);
                const __m512i last_two_chars = lookup_cur_prev(loaded_chunk, prev_chunk_for_pt3, end_indices);

                // The upper byte of each u16 is the hash; `>> 8` moves it to the lower byte.
                const __m512i hash = _mm512_srli_epi16(
                    _mm512_mullo_epi16(_mm512_xor_si512(_mm512_slli_epi16(identifier_lens, 14), first_two_chars), last_two_chars), 8);

                const __m512i super_looked_up_starts = perm2b(azp_keyword_starts, hash);
                const __m512i super_looked_up_lens = perm2b(azp_keyword_lens, hash);

                const u8x64 half_0 = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                      0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
                const u8x64 half_1 = half_0 + 8;

#define MATCH_GROUP(k)                                                                                                         \
    ({                                                                                                                         \
        const __m512i lookup_starts_group = (__m512i)__builtin_shufflevector((u8x64)super_looked_up_starts, (u8x64)super_looked_up_starts, GROUP_OF_8(k)); \
        const __m512i lookup_lens_group = (__m512i)__builtin_shufflevector((u8x64)super_looked_up_lens, (u8x64)super_looked_up_lens, GROUP_OF_8(k));       \
        const __m512i starts_group = (__m512i)__builtin_shufflevector((u8x64)start_indices, (u8x64)start_indices, GROUP_OF_8(k));                        \
        const __m512i lens_group = (__m512i)__builtin_shufflevector((u8x64)identifier_lens, (u8x64)identifier_lens, GROUP_OF_8(k));                      \
        uint8_t cur_matched = 0xFF;                                                                                            \
        for (int half = 0; half < 2; half++) {                                                                                 \
            const __m512i indices = (__m512i)(half == 0 ? half_0 : half_1);                                                     \
            const __m512i keyword_candidates_from_chunk_half = _mm512_mask_blend_epi8(                                         \
                _mm512_cmplt_epu8_mask(indices, lens_group), lens_group,                                                       \
                lookup_cur_prev(loaded_chunk, prev_chunk_for_pt3, _mm512_add_epi8(starts_group, indices)));                    \
            const __m512i keywords_from_table2_half = _mm512_mask_blend_epi8(                                                  \
                _mm512_cmplt_epu8_mask(indices, lookup_lens_group), lookup_lens_group,                                         \
                perm4b(azp_keyword_superstring, _mm512_add_epi8(lookup_starts_group, indices)));                               \
            cur_matched &= _mm512_cmpeq_epi64_mask(keyword_candidates_from_chunk_half, keywords_from_table2_half);           \
        }                                                                                                                      \
        cur_matched;                                                                                                           \
    })

                // There are at most 22 multi-character identifiers in a 64-byte chunk, including a carry-over.
                uint32_t matched = MATCH_GROUP(0);
                if (__builtin_expect(__builtin_popcountll(possible_kw_starts_longer_than_1_lanes[j]) > 8, 0)) {
                    matched |= (uint32_t)MATCH_GROUP(1) << 8;
                    matched |= (uint32_t)MATCH_GROUP(2) << 16;
                }
#undef MATCH_GROUP

                // The hash is 7 bits, but in our vector each one still occupies 2 bytes of space. Keep the lower ones.
                // Keywords are the tags with the highest bit set.
                const u8x64 h = (u8x64)hash;
                const __m256i potential_kinds = _mm256_or_si256(
                    (__m256i)__builtin_shufflevector(h, h, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42,
                                                     44, 46, 48, 50, 52, 54, 56, 58, 60, 62),
                    _mm256_set1_epi8((char)128));
                const __m256i matched_kinds = _mm256_mask_blend_epi8(matched, _mm256_set1_epi8((char)AZP_TAG_IDENTIFIER), potential_kinds);

                kinds = _mm512_mask_expand_epi8(kinds, possible_kw_starts_longer_than_1_lanes[j], _mm512_castsi256_si512(matched_kinds));
            }

            // Write out the kinds, starts and ends. Starts and ends are written independently, so a token that crosses into
            // the next 64-byte chunk has its end written there. When the crossing token's kind is only known now, we go
            // back one slot and write over the kind that the previous 64-byte chunk guessed.
            const uint64_t kind_rewrite = kind_rewrites_lanes[j] & 1;
            _mm512_storeu_si512(cur_kind - kind_rewrite, _mm512_maskz_compress_epi8(all_starts_lanes[j] | kind_rewrite, kinds));

            const uint32_t lane_pos = chunk_pos + 64 * j;
            const __m512i start_indices = bits_to_indices(all_starts_lanes[j]);
            const __m512i end_indices = bits_to_indices(all_ends_lanes[j]);
            const __m512i start_offset = _mm512_set1_epi32((int)lane_pos);
            const __m512i end_offset = _mm512_set1_epi32((int)(lane_pos + 1)); // Exclusive, like std.zig.Token.Loc
            _mm512_storeu_si512(cur_start + 0, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(start_indices, 0)), start_offset));
            _mm512_storeu_si512(cur_start + 16, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(start_indices, 1)), start_offset));
            _mm512_storeu_si512(cur_start + 32, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(start_indices, 2)), start_offset));
            _mm512_storeu_si512(cur_start + 48, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(start_indices, 3)), start_offset));
            _mm512_storeu_si512(cur_end + 0, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(end_indices, 0)), end_offset));
            _mm512_storeu_si512(cur_end + 16, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(end_indices, 1)), end_offset));
            _mm512_storeu_si512(cur_end + 32, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(end_indices, 2)), end_offset));
            _mm512_storeu_si512(cur_end + 48, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(end_indices, 3)), end_offset));
            if (j == 0 && first_token_crosses) cur_start[0] = carried_start;

            const unsigned num_starts = (unsigned)__builtin_popcountll(all_starts_lanes[j]);
            cur_kind += num_starts;
            cur_start += num_starts;
            cur_end += __builtin_popcountll(all_ends_lanes[j]);

            prev_chunk_for_pt3 = loaded_chunk;
            carry_kind_mask = ~0ull;
        }

        carried_carriages = o_carriages;
        carried_slashes = o_slashes;
        carried_backslashes = o_backslashes;
        carried_ats = ats;
        carried_alpha_numeric_underscores = alpha_numeric_underscores;
        carried_identifier_or_number_or_builtin_ends = identifier_or_number_or_builtin_ends;
        carried_local_word_ends = local_word_ends;
        carried_all_starts = all_starts;

        if (reached_eof) break;
    }

    // Because the main loop incorrectly allows a trailing carriage return, throw here if applicable.
    if (source_len != 0 && source[source_len - 1] == '\r') return AZP_ERROR_INVALID_CARRIAGE_RETURN;

    // No UTF-8 sequence can start or end in a 64-byte chunk that is all ASCII.
    if (non_ascii_start < non_ascii_end) {
        const size_t end = non_ascii_end < source_len ? non_ascii_end : source_len;
        if (!utf8_is_valid(source + non_ascii_start, end - non_ascii_start)) return AZP_ERROR_INVALID_UTF8;
    }

    return cur_kind - kinds_out;
}
