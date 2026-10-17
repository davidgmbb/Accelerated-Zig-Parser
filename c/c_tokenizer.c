// The vectorized C tokenizer. See c_tokenizer.h, and src/CompactingTokenizer.zig for the 512-byte-chunk design it
// shares with the Zig tokenizer.
//
// Each iteration tokenizes a 512-byte chunk as eight 64-byte vectors, with a 512-bit bitstring per character class
// (`u64x8`, lane 0 holding the lowest bits). Phase I classifies each vector, Phase II finds where every token starts
// and ends with bitstring arithmetic, and Phase III compresses the tokens' kinds, starts and ends into the output.
// Phase I runs a chunk ahead, and also classifies the first vector after the chunk, so that Phase II can look ahead:
// a token that crosses into the next chunk gets its start and kind from this chunk, and its end from the next one.
//
// A few rare inputs aren't modeled by the bitstrings, and make the tokenizer start over with the scalar lexer
// (`c_tokenize_scalar`): a line splice between characters that could make one token (outside of literals and line
// comments), a splice after a backslash or apostrophe or with whitespace before its newline, a non-ASCII byte or UCN
// outside of literals and comments (a UTF-8 identifier or a stray character), a stray backslash before a quote or
// backslash, the overlapping comment delimiters "/*/" and "*/*", an unterminated block comment, block comments whose
// lines need more than a few rescans to settle, and trigraphs (when enabled). That's under 1% of the bytes of the
// C corpora it was tested on.
#include "c_tokenizer.h"

#if !defined(__clang__)
#error "c_tokenizer.c uses Clang vector extensions"
#endif
#if !(defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VBMI__) && \
      defined(__AVX512VBMI2__) && defined(__AVX512VPOPCNTDQ__) && defined(__BMI2__))
#error "c_tokenizer.c needs AVX-512 F/BW/VL/VBMI/VBMI2/VPOPCNTDQ and BMI2 (e.g. -march=znver4, znver5 or tigerlake)"
#endif

#include <immintrin.h>
#include <stdbool.h>
#include <string.h>

#include "c_tokenizer_tables.h"

// The channels of 2-character punctuators, and those that need the same character twice.
#define OP2_CHANNELS (CHANNEL_EQ | CHANNEL_SELF | CHANNEL_SHIFT | CHANNEL_GT | CHANNEL_LTDIG | CHANNEL_HASHDIG)
#define SAME_CHARACTER_CHANNELS (CHANNEL_SELF | CHANNEL_SHIFT | CHANNEL_DOTS)

#define INLINE static inline __attribute__((always_inline))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

typedef uint64_t u64x8 __attribute__((ext_vector_type(8)));
typedef uint8_t u8x64 __attribute__((ext_vector_type(64)));
typedef uint8_t u8x16 __attribute__((ext_vector_type(16)));

static const char* const c_token_kind_names[C_TOKEN_KIND_COUNT] =
{
#define C_TOKEN_NAME(name, clang_name) #clang_name,
    C_TOKEN_KINDS(C_TOKEN_NAME)
#undef C_TOKEN_NAME
};

const char* c_token_kind_name(uint8_t kind)
{
    uint8_t k = kind & (uint8_t)~C_TOKEN_START_OF_LINE;
    return k < C_TOKEN_KIND_COUNT ? c_token_kind_names[k] : "<invalid kind>";
}

static const uint8_t iota_bytes[64] __attribute__((aligned(64))) =
{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
};

// ---------------------------------------------------------------------------------------------------------------------
// 512-bit bitstrings

// `a << i` and `a >> i` within the chunk, for a constant 0 < i < 64.
INLINE u64x8 shl(u64x8 a, int i)
{
    return (__builtin_shufflevector(a, (u64x8){0}, 8, 0, 1, 2, 3, 4, 5, 6) >> (64 - i)) | (a << i);
}

INLINE u64x8 shr(u64x8 a, int i)
{
    return (__builtin_shufflevector(a, (u64x8){0}, 1, 2, 3, 4, 5, 6, 7, 8) << (64 - i)) | (a >> i);
}

// `a << i`, shifting in the top `i` bits of the previous chunk's bitstring `previous`.
INLINE u64x8 shl_carry(u64x8 a, u64x8 previous, int i)
{
    return (__builtin_shufflevector(a, previous, 15, 0, 1, 2, 3, 4, 5, 6) >> (64 - i)) | (a << i);
}

// `a >> i`, shifting in the low `i` bits of `next`, the bitstring of the next chunk's first 64 bytes.
INLINE u64x8 shr_next(u64x8 a, uint64_t next, int i)
{
    return (__builtin_shufflevector(a, (u64x8){next}, 1, 2, 3, 4, 5, 6, 7, 8) << (64 - i)) | (a >> i);
}

// 512-bit subtraction: a lane borrows when the lane below it borrowed out (Kogge-Stone, propagated in a mask register).
INLINE u64x8 sub(u64x8 a, u64x8 b)
{
    const uint8_t carries = _mm512_cmplt_epu64_mask((__m512i)a, (__m512i)b);
    const uint8_t equal = _mm512_cmpeq_epi64_mask((__m512i)a, (__m512i)b);
    const uint8_t borrows = equal ^ (uint8_t)((carries << 1) + equal);
    const u64x8 difference = a - b;
    return (u64x8)_mm512_mask_add_epi64((__m512i)difference, borrows, (__m512i)difference, _mm512_set1_epi64(-1));
}

// 512-bit addition with a carry in and out.
INLINE u64x8 add_carry(u64x8 a, u64x8 b, uint32_t carry_in, uint32_t* carry_out)
{
    const u64x8 sum = a + b;
    const uint32_t generates = _mm512_cmplt_epu64_mask((__m512i)sum, (__m512i)a);
    const uint32_t propagates = _mm512_cmpeq_epi64_mask((__m512i)sum, _mm512_set1_epi64(-1));
    const uint32_t carries = ((generates << 1) | carry_in) + propagates;
    *carry_out = (carries >> 8) & 1;
    return (u64x8)_mm512_mask_sub_epi64((__m512i)sum, (uint8_t)(carries ^ propagates), (__m512i)sum, _mm512_set1_epi64(-1));
}

INLINE bool is_zero(u64x8 x)
{
    return _mm512_test_epi64_mask((__m512i)x, (__m512i)x) == 0;
}

INLINE uint64_t bit_511(u64x8 x)
{
    return x[7] >> 63;
}

// The position of the highest set bit below `limit` (at most 512), or -1 if there is none.
static int32_t highest_bit_below(u64x8 x, uint32_t limit)
{
    int32_t result = -1;
    for (int32_t lane = 7; lane >= 0; lane -= 1)
    {
        const uint32_t lane_start = (uint32_t)lane * 64;
        if (lane_start >= limit)
        {
            continue;
        }
        uint64_t bits = x[lane];
        if (limit - lane_start < 64)
        {
            bits &= _bzhi_u64(~0ull, limit - lane_start);
        }
        if (bits != 0)
        {
            result = (int32_t)lane_start + 63 - (int32_t)__builtin_clzll(bits);
            break;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------------------------------------------------
// Vectors

INLINE __m512i splat8(uint8_t x)
{
    return _mm512_set1_epi8((char)x);
}

// Looks up 64 bytes in a 16-byte table (repeated across the vector). Indices >= 0x80 give 0.
INLINE __m512i vpshufb(const uint8_t table[16], __m512i indices)
{
    return _mm512_shuffle_epi8(_mm512_broadcast_i32x4(_mm_load_si128((const __m128i*)table)), indices);
}

// Looks up 64 bytes in a 128-byte table, using the low 7 bits of each index.
INLINE __m512i perm2b(const uint8_t table[128], __m512i indices)
{
    return _mm512_permutex2var_epi8(_mm512_load_si512(table), indices, _mm512_load_si512(table + 64));
}

// The indices of the set bits of `mask`, in order.
INLINE __m512i bits_to_indices(uint64_t mask)
{
    return _mm512_maskz_compress_epi8(mask, _mm512_load_si512(iota_bytes));
}

// `current` shifted up by one byte, shifting in the last byte of `previous`.
INLINE __m512i shift_in_previous(__m512i current, __m512i previous)
{
    return (__m512i)__builtin_shufflevector((u8x64)current, (u8x64)previous, 127, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                            56, 57, 58, 59, 60, 61, 62);
}

// The kinds of up to 16 multi-character punctuators that start at `start_indices` of `current` (continuing into
// `next`), by a perfect hash of their first three characters (two for 2-character punctuators). `long_ones` marks the
// punctuators of three or more characters.
INLINE __m128i punctuator_kinds(__m512i current, __m512i next, __m128i start_indices, uint16_t long_ones)
{
    // Per punctuator, the offsets of its bytes from its start; 0x80 zeroes a byte.
    const __m512i offsets = _mm512_mask_blend_epi32(long_ones, _mm512_set1_epi32((int)0x80800100), _mm512_set1_epi32((int)0x80020100));
    const u8x16 s = (u8x16)start_indices;
    const u8x64 repeated = __builtin_shufflevector(s, s, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6,
                                                   6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12,
                                                   12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15);
    const __m512i byte_indices = _mm512_add_epi8((__m512i)repeated, offsets);
    const __m512i bytes = _mm512_maskz_permutex2var_epi8(_mm512_cmplt_epu8_mask(offsets, splat8(0x80)), current, byte_indices, next);
    const __m512i hashes = _mm512_srli_epi32(_mm512_mullo_epi32(bytes, _mm512_set1_epi32((int)OP_HASH_MULTIPLIER)), 26);
    return _mm512_cvtepi32_epi8(_mm512_permutexvar_epi8(hashes, _mm512_load_si512(c_op_hash_kinds)));
}

// Could the characters before and after a line splice be part of the same token (outside of literals and comments)?
// `earlier` is the character before `before`.
static bool splice_joins(uint8_t earlier, uint8_t before, uint8_t after)
{
    const bool before_identifier = before >= 0x80 || (c_char_properties[before] & CHAR_IDENTIFIER);
    const bool after_identifier = after >= 0x80 || (c_char_properties[after] & CHAR_IDENTIFIER);
    bool result;
    if (after == '\\')
    {
        result = true; // Another splice (or a UCN), so `before` may join what comes after it.
    }
    else if (before_identifier)
    {
        // An identifier or pp-number that goes on (with a '.', a digit separator, an exponent's sign or a UCN), or
        // a prefix and its literal.
        const bool exponent = before == 'e' || before == 'E' || before == 'p' || before == 'P';
        result = after_identifier || after == '.' || after == '\'' || after == '"' || after == '\\' || (exponent && (after == '+' || after == '-'));
    }
    else if ((before == '+' || before == '-') && (earlier == 'e' || earlier == 'E' || earlier == 'p' || earlier == 'P') &&
             (after_identifier || after == '.' || after == '\'' || after == '\\'))
    {
        result = true; // The sign of an exponent, in a pp-number that goes on.
    }
    else if (before == '.')
    {
        result = after == '.' || after_identifier || after == '\''; // An ellipsis, or a pp-number.
    }
    else
    {
        // A punctuator of two or more characters (with ".." and ":%" for "..." and "%:%:"), or a comment's start.
        const uint8_t first = c_operator_classes[before & 0x7F];
        const uint8_t second = after < 0x80 ? c_operator_classes[after] : 0;
        uint8_t channels = c_first_char_channels[first & 15] & c_second_char_channels[second & 15];
        if (first != second)
        {
            channels &= (uint8_t)~SAME_CHARACTER_CHANNELS;
        }
        result = (before < 0x80 && channels != 0) || (before == '/' && (after == '/' || after == '*')) || (before == ':' && after == '%');
    }
    return result;
}

// Is there a trigraph ("??" and one of =/'()!<>-) in the source?
static bool contains_trigraph(const uint8_t* source, uint64_t length)
{
    const uint8_t* end = source + length;
    for (const uint8_t* p = source; p + 2 < end;)
    {
        const uint8_t* question = memchr(p, '?', (uint64_t)(end - p - 2));
        if (question == 0)
        {
            break;
        }
        if (question[1] == '?' && memchr("=/'()!<>-", question[2], 9) != 0)
        {
            return true;
        }
        p = question + 1;
    }
    return false;
}

// Loads the 64 bytes at `lane` (relative to the 64-byte-aligned `base`), reading the bytes outside of
// [`begin`, `length`) of the source (which starts at `base + base_offset`) as newlines.
INLINE __m512i load_lane(const uint8_t* base, int64_t base_offset, uint64_t begin, uint64_t length, int64_t lane)
{
    const int64_t first = -base_offset - lane + (int64_t)begin; // The first valid byte of the lane.
    const int64_t end = -base_offset - lane + (int64_t)length; // Past the last valid byte.
    __m512i result;
    if (LIKELY(first <= 0 && end >= 64))
    {
        result = _mm512_load_si512(base + lane);
    }
    else
    {
        const uint64_t below_end = end <= 0 ? 0 : end >= 64 ? ~0ull : _bzhi_u64(~0ull, (uint32_t)end);
        const uint64_t from_first = first <= 0 ? ~0ull : first >= 64 ? 0 : ~_bzhi_u64(~0ull, (uint32_t)first);
        result = _mm512_mask_loadu_epi8(splat8('\n'), below_end & from_first, base + lane);
    }
    return result;
}

// The bits of a lane at or past the end of the source.
INLINE uint64_t lane_padding(int64_t base_offset, uint64_t length, int64_t lane)
{
    const int64_t end = -base_offset - lane + (int64_t)length;
    return end <= 0 ? ~0ull : end >= 64 ? 0 : ~_bzhi_u64(~0ull, (uint32_t)end);
}

// ---------------------------------------------------------------------------------------------------------------------

// The character-class bitstrings that Phase I makes, as `o[mask][lane]`. Lane 8 is the next chunk's first lane.
enum
{
    M_NEWLINE, // \n \r, and the bytes past the end of the source
    M_WHITESPACE, // ' ' \t \v \f \0
    M_IDENTIFIER, // [A-Za-z0-9_$]
    M_DIGIT,
    M_SEPARATOR_FOLLOWER, // [A-Za-z0-9_]: what may follow a digit separator
    M_QUOTE,
    M_APOSTROPHE,
    M_BACKSLASH,
    M_SLASH,
    M_STAR,
    M_DOT,
    M_EXPSIGN, // a + or - after e, E, p or P
    M_HASHDIG, // the ':' of "%:"
    M_OP2, // the last character of a pair of characters that make a punctuator
    M_OP3, // the last character of "...", "<<=" or ">>="
    M_NONASCII,
    NUM_MASKS,
};


// What the chunk starts inside of.
typedef enum State
{
    STATE_OUTSIDE,
    STATE_STRING,
    STATE_CHAR,
    STATE_LINE_COMMENT,
    STATE_BLOCK_COMMENT,
} State;

#define NEEDS_SCALAR C_TOKENIZER_ERROR_NEEDS_SCALAR

// Phase I for one 64-byte vector: its character-class bitstrings, stored to `o[...][lane]`.
INLINE void classify(__m512i raw, uint64_t o[][16], unsigned lane, __m512i* previous_classes, __m512i* previous_pairs)
{
    const uint64_t nonascii = _mm512_movepi8_mask(raw);
    __m512i properties = perm2b(c_char_properties, raw);
    __m512i classes = perm2b(c_operator_classes, raw);
    if (UNLIKELY(nonascii != 0))
    {
        properties = _mm512_maskz_mov_epi8(~nonascii, properties);
        classes = _mm512_maskz_mov_epi8(~nonascii, classes);
    }

    // See c/gen_c_tokenizer_tables.py for the channels.
    const __m512i previous = shift_in_previous(classes, *previous_classes);
    const __m512i allowed = _mm512_mask_blend_epi8(_mm512_cmpeq_epi8_mask(previous, classes), splat8((uint8_t)~SAME_CHARACTER_CHANNELS), splat8(0xFF));
    const __m512i pairs = vpshufb(c_second_char_channels, classes) & vpshufb(c_first_char_channels, previous) & allowed;
    const __m512i triples = vpshufb(c_third_char_channels, classes) & shift_in_previous(pairs, *previous_pairs);

    o[M_NEWLINE][lane] = _mm512_test_epi8_mask(properties, splat8(CHAR_NEWLINE));
    o[M_WHITESPACE][lane] = _mm512_test_epi8_mask(properties, splat8(CHAR_WHITESPACE));
    o[M_IDENTIFIER][lane] = _mm512_test_epi8_mask(properties, splat8(CHAR_IDENTIFIER));
    o[M_DIGIT][lane] = _mm512_test_epi8_mask(properties, splat8(CHAR_DIGIT));
    o[M_SEPARATOR_FOLLOWER][lane] = _mm512_test_epi8_mask(properties, splat8(CHAR_SEPARATOR_FOLLOWER));
    o[M_QUOTE][lane] = _mm512_cmpeq_epi8_mask(raw, splat8('"'));
    o[M_APOSTROPHE][lane] = _mm512_cmpeq_epi8_mask(raw, splat8('\''));
    o[M_BACKSLASH][lane] = _mm512_cmpeq_epi8_mask(raw, splat8('\\'));
    o[M_SLASH][lane] = _mm512_cmpeq_epi8_mask(raw, splat8('/'));
    o[M_STAR][lane] = _mm512_cmpeq_epi8_mask(raw, splat8('*'));
    o[M_DOT][lane] = _mm512_cmpeq_epi8_mask(raw, splat8('.'));
    o[M_EXPSIGN][lane] = _mm512_test_epi8_mask(pairs, splat8(CHANNEL_EXPSIGN));
    o[M_HASHDIG][lane] = _mm512_test_epi8_mask(pairs, splat8(CHANNEL_HASHDIG));
    o[M_OP2][lane] = _mm512_test_epi8_mask(pairs, splat8(OP2_CHANNELS));
    o[M_OP3][lane] = _mm512_test_epi8_mask(triples, triples);
    o[M_NONASCII][lane] = nonascii;

    *previous_classes = classes;
    *previous_pairs = pairs;
}

// The byte at `position` of the source, or a newline outside of [`begin`, `length`).
INLINE uint8_t source_byte(const uint8_t* source, uint64_t begin, uint64_t length, int64_t position)
{
    return position >= (int64_t)begin && position < (int64_t)length ? source[position] : '\n';
}

// The kind of the prefixed literal whose prefix starts at `position`: L, u8, u or U and then a quote.
static uint8_t prefixed_literal_kind(const uint8_t* source, uint64_t begin, uint64_t length, int64_t position)
{
    const uint8_t letter = source_byte(source, begin, length, position);
    int64_t opener = position + 1;
    uint8_t prefix = letter == 'L' ? 1 : letter == 'U' ? 4 : 3;
    if (letter == 'u' && source_byte(source, begin, length, position + 1) == '8')
    {
        prefix = 2;
        opener = position + 2;
    }
    const uint8_t base = source_byte(source, begin, length, opener) == '"' ? C_TOKEN_STRING_LITERAL : C_TOKEN_CHAR_CONSTANT;
    return (uint8_t)(base + prefix);
}

// The literals and comments of a chunk, found line by line in parallel: each cursor seeks the next delimiter on its
// line, then the end of what that starts. A block comment can end on a later line; then whatever was found on the
// lines it covers is forgotten, and the line it ends on is scanned again from its end.
typedef struct Scan Scan;
struct Scan
{
    u64x8 literal_starts;
    u64x8 literal_closers;
    u64x8 literal_unterminated; // The newline that ends an unterminated literal.
    u64x8 line_comment_starts;
    u64x8 block_region;
    u64x8 block_ends;
    bool spans_lines; // A block comment spans lines, so some line may have been scanned from the wrong state.
};

INLINE Scan scan_lines(u64x8 cursors, State state, uint32_t block_comment_skip, u64x8 line_break, u64x8 bounds, u64x8 quote_bounds,
                       u64x8 apostrophe_closers, u64x8 apostrophe_bounds, u64x8 line_comment_marks, u64x8 block_comment_marks,
                       u64x8 block_comment_closers)
{
    const u64x8 zero = {0};
    const u64x8 bit_0 = {1, 0, 0, 0, 0, 0, 0, 0};
    Scan result = { .literal_starts = zero, .literal_closers = zero, .literal_unterminated = zero, .line_comment_starts = zero,
                    .block_region = zero, .block_ends = zero, .spans_lines = false };
    u64x8 pseudo_quote = state == STATE_STRING ? bit_0 : zero;
    u64x8 pseudo_apostrophe = state == STATE_CHAR ? bit_0 : zero;
    u64x8 pseudo_block_cursor = zero;
    u64x8 pseudo_block_start = zero;
    if (state == STATE_BLOCK_COMMENT)
    {
        pseudo_block_cursor[0] = 1ull << block_comment_skip;
        pseudo_block_start = bit_0;
    }
    const u64x8 every_line_start = shl(line_break, 1) | bit_0;
    for (;;)
    {
        const u64x8 starts = bounds & ~sub(bounds | line_break, cursors);
        const u64x8 quote_starts = starts & quote_bounds;
        const u64x8 apostrophe_starts = starts & apostrophe_bounds;
        const u64x8 block_opens = starts & block_comment_marks;
        result.line_comment_starts |= starts & line_comment_marks;

        // A literal ends at its closing quote, or (unterminated) at the end of its line.
        u64x8 next_cursors = zero;
        const u64x8 quote_lines = quote_starts | pseudo_quote;
        const u64x8 apostrophe_lines = apostrophe_starts | pseudo_apostrophe;
        if (!is_zero(quote_lines | apostrophe_lines))
        {
            const u64x8 closers = line_break | (quote_bounds & sub(line_break, quote_lines)) | (apostrophe_closers & sub(line_break, apostrophe_lines));
            const u64x8 ends = closers & ~sub(closers, shl(quote_starts | apostrophe_starts, 1) | pseudo_quote | pseudo_apostrophe);
            result.literal_starts |= quote_starts | apostrophe_starts;
            result.literal_closers |= ends & ~line_break;
            result.literal_unterminated |= ends & line_break;
            next_cursors = shl(ends & ~line_break, 1);
        }

        // A block comment ends at the first "*/" whose '*' isn't its own.
        if (!is_zero(block_opens | pseudo_block_start))
        {
            // Several searches can end at the same "*/" (one from each line of a comment that seems to start a
            // comment); one that starts on it must not take part in the subtraction, or the others would skip it.
            // (The search of a "/*" in the last 3 bits starts in the next chunk.)
            const u64x8 searches = shl(block_opens, 3) | pseudo_block_cursor;
            const u64x8 ends = (block_comment_closers & ~sub(block_comment_closers, searches & ~block_comment_closers)) | (searches & block_comment_closers);
            const u64x8 region = sub(ends, block_opens | pseudo_block_start) | block_opens | pseudo_block_start | ends;
            if (!is_zero(region & line_break))
            {
                // The rest of each line a comment ends on, with the newline that ends it. (Two comments can end on
                // the same line, so the searches that start on a newline are kept out of the subtraction, and the
                // subtraction's holes at the other searches are filled in.)
                const u64x8 after_ends = shl(ends, 1);
                const u64x8 rest_of_line = (sub(line_break, after_ends & ~line_break) ^ line_break) | after_ends;
                const u64x8 keep = ~(region | rest_of_line);
                result.literal_starts &= keep;
                result.literal_closers &= keep;
                result.literal_unterminated &= keep;
                result.line_comment_starts &= keep;
                result.block_region &= keep;
                next_cursors &= keep;
                result.spans_lines = true;
            }
            result.block_region |= region;
            result.block_ends |= ends;
            next_cursors |= shl(ends, 1);
            // One cursor per line: a later one on the same line is redone by the first.
            next_cursors &= ~sub(next_cursors | line_break, every_line_start);
        }

        cursors = next_cursors;
        pseudo_quote = zero;
        pseudo_apostrophe = zero;
        pseudo_block_cursor = zero;
        pseudo_block_start = zero;
        if (is_zero(cursors))
        {
            break;
        }
    }
    return result;
}

static int64_t tokenize_chunks(const uint8_t* source, uint64_t length, uint8_t* restrict kinds_out, uint32_t* restrict starts_out,
                               uint32_t* restrict ends_out)
{
    const u64x8 zero = {0};
    const u64x8 bit_0 = {1, 0, 0, 0, 0, 0, 0, 0};
    const u64x8 most_significant = {0, 0, 0, 0, 0, 0, 0, 1ull << 63};
    const u64x8 odd_bits = 0xAAAAAAAAAAAAAAAAull;

    // The source is read in 64-byte-aligned lanes; the bytes before it (and its UTF-8 byte order mark) and after it
    // read as newlines. Positions are relative to the source, so the first chunk can start at a negative one.
    const uint64_t begin = length >= 3 && source[0] == 0xEF && source[1] == 0xBB && source[2] == 0xBF ? 3 : 0;
    const uint8_t* const base = (const uint8_t*)((uintptr_t)source & ~(uintptr_t)63);
    const int64_t base_offset = (int64_t)(base - source);

    // What the previous chunk passes on. Bitstrings only matter for their top bits.
    u64x8 previous_identifier = zero;
    u64x8 previous_apostrophe = zero;
    u64x8 previous_star = zero;
    u64x8 previous_number_token = zero;
    u64x8 previous_identifier_token = zero;
    u64x8 previous_splice_backslash = zero;
    u64x8 previous_splice_cr = zero;
    u64x8 next_is_escaped = zero; // At bit 0.
    uint32_t number_carry = 0; // A pp-number continues into the chunk.
    State state = STATE_OUTSIDE;
    uint32_t block_comment_skip = 0; // A "*/" in bits below this can't close the block comment the chunk starts in.
    bool pending_start_of_line = true;
    uint64_t covered = 0; // The bits at the start of the chunk that a punctuator from the previous chunk covers,
    uint64_t covered_end = 0; // and the last of them.
    uint64_t suppressed_starts = 0; // The quote of a prefixed literal that starts in the previous chunk,
    uint64_t suppressed_ends = 0; // and the '8' of its "u8" prefix.

    uint8_t* cur_kind = kinds_out;
    uint32_t* cur_start = starts_out;
    uint32_t* cur_end = ends_out;

    // Phase I runs a chunk ahead: the masks it stores 8 bytes at a time are loaded as 64-byte bitstrings, which the
    // store buffer can't forward, so they are only loaded an iteration after they're stored. Each buffer holds a
    // chunk's 64-byte vectors and their masks (`masks[...][mask][lane]`), with the next chunk's first lane as lane 8.
    uint64_t masks[2][NUM_MASKS][16] __attribute__((aligned(64)));
    __m512i raws[2][9];
    __m512i previous_classes = _mm512_setzero_si512();
    __m512i previous_pairs = _mm512_setzero_si512();
    for (unsigned j = 0; j < 9; j += 1)
    {
        raws[0][j] = load_lane(base, base_offset, begin, length, 64 * j);
        classify(raws[0][j], masks[0], j, &previous_classes, &previous_pairs);
    }

    for (int64_t chunk = 0, current = 0;; chunk += 512, current ^= 1)
    {
        uint64_t (*o)[16] = masks[current];
        const __m512i* raw = raws[current];
        const int64_t chunk_position = base_offset + chunk; // The source position of bit 0.
        const bool last_chunk = chunk_position + 512 >= (int64_t)length;

        // I. Classification of the next chunk (its lane 0 is this chunk's lane 8).
        if (LIKELY(!last_chunk))
        {
            uint64_t (*next_o)[16] = masks[current ^ 1];
            __m512i* next_raw = raws[current ^ 1];
            next_raw[0] = raw[8];
            for (unsigned m = 0; m < NUM_MASKS; m += 1)
            {
                next_o[m][0] = o[m][8];
            }
            for (unsigned j = 1; j < 9; j += 1)
            {
                next_raw[j] = load_lane(base, base_offset, begin, length, chunk + 512 + 64 * j);
                classify(next_raw[j], next_o, j, &previous_classes, &previous_pairs);
            }
        }
        u64x8 padding = zero;
        uint64_t next_padding = 0;
        if (UNLIKELY(chunk_position + 576 > (int64_t)length))
        {
            for (unsigned j = 0; j < 8; j += 1)
            {
                padding[j] = lane_padding(base_offset, length, chunk + 64 * j);
            }
            next_padding = lane_padding(base_offset, length, chunk + 512);
        }

#define MASK(m) (*(const u64x8*)o[m])
#define NEXT(m) (o[m][8])
        const u64x8 newline = MASK(M_NEWLINE);
        const u64x8 whitespace = MASK(M_WHITESPACE);
        const u64x8 identifier = MASK(M_IDENTIFIER);
        const u64x8 digit = MASK(M_DIGIT);
        const u64x8 separator_follower = MASK(M_SEPARATOR_FOLLOWER);
        const u64x8 quote = MASK(M_QUOTE);
        const u64x8 apostrophe = MASK(M_APOSTROPHE);
        const u64x8 backslash = MASK(M_BACKSLASH);
        const u64x8 slash = MASK(M_SLASH);
        const u64x8 star = MASK(M_STAR);
        const u64x8 dot = MASK(M_DOT);
        const u64x8 exponent_sign = MASK(M_EXPSIGN);
        const u64x8 hash_digraph = MASK(M_HASHDIG);
        const u64x8 pair_ends = MASK(M_OP2);
        const u64x8 triple_ends = MASK(M_OP3);
        const u64x8 nonascii = MASK(M_NONASCII);

        // II. Analysis

        // Line splices: a backslash right before a newline (not the end of the source). They don't break lines and
        // read as whitespace, which is right unless the characters around one could be part of the same token.
        const u64x8 splice_backslash = backslash & shr_next(newline & ~padding, NEXT(M_NEWLINE) & ~next_padding, 1);
        u64x8 splice = zero;
        u64x8 joining_splices = zero; // Splices between characters that would make one token outside of literals and comments,
        u64x8 closing_splices = zero; // and between the '*' and '/' that would end a block comment.
        if (UNLIKELY(!is_zero(splice_backslash | previous_splice_backslash | previous_splice_cr)))
        {
            u64x8 cr;
            for (unsigned j = 0; j < 8; j += 1)
            {
                cr[j] = _mm512_cmpeq_epi8_mask(raw[j], splat8('\r'));
            }
            const uint64_t next_cr = _mm512_cmpeq_epi8_mask(raw[8], splat8('\r'));
            const u64x8 line_feed = newline & ~padding & ~cr;
            const u64x8 first_newline = shl_carry(splice_backslash, previous_splice_backslash, 1);
            const u64x8 second_newline = shl_carry(first_newline & cr, previous_splice_cr, 1) & line_feed;
            // In a splice, "\n\r" is one newline.
            if (!is_zero(first_newline & line_feed & shr_next(cr, next_cr, 1)))
            {
                return NEEDS_SCALAR;
            }
            splice = splice_backslash | first_newline | second_newline;
            for (unsigned j = 0; j < 8; j += 1)
            {
                for (uint64_t bits = splice_backslash[j]; bits != 0; bits &= bits - 1)
                {
                    const int64_t position = chunk_position + 64 * j + __builtin_ctzll(bits);
                    const uint8_t before = source_byte(source, begin, length, position - 1);
                    const bool crlf = source_byte(source, begin, length, position + 1) == '\r' && source_byte(source, begin, length, position + 2) == '\n';
                    const uint8_t after = source_byte(source, begin, length, position + (crlf ? 3 : 2));
                    // A splice inside an escape sequence, or after an apostrophe (which can make it a digit
                    // separator, or an empty character constant).
                    if (before == '\\' || before == '\'')
                    {
                        return NEEDS_SCALAR;
                    }
                    if (splice_joins(source_byte(source, begin, length, position - 2), before, after))
                    {
                        joining_splices[j] |= bits & -bits;
                    }
                    if (before == '*' && (after == '/' || after == '\\'))
                    {
                        closing_splices[j] |= bits & -bits;
                    }
                }
            }
            previous_splice_backslash = splice_backslash & most_significant;
            previous_splice_cr = first_newline & cr & most_significant;
        }

        // A backslash, whitespace and a newline is a line splice too (that's rare).
        const u64x8 backslash_whitespace = backslash & shr_next(whitespace, NEXT(M_WHITESPACE), 1);
        if (UNLIKELY(!is_zero(backslash_whitespace)))
        {
            for (unsigned j = 0; j < 8; j += 1)
            {
                for (uint64_t bits = backslash_whitespace[j]; bits != 0; bits &= bits - 1)
                {
                    int64_t p = chunk_position + 64 * j + __builtin_ctzll(bits) + 1;
                    while (p < (int64_t)length && (source[p] == ' ' || source[p] == '\t' || source[p] == '\v' || source[p] == '\f'))
                    {
                        p += 1;
                    }
                    if (p < (int64_t)length && (source[p] == '\n' || source[p] == '\r'))
                    {
                        return NEEDS_SCALAR;
                    }
                }
            }
        }

        const u64x8 line_break = newline & ~splice;

        // Matches the character after any odd number of backslashes (simdjson's algorithm by John Keiser).
        u64x8 escaped;
        {
            const u64x8 escape_backslash = backslash & ~splice_backslash;
            const u64x8 potential_escape = escape_backslash & ~next_is_escaped;
            const u64x8 maybe_escaped = shl(potential_escape, 1);
            const u64x8 even_series_codes_and_odd_bits = sub(maybe_escaped | odd_bits, potential_escape);
            const u64x8 escape_and_terminal_code = even_series_codes_and_odd_bits ^ odd_bits;
            escaped = escape_and_terminal_code ^ (escape_backslash | next_is_escaped);
            next_is_escaped = (u64x8){bit_511(escape_and_terminal_code & escape_backslash), 0, 0, 0, 0, 0, 0, 0};
        }

        // pp-numbers: a digit (or a '.' and a digit) that doesn't continue an identifier, and everything that
        // continues it: identifier characters, '.', a sign after e/E/p/P and a digit separator.
        const u64x8 apostrophe_separator = apostrophe & shr_next(separator_follower, NEXT(M_SEPARATOR_FOLLOWER), 1);
        const u64x8 number_sign = exponent_sign & ~shl_carry(apostrophe, previous_apostrophe, 2); // Not "1'e+".
        const u64x8 number_continue = identifier | dot | number_sign | apostrophe_separator;
        const u64x8 number_head = (digit & ~shl_carry(identifier, previous_identifier, 1)) | (dot & shr_next(digit, NEXT(M_DIGIT), 1));
        uint32_t number_carry_out;
        const u64x8 number_sum = add_carry(number_continue, number_head, number_carry, &number_carry_out);
        const u64x8 number_region = ((number_sum ^ number_continue) | number_head) & number_continue;

        // Delimiters of literals and comments. A digit separator isn't a quote. Comment delimiters are marked at
        // their '/': the first character of "//" and "/*", and the last of "*/".
        const u64x8 quote_bounds = quote & ~escaped;
        const u64x8 apostrophe_closers = apostrophe & ~escaped;
        const u64x8 apostrophe_bounds = apostrophe_closers & ~number_region;
        const u64x8 line_comment_marks = slash & shr_next(slash, NEXT(M_SLASH), 1);
        const u64x8 block_comment_marks = slash & shr_next(star, NEXT(M_STAR), 1);
        const u64x8 block_comment_closers = slash & shl_carry(star, previous_star, 1);
        const uint64_t next_closers = NEXT(M_SLASH) & ((NEXT(M_STAR) << 1) | bit_511(star));
        if (UNLIKELY(!is_zero((block_comment_marks & block_comment_closers) | (block_comment_marks & shr_next(block_comment_closers, next_closers, 2)))))
        {
            return NEEDS_SCALAR; // "*/*" or "/*/"
        }

        // Find the literals and comments, line by line in parallel (see `scan_lines`). The scan starts each line
        // outside of comments; if a block comment spans lines, that can be wrong, so the chunk is scanned again with
        // each line's state from the previous scan until the two agree (which is then the sequential result).
        const u64x8 bounds = quote_bounds | apostrophe_bounds | line_comment_marks | block_comment_marks;
        const u64x8 scan_cursors = shl(line_break, 1) | (state == STATE_OUTSIDE ? bit_0 : zero);
        Scan scan = scan_lines(scan_cursors, state, block_comment_skip, line_break, bounds, quote_bounds, apostrophe_closers, apostrophe_bounds,
                               line_comment_marks, block_comment_marks, block_comment_closers);
        if (UNLIKELY(scan.spans_lines))
        {
            for (unsigned round = 0;; round += 1)
            {
                if (round == 4)
                {
                    return NEEDS_SCALAR;
                }
                // A line that starts inside a block comment is scanned from the end of the comment, if it's on the line.
                const u64x8 inside = shl(line_break & scan.block_region, 1);
                const u64x8 seek = block_comment_closers | line_break;
                const u64x8 first_closers = block_comment_closers & ~sub(seek, inside);
                const Scan rescan = scan_lines((scan_cursors & ~inside) | shl(first_closers, 1), state, block_comment_skip, line_break, bounds, quote_bounds,
                                               apostrophe_closers, apostrophe_bounds, line_comment_marks, block_comment_marks, block_comment_closers);
                const bool same = is_zero((rescan.block_region ^ scan.block_region) & line_break) &&
                                  (bit_511(rescan.block_region) & ~bit_511(rescan.block_ends)) == (bit_511(scan.block_region) & ~bit_511(scan.block_ends));
                scan = rescan;
                if (same)
                {
                    break;
                }
            }
        }
        u64x8 literal_starts = scan.literal_starts;
        const u64x8 literal_closers = scan.literal_closers;
        u64x8 literal_unterminated = scan.literal_unterminated;
        const u64x8 line_comment_starts = scan.line_comment_starts;
        const u64x8 block_region = scan.block_region;
        const u64x8 block_ends = scan.block_ends;

        const u64x8 carried_literal = state == STATE_STRING || state == STATE_CHAR ? bit_0 : zero;
        const u64x8 line_comment_region = sub(line_break, line_comment_starts | (state == STATE_LINE_COMMENT ? bit_0 : zero)) & ~line_break;
        const u64x8 literal_region = sub(literal_closers | literal_unterminated, literal_starts | carried_literal) | literal_closers;
        const u64x8 outside = ~(literal_region | line_comment_region | block_region);

        // What the next chunk starts inside of. A literal open at the end of the chunk is unterminated if the next
        // chunk starts with a newline (the end of the source counts): then it ends here.
        State next_state = STATE_OUTSIDE;
        uint32_t next_block_comment_skip = 0;
        bool unterminated_at_end = false;
        if (bit_511(block_region) && !bit_511(block_ends))
        {
            if (last_chunk)
            {
                return NEEDS_SCALAR; // An unterminated block comment.
            }
            next_state = STATE_BLOCK_COMMENT;
            // A "*/" can't use the '*' of a "/*" that starts at bit 509, 510 or 511.
            const uint64_t opener_bits = (block_comment_marks[7] & block_region[7]) >> 61;
            next_block_comment_skip = opener_bits == 0 ? 0 : 63 - (uint32_t)__builtin_clzll(opener_bits);
        }
        else if (bit_511(literal_region) && !bit_511(literal_closers))
        {
            const int32_t last_start = highest_bit_below(literal_starts, 512);
            const bool is_char = last_start >= 0 ? ((apostrophe[last_start >> 6] >> (last_start & 63)) & 1) != 0 : state == STATE_CHAR;
            next_state = is_char ? STATE_CHAR : STATE_STRING;
            // (The newline that starts the next chunk is part of a splice if this chunk ends with its backslash, or
            // with its "\\\r" and the newline is the '\n' after that.)
            const bool next_splice_newline = bit_511(splice_backslash) ||
                                             (bit_511(previous_splice_cr) && (_mm512_cmpeq_epi8_mask(raw[8], splat8('\n')) & ~next_padding & 1));
            if ((NEXT(M_NEWLINE) & 1) && !next_splice_newline)
            {
                next_state = STATE_OUTSIDE;
                unterminated_at_end = true;
            }
        }
        else if (bit_511(line_comment_region))
        {
            next_state = STATE_LINE_COMMENT;
        }

        // Inputs the bitstrings don't model.
        if (UNLIKELY(!is_zero(((nonascii | joining_splices) & outside) | (closing_splices & block_region))))
        {
            return NEEDS_SCALAR;
        }
        // A stray backslash is a one-character unknown token, unless it starts a UCN (\\u, \\U or \\N) or comes before
        // a quote or another backslash (which the escape bitstring would have it escape).
        const u64x8 stray_backslash = backslash & ~splice_backslash & outside;
        if (UNLIKELY(!is_zero(stray_backslash)))
        {
            u64x8 risky;
            for (unsigned j = 0; j < 8; j += 1)
            {
                risky[j] = _mm512_cmpeq_epi8_mask(raw[j], splat8('u')) | _mm512_cmpeq_epi8_mask(raw[j], splat8('U')) |
                           _mm512_cmpeq_epi8_mask(raw[j], splat8('N')) | quote[j] | apostrophe[j] | backslash[j];
            }
            const uint64_t next_risky = _mm512_cmpeq_epi8_mask(raw[8], splat8('u')) | _mm512_cmpeq_epi8_mask(raw[8], splat8('U')) |
                                        _mm512_cmpeq_epi8_mask(raw[8], splat8('N')) | NEXT(M_QUOTE) | NEXT(M_APOSTROPHE) | NEXT(M_BACKSLASH);
            if (!is_zero(stray_backslash & shr_next(risky, next_risky, 1)))
            {
                return NEEDS_SCALAR;
            }
        }

        // Literals (by their opening quote) that are unknown tokens: unterminated ones, found by mapping the newline
        // that ends each one back to its quote (if it's in this chunk), and empty character constants.
        u64x8 unknown_openers = literal_starts & apostrophe & shr_next(literal_closers, NEXT(M_APOSTROPHE) & 1, 1);
        if (UNLIKELY(!is_zero(literal_unterminated) || unterminated_at_end))
        {
            bool carried_literal_unknown = false;
            for (uint32_t lane = 0; lane < 9; lane += 1)
            {
                uint64_t bits = lane < 8 ? literal_unterminated[lane] : (uint64_t)unterminated_at_end;
                for (; bits != 0; bits &= bits - 1)
                {
                    const int32_t opener = highest_bit_below(literal_starts, lane * 64 + (uint32_t)__builtin_ctzll(bits));
                    // (The quote of a prefixed literal that starts in the previous chunk belongs to that token.)
                    if (opener >= 0 && !(opener < 64 && ((suppressed_starts >> opener) & 1)))
                    {
                        unknown_openers[opener >> 6] |= 1ull << (opener & 63);
                    }
                    else
                    {
                        carried_literal_unknown = true;
                    }
                }
            }
            // The literal the chunk starts inside of is the first token whose end isn't written yet.
            if (carried_literal_unknown)
            {
                const uint64_t index = (uint64_t)(cur_end - ends_out);
                kinds_out[index] = C_TOKEN_UNKNOWN | (kinds_out[index] & C_TOKEN_START_OF_LINE);
            }
        }
        const u64x8 literal_token_ends = literal_closers | shr(literal_unterminated, 1) | (unterminated_at_end ? most_significant : zero);

        // Multi-character punctuators: candidates from Phase I that start outside of literals, comments and
        // pp-numbers. An ellipsis only needs its first '.' outside of a pp-number: "...5" is "..." and "5", even
        // though ".5" looks like a pp-number.
        const u64x8 number_token_with_dots = number_region & outside;
        const u64x8 covered_bits = {covered, 0, 0, 0, 0, 0, 0, 0};
        const u64x8 candidate = outside & ~number_token_with_dots & ~covered_bits;
        const u64x8 hash_digraphs = shr_next(hash_digraph, NEXT(M_HASHDIG), 1); // The '%' of each "%:".
        u64x8 quads = hash_digraphs & shr_next(hash_digraphs, NEXT(M_HASHDIG) >> 1, 2) & candidate;
        u64x8 triples = shr_next(triple_ends, NEXT(M_OP3), 2) & candidate & ~quads;
        u64x8 pairs = shr_next(pair_ends, NEXT(M_OP2), 1) & candidate & ~triples & ~quads;
        u64x8 covers = shl(pairs, 1) | shl(triples, 1) | shl(triples, 2) | shl(quads, 1) | shl(quads, 2) | shl(quads, 3);
        if (UNLIKELY(!is_zero((pairs | triples | quads) & covers)))
        {
            // Overlapping candidates, like "+++": take the longest from the left, then the next one that is left.
            u64x8 open_pairs = pairs;
            u64x8 open_triples = triples;
            u64x8 open_quads = quads;
            pairs = zero;
            triples = zero;
            quads = zero;
            do
            {
                const u64x8 open_covers = shl(open_pairs, 1) | shl(open_triples, 1) | shl(open_triples, 2) | shl(open_quads, 1) |
                                          shl(open_quads, 2) | shl(open_quads, 3);
                const u64x8 take_pairs = open_pairs & ~open_covers;
                const u64x8 take_triples = open_triples & ~open_covers;
                const u64x8 take_quads = open_quads & ~open_covers;
                pairs |= take_pairs;
                triples |= take_triples;
                quads |= take_quads;
                const u64x8 taken = take_pairs | take_triples | take_quads | shl(take_pairs, 1) | shl(take_triples, 1) | shl(take_triples, 2) |
                                    shl(take_quads, 1) | shl(take_quads, 2) | shl(take_quads, 3);
                open_pairs &= ~taken;
                open_triples &= ~taken;
                open_quads &= ~taken;
            } while (!is_zero(open_pairs | open_triples | open_quads));
            covers = shl(pairs, 1) | shl(triples, 1) | shl(triples, 2) | shl(quads, 1) | shl(quads, 2) | shl(quads, 3);
        }
        const u64x8 multi_starts = pairs | triples | quads;
        const u64x8 multi_ends = shl(pairs, 1) | shl(triples, 2) | shl(quads, 3);
        // A punctuator that crosses into the next chunk covers its first bits.
        uint64_t next_covered = 0;
        uint64_t next_covered_end = 0;
        {
            uint32_t overhang = 0; // How many characters of a punctuator starting at bit 509, 510 or 511 are past bit 511.
            if (bit_511(pairs))
            {
                overhang = 1;
            }
            else if ((triples[7] >> 62) != 0)
            {
                overhang = (uint32_t)(triples[7] >> 62) == 1 ? 1 : 2;
            }
            else if ((quads[7] >> 61) != 0)
            {
                overhang = 64 - (uint32_t)__builtin_clzll(quads[7] >> 61);
            }
            if (overhang != 0)
            {
                next_covered = _bzhi_u64(~0ull, overhang);
                next_covered_end = 1ull << (overhang - 1);
            }
        }

        // pp-numbers and identifiers (an identifier character in a pp-number is part of it).
        const u64x8 ellipses = triples & dot;
        const u64x8 number_token = number_token_with_dots & ~(ellipses | shl(ellipses, 1) | shl(ellipses, 2)) & ~covered_bits;
        const u64x8 number_starts = number_token & ~shl_carry(number_token, previous_number_token, 1);
        if (UNLIKELY(!is_zero(number_starts & ~(digit | dot))))
        {
            return NEEDS_SCALAR; // A digit separator that is really a quote, as in 'a'b'c'.
        }
        const uint64_t next_number_continues = (NEXT(M_IDENTIFIER) | NEXT(M_DOT) | (NEXT(M_EXPSIGN) & ~((apostrophe[7] >> 62) & 1)) |
                                                (NEXT(M_APOSTROPHE) & (NEXT(M_SEPARATOR_FOLLOWER) >> 1))) & 1;
        const u64x8 number_ends = number_token & ~shr_next(number_token, next_number_continues, 1);
        const u64x8 identifier_token = identifier & outside & ~number_token;
        const u64x8 identifier_starts = identifier_token & ~shl_carry(identifier_token, previous_identifier_token, 1);
        u64x8 identifier_ends = identifier_token & ~shr_next(identifier_token, NEXT(M_IDENTIFIER) & 1, 1);
        identifier_ends[0] &= ~suppressed_ends;

        // Every other character outside of literals and comments is a one-character token.
        const u64x8 other = ~(identifier | whitespace | newline | quote | apostrophe | backslash | nonascii);
        const u64x8 singles = (other | stray_backslash) & outside & ~number_token & ~multi_starts & ~covers & ~covered_bits;

        // Literals with a prefix (L, u8, u or U) start at the prefix, which would otherwise be an identifier.
        u64x8 literal_starts_token = literal_starts;
        literal_starts_token[0] &= ~suppressed_starts;
        u64x8 prefix_starts = zero;
        u64x8 unknown_starts = zero;
        uint64_t next_suppressed_starts = 0;
        uint64_t next_suppressed_ends = 0;
        const uint64_t next_quotes = (NEXT(M_QUOTE) | NEXT(M_APOSTROPHE)) & 3;
        if (UNLIKELY(!is_zero(literal_starts & shl_carry(identifier_token, previous_identifier_token, 1)) || (bit_511(identifier_token) && next_quotes != 0)))
        {
            u64x8 letter;
            u64x8 u;
            u64x8 eight;
            for (unsigned j = 0; j < 8; j += 1)
            {
                letter[j] = _mm512_cmpeq_epi8_mask(raw[j], splat8('L')) | _mm512_cmpeq_epi8_mask(raw[j], splat8('U'));
                u[j] = _mm512_cmpeq_epi8_mask(raw[j], splat8('u'));
                eight[j] = _mm512_cmpeq_epi8_mask(raw[j], splat8('8'));
            }
            letter |= u;
            const uint64_t next_eight = _mm512_cmpeq_epi8_mask(raw[8], splat8('8'));
            const u64x8 one_letter = identifier_starts & letter & shr_next(literal_starts, next_quotes, 1);
            const u64x8 u8 = identifier_starts & u & shr_next(eight, next_eight, 1) & shr_next(literal_starts, next_quotes, 2);
            prefix_starts = one_letter | u8;
            identifier_ends &= ~(one_letter | shl(u8, 1));
            literal_starts_token &= ~(shl(one_letter, 1) | shl(u8, 2));
            // A prefixed literal is unknown if its quote starts one (an empty character constant in the next chunk
            // is "''" there).
            const uint64_t next_unknown = NEXT(M_APOSTROPHE) & (NEXT(M_APOSTROPHE) >> 1);
            unknown_starts = (one_letter & shr_next(unknown_openers, next_unknown, 1)) | (u8 & shr_next(unknown_openers, next_unknown, 2));
            next_suppressed_starts = bit_511(one_letter) | ((u8[7] >> 62) & 1) | (bit_511(u8) << 1);
            next_suppressed_ends = bit_511(u8);
        }
        unknown_starts |= unknown_openers & literal_starts_token;

        u64x8 all_ends = number_ends | identifier_ends | multi_ends | singles | literal_token_ends;
        all_ends[0] |= covered_end;
        const u64x8 all_starts = number_starts | identifier_starts | multi_starts | singles | literal_starts_token;

        // The first token after each newline that starts a logical line (not one in a block comment).
        const u64x8 line_starts = line_break & ~block_region;
        const u64x8 seek_in = all_starts | line_starts;
        const u64x8 seek = sub(seek_in, shl(line_starts, 1) | (pending_start_of_line ? bit_0 : zero));
        const u64x8 start_of_line = all_starts & ~seek;
        pending_start_of_line = bit_511(line_starts) || bit_511(seek & ~seek_in);

        // III. Write-out
        uint64_t starts_lanes[8];
        uint64_t ends_lanes[8];
        uint64_t start_of_line_lanes[8];
        uint64_t multi_lanes[8];
        uint64_t long_lanes[8];
        uint64_t dot_number_lanes[8];
        uint64_t prefix_lanes[8];
        uint64_t unknown_lanes[8];
        const u64x8 long_ones = triples | quads;
        const u64x8 dot_numbers = number_starts & dot;
        memcpy(starts_lanes, &all_starts, 64);
        memcpy(ends_lanes, &all_ends, 64);
        memcpy(start_of_line_lanes, &start_of_line, 64);
        memcpy(multi_lanes, &multi_starts, 64);
        memcpy(long_lanes, &long_ones, 64);
        memcpy(dot_number_lanes, &dot_numbers, 64);
        memcpy(prefix_lanes, &prefix_starts, 64);
        memcpy(unknown_lanes, &unknown_starts, 64);

        for (unsigned j = 0; j < 8; j += 1)
        {
            // The kind of each character's token if it starts one, then the exceptions.
            __m512i kinds = perm2b(c_char_kinds, raw[j]);
            kinds = _mm512_mask_mov_epi8(kinds, dot_number_lanes[j], splat8(C_TOKEN_NUMERIC_CONSTANT));
            const uint64_t multi = multi_lanes[j];
            if (multi != 0)
            {
                const __m512i indices = bits_to_indices(multi);
                const uint64_t long_multi = _pext_u64(long_lanes[j], multi);
                const __m128i low = punctuator_kinds(raw[j], raw[j + 1], _mm512_castsi512_si128(indices), (uint16_t)long_multi);
                __m128i high = _mm_setzero_si128();
                // More than 16 multi-character punctuators in 64 bytes is rare (at most 32 fit).
                if (UNLIKELY(__builtin_popcountll(multi) > 16))
                {
                    high = punctuator_kinds(raw[j], raw[j + 1], _mm512_extracti32x4_epi32(indices, 1), (uint16_t)(long_multi >> 16));
                }
                kinds = _mm512_mask_expand_epi8(kinds, multi, _mm512_castsi256_si512(_mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1)));
            }
            kinds = _mm512_mask_mov_epi8(kinds, start_of_line_lanes[j], _mm512_or_si512(kinds, splat8(C_TOKEN_START_OF_LINE)));
            _mm512_storeu_si512(cur_kind, _mm512_maskz_compress_epi8(starts_lanes[j], kinds));

            const int64_t lane_position = chunk_position + 64 * j;
            // Prefixed and unknown literals (rare).
            for (uint64_t bits = prefix_lanes[j] | unknown_lanes[j]; UNLIKELY(bits != 0); bits &= bits - 1)
            {
                const uint32_t bit = (uint32_t)__builtin_ctzll(bits);
                const uint64_t index = (uint64_t)__builtin_popcountll(starts_lanes[j] & _bzhi_u64(~0ull, bit));
                const uint8_t kind = (unknown_lanes[j] >> bit) & 1 ? C_TOKEN_UNKNOWN : prefixed_literal_kind(source, begin, length, lane_position + bit);
                cur_kind[index] = kind | (cur_kind[index] & C_TOKEN_START_OF_LINE);
            }

            // The starts and ends, 16 at a time: a 64-byte lane has 9 tokens on average, so the first 16 usually do.
            const uint64_t num_starts = (uint64_t)__builtin_popcountll(starts_lanes[j]);
            const uint64_t num_ends = (uint64_t)__builtin_popcountll(ends_lanes[j]);
            const __m512i start_indices = bits_to_indices(starts_lanes[j]);
            const __m512i end_indices = bits_to_indices(ends_lanes[j]);
            const __m512i start_offset = _mm512_set1_epi32((int)(uint32_t)lane_position);
            const __m512i end_offset = _mm512_set1_epi32((int)(uint32_t)(lane_position + 1)); // Exclusive.
            _mm512_storeu_si512(cur_start, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(start_indices)), start_offset));
            _mm512_storeu_si512(cur_end, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(end_indices)), end_offset));
            if (UNLIKELY((num_starts | num_ends) > 16))
            {
                _mm512_storeu_si512(cur_start + 16, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(start_indices, 1)), start_offset));
                _mm512_storeu_si512(cur_start + 32, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(start_indices, 2)), start_offset));
                _mm512_storeu_si512(cur_start + 48, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(start_indices, 3)), start_offset));
                _mm512_storeu_si512(cur_end + 16, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(end_indices, 1)), end_offset));
                _mm512_storeu_si512(cur_end + 32, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(end_indices, 2)), end_offset));
                _mm512_storeu_si512(cur_end + 48, _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(end_indices, 3)), end_offset));
            }

            cur_kind += num_starts;
            cur_start += num_starts;
            cur_end += num_ends;
        }

        previous_identifier = identifier;
        previous_apostrophe = apostrophe;
        previous_star = star;
        previous_number_token = number_token;
        previous_identifier_token = identifier_token;
        number_carry = number_carry_out;
        state = next_state;
        block_comment_skip = next_block_comment_skip;
        covered = next_covered;
        covered_end = next_covered_end;
        suppressed_starts = next_suppressed_starts;
        suppressed_ends = next_suppressed_ends;
#undef MASK
#undef NEXT

        if (last_chunk)
        {
            break;
        }
    }

    return cur_kind - kinds_out;
}

int64_t c_tokenize_vectorized(const uint8_t* source, uint64_t length, uint8_t* kinds, uint32_t* starts, uint32_t* ends, uint32_t flags)
{
    int64_t result;
    if (length > UINT32_MAX - 128)
    {
        result = C_TOKENIZER_ERROR_FILE_TOO_BIG;
    }
    else if (length == 0)
    {
        result = 0;
    }
    else if ((flags & C_TOKENIZER_TRIGRAPHS) && contains_trigraph(source, length))
    {
        result = C_TOKENIZER_ERROR_NEEDS_SCALAR;
    }
    else
    {
        result = tokenize_chunks(source, length, kinds, starts, ends);
    }
    return result;
}

int64_t c_tokenize(const uint8_t* source, uint64_t length, uint8_t* kinds, uint32_t* starts, uint32_t* ends, uint32_t flags)
{
    int64_t result = c_tokenize_vectorized(source, length, kinds, starts, ends, flags);
    if (result == C_TOKENIZER_ERROR_NEEDS_SCALAR)
    {
        result = c_tokenize_scalar(source, length, kinds, starts, ends, flags);
    }
    return result;
}
