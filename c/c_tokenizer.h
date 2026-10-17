// A compacting AVX-512 tokenizer for C and C++ source code that hasn't been preprocessed yet: it produces the
// preprocessing tokens of translation phase 3 (after line splicing and, optionally, trigraph replacement), for a
// preprocessor to run on afterwards. It is the 512-byte-chunk design of src/CompactingTokenizer.zig applied to C.
//
// Every kind of source is tokenized exactly like clang's raw lexer in C23 mode (`clang -cc1 -std=c23
// -dump-raw-tokens`), or in C++26 mode with `C_TOKENIZER_CPLUSPLUS` (`clang -cc1 -x c++ -std=c++26
// -dump-raw-tokens`): line splices anywhere, digraphs, UCNs and UTF-8 identifiers, digit separators, unterminated
// literals and stray characters, and in C++ raw string literals, user-defined literal suffixes (which are part of
// the literal's token), `<=>`, `->*`, `.*` and the `<::` rule. Common code takes the vectorized path; the rare
// lines it can't handle exactly (e.g. a line splice in the middle of a token, a non-ASCII identifier or a raw string
// literal) are tokenized by a scalar lexer.
//
// Differences from the raw lexer's output, by design:
//   - Whitespace and comments aren't tokens. An unterminated block comment is a `C_TOKEN_UNKNOWN` token, as in clang.
//   - A token's start is its first character after any line splices (clang starts it at the first splice).
//   - Digraphs have the kind of the punctuator they stand for (`<:` is `C_TOKEN_L_SQUARE`), like clang.
//   - Header names aren't recognized: after `#include`, re-lex the `<...>` form from the source starting at the `<`.
//   - A named universal character name `\N{...}` is accepted in an identifier if the name is written in capital
//     letters, digits, spaces and hyphens; clang also checks that the name exists.
//
// Requires Clang and AVX-512 F/BW/VL/VBMI/VBMI2/VPOPCNTDQ plus BMI2 (e.g. -march=znver4, znver5 or tigerlake).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The kinds of preprocessing tokens. Each name is the clang token kind it matches (e.g. `C_TOKEN_STRING_LITERAL` is
// clang's `string_literal`). The prefixed character constants and string literals follow their unprefixed kind in
// the order L, u8, u, U (a raw string literal has the kind of its prefix without the R). The last three are C++'s.
#define C_TOKEN_KINDS(X) \
    X(UNKNOWN, unknown) \
    X(RAW_IDENTIFIER, raw_identifier) \
    X(NUMERIC_CONSTANT, numeric_constant) \
    X(CHAR_CONSTANT, char_constant) \
    X(WIDE_CHAR_CONSTANT, wide_char_constant) \
    X(UTF8_CHAR_CONSTANT, utf8_char_constant) \
    X(UTF16_CHAR_CONSTANT, utf16_char_constant) \
    X(UTF32_CHAR_CONSTANT, utf32_char_constant) \
    X(STRING_LITERAL, string_literal) \
    X(WIDE_STRING_LITERAL, wide_string_literal) \
    X(UTF8_STRING_LITERAL, utf8_string_literal) \
    X(UTF16_STRING_LITERAL, utf16_string_literal) \
    X(UTF32_STRING_LITERAL, utf32_string_literal) \
    X(L_SQUARE, l_square) \
    X(R_SQUARE, r_square) \
    X(L_PAREN, l_paren) \
    X(R_PAREN, r_paren) \
    X(L_BRACE, l_brace) \
    X(R_BRACE, r_brace) \
    X(PERIOD, period) \
    X(ELLIPSIS, ellipsis) \
    X(AMP, amp) \
    X(AMPAMP, ampamp) \
    X(AMPEQUAL, ampequal) \
    X(STAR, star) \
    X(STAREQUAL, starequal) \
    X(PLUS, plus) \
    X(PLUSPLUS, plusplus) \
    X(PLUSEQUAL, plusequal) \
    X(MINUS, minus) \
    X(ARROW, arrow) \
    X(MINUSMINUS, minusminus) \
    X(MINUSEQUAL, minusequal) \
    X(TILDE, tilde) \
    X(EXCLAIM, exclaim) \
    X(EXCLAIMEQUAL, exclaimequal) \
    X(SLASH, slash) \
    X(SLASHEQUAL, slashequal) \
    X(PERCENT, percent) \
    X(PERCENTEQUAL, percentequal) \
    X(LESS, less) \
    X(LESSLESS, lessless) \
    X(LESSEQUAL, lessequal) \
    X(LESSLESSEQUAL, lesslessequal) \
    X(GREATER, greater) \
    X(GREATERGREATER, greatergreater) \
    X(GREATEREQUAL, greaterequal) \
    X(GREATERGREATEREQUAL, greatergreaterequal) \
    X(CARET, caret) \
    X(CARETEQUAL, caretequal) \
    X(PIPE, pipe) \
    X(PIPEPIPE, pipepipe) \
    X(PIPEEQUAL, pipeequal) \
    X(QUESTION, question) \
    X(COLON, colon) \
    X(COLONCOLON, coloncolon) \
    X(SEMI, semi) \
    X(EQUAL, equal) \
    X(EQUALEQUAL, equalequal) \
    X(COMMA, comma) \
    X(HASH, hash) \
    X(HASHHASH, hashhash) \
    X(PERIODSTAR, periodstar) \
    X(ARROWSTAR, arrowstar) \
    X(SPACESHIP, spaceship)

typedef enum CTokenKind
{
#define C_TOKEN_ENUM(name, clang_name) C_TOKEN_##name,
    C_TOKEN_KINDS(C_TOKEN_ENUM)
#undef C_TOKEN_ENUM
    C_TOKEN_KIND_COUNT,
} CTokenKind;

// Set in a token's kind byte if it's the first token of a logical source line (comments don't count as tokens), so a
// `#` with this bit starts a preprocessing directive, which ends before the next token with this bit.
#define C_TOKEN_START_OF_LINE 0x80

typedef enum CTokenizerFlags
{
    // Replace trigraphs (`??=` is `#`, ...) in translation phase 1, as C17 and C++14 and earlier require. Off by
    // default, like C23, C++17 and clang's GNU modes.
    C_TOKENIZER_TRIGRAPHS = 1,
    // Tokenize C++ (C++26) instead of C.
    C_TOKENIZER_CPLUSPLUS = 2,
} CTokenizerFlags;

typedef enum CTokenizerError
{
    C_TOKENIZER_ERROR_FILE_TOO_BIG = -1, // Token positions are 32-bit.
    C_TOKENIZER_ERROR_NEEDS_SCALAR = -2, // Only from c_tokenize_vectorized.
} CTokenizerError;

// The number of entries `kinds`, `starts` and `ends` must each have room for when tokenizing `length` bytes.
#define C_TOKENIZER_CAPACITY(length) ((length) + 128)

// Tokenizes `length` bytes of C source. Token `i` is `source[starts[i] .. ends[i]]`, of kind `kinds[i] &
// ~C_TOKEN_START_OF_LINE` (a `CTokenKind`). Returns the number of tokens, or a negative `CTokenizerError`. `source`
// needs no padding or terminator, and `kinds`, `starts` and `ends` need `C_TOKENIZER_CAPACITY(length)` entries.
int64_t c_tokenize(const uint8_t* source, uint64_t length, uint8_t* kinds, uint32_t* starts, uint32_t* ends, uint32_t flags);

// c_tokenize without the scalar lexer: returns C_TOKENIZER_ERROR_NEEDS_SCALAR for the rare inputs that need it for a
// part (see c_tokenizer.c), with the output arrays clobbered. For testing and benchmarking.
int64_t c_tokenize_vectorized(const uint8_t* source, uint64_t length, uint8_t* kinds, uint32_t* starts, uint32_t* ends, uint32_t flags);

// The same tokenization by a straightforward scalar lexer (which `c_tokenize` uses for the rare parts of a source
// that its vectorized path doesn't handle). It writes exactly the returned number of tokens.
int64_t c_tokenize_scalar(const uint8_t* source, uint64_t length, uint8_t* kinds, uint32_t* starts, uint32_t* ends, uint32_t flags);

// The clang name of a token kind (without the start-of-line bit), e.g. "raw_identifier" or "plusequal".
const char* c_token_kind_name(uint8_t kind);

#ifdef __cplusplus
}
#endif
