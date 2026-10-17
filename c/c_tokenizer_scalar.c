// The scalar C lexer: the reference that c_tokenize's vectorized path is checked against, and what it falls back on.
// It follows clang's raw lexer (clang/lib/Lex/Lexer.cpp) in C23 mode, and the functions are named after clang's.
#include "c_tokenizer.h"
#include "c_tokenizer_unicode.h"

#include <stdbool.h>
#include <string.h>

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof((x)[0]))
#define END_OF_FILE (-1)
// The code point of a named UCN (`\N{...}`) whose name looks valid. See the header.
#define NAMED_UCN 0xFFFFFFFFu

typedef struct ScalarLexer ScalarLexer;
struct ScalarLexer
{
    const uint8_t* source;
    uint64_t length;
    bool trigraphs;
    bool cplusplus;
};

typedef struct TokenOutput TokenOutput;
struct TokenOutput
{
    uint8_t* kinds;
    uint32_t* starts;
    uint32_t* ends;
    uint64_t count;
    bool start_of_line;
};

static bool is_whitespace(int32_t c)
{
    return c == ' ' || c == '\t' || c == '\f' || c == '\v' || c == '\n' || c == '\r';
}

static bool is_digit(int32_t c)
{
    return c >= '0' && c <= '9';
}

static bool is_ascii_identifier_continue(int32_t c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || is_digit(c) || c == '_';
}

static int32_t hex_digit_value(int32_t c)
{
    int32_t result = -1;
    if (is_digit(c))
    {
        result = c - '0';
    }
    else if (c >= 'a' && c <= 'f')
    {
        result = c - 'a' + 10;
    }
    else if (c >= 'A' && c <= 'F')
    {
        result = c - 'A' + 10;
    }
    return result;
}

static int32_t decode_trigraph(uint8_t c)
{
    int32_t result = 0;
    switch (c)
    {
        case '=': result = '#'; break;
        case '/': result = '\\'; break;
        case '\'': result = '^'; break;
        case '(': result = '['; break;
        case ')': result = ']'; break;
        case '!': result = '|'; break;
        case '<': result = '{'; break;
        case '>': result = '}'; break;
        case '-': result = '~'; break;
    }
    return result;
}

static bool code_point_in(const uint32_t (*ranges)[2], uint64_t count, uint32_t code_point)
{
    uint64_t low = 0;
    uint64_t high = count;
    while (low < high)
    {
        uint64_t middle = low + (high - low) / 2;
        if (code_point < ranges[middle][0])
        {
            high = middle;
        }
        else if (code_point > ranges[middle][1])
        {
            low = middle + 1;
        }
        else
        {
            return true;
        }
    }
    return false;
}

static bool is_identifier_start_code_point(uint32_t code_point)
{
    return code_point == NAMED_UCN || code_point_in(c_identifier_start_ranges, ARRAY_LENGTH(c_identifier_start_ranges), code_point);
}

static bool is_identifier_continue_code_point(uint32_t code_point)
{
    return code_point == NAMED_UCN || code_point_in(c_identifier_continue_ranges, ARRAY_LENGTH(c_identifier_continue_ranges), code_point);
}

// clang's getEscapedNewLineSize: the size of the whitespace and newline after a backslash, if they make it a line
// splice ("\r\n" and "\n\r" count as one newline), or 0.
static uint64_t escaped_newline_size(const ScalarLexer* lexer, uint64_t position)
{
    uint64_t size = 0;
    while (position + size < lexer->length && is_whitespace(lexer->source[position + size]))
    {
        size += 1;
        uint8_t c = lexer->source[position + size - 1];
        if (c == '\n' || c == '\r')
        {
            if (position + size < lexer->length)
            {
                uint8_t next = lexer->source[position + size];
                if ((next == '\n' || next == '\r') && next != c)
                {
                    size += 1;
                }
            }
            return size;
        }
    }
    return 0;
}

// clang's getCharAndSize: the character at `position` after trigraph replacement and line splicing, the number of
// bytes it takes up (`size`, including the line splices before it) and where the character itself starts.
static int32_t get_char(const ScalarLexer* lexer, uint64_t position, uint64_t* size, uint64_t* char_start)
{
    uint64_t p = position;
    int32_t result = END_OF_FILE;
    while (p < lexer->length)
    {
        int32_t c = lexer->source[p];
        uint64_t char_size = 1;
        if (lexer->trigraphs && c == '?' && p + 2 < lexer->length && lexer->source[p + 1] == '?')
        {
            int32_t trigraph = decode_trigraph(lexer->source[p + 2]);
            if (trigraph != 0)
            {
                c = trigraph;
                char_size = 3;
            }
        }
        if (c == '\\')
        {
            uint64_t newline_size = escaped_newline_size(lexer, p + char_size);
            if (newline_size != 0)
            {
                p += char_size + newline_size;
                continue;
            }
        }
        *char_start = p;
        result = c;
        p += char_size;
        break;
    }
    if (result == END_OF_FILE)
    {
        *char_start = lexer->length;
    }
    *size = p - position;
    return result;
}

static int32_t peek_char(const ScalarLexer* lexer, uint64_t position, uint64_t* size)
{
    uint64_t char_start;
    return get_char(lexer, position, size, &char_start);
}

// llvm::convertUTF8Sequence with strictConversion: the length of the UTF-8 sequence at `position`, or 0 if it isn't
// valid (overlong, a surrogate, past U+10FFFF or cut short by the end of the file).
static uint64_t decode_utf8(const ScalarLexer* lexer, uint64_t position, uint32_t* code_point)
{
    const uint8_t* s = lexer->source + position;
    uint64_t available = lexer->length - position;
    uint8_t lead = s[0];
    uint64_t length = 0;
    uint32_t value = 0;
    uint8_t second_min = 0x80;
    uint8_t second_max = 0xBF;
    if (lead >= 0xC2 && lead <= 0xDF)
    {
        length = 2;
        value = lead & 0x1F;
    }
    else if (lead >= 0xE0 && lead <= 0xEF)
    {
        length = 3;
        value = lead & 0x0F;
        second_min = lead == 0xE0 ? 0xA0 : 0x80;
        second_max = lead == 0xED ? 0x9F : 0xBF;
    }
    else if (lead >= 0xF0 && lead <= 0xF4)
    {
        length = 4;
        value = lead & 0x07;
        second_min = lead == 0xF0 ? 0x90 : 0x80;
        second_max = lead == 0xF4 ? 0x8F : 0xBF;
    }

    uint64_t result = 0;
    if (length != 0 && available >= length && s[1] >= second_min && s[1] <= second_max)
    {
        result = length;
        for (uint64_t i = 1; i < length; i += 1)
        {
            if ((s[i] & 0xC0) != 0x80)
            {
                result = 0;
            }
            value = (value << 6) | (s[i] & 0x3F);
        }
        *code_point = value;
    }
    return result;
}

// clang's tryReadUCN: reads the universal character name whose backslash `get_char` returns at `position`. Returns
// false if there is none. Otherwise `end` is past it and `code_point` is its code point, or 0 if C doesn't allow it
// outside of literals (clang still treats the whole UCN as one character then).
static bool read_ucn(const ScalarLexer* lexer, uint64_t position, uint64_t* end, uint32_t* code_point)
{
    uint64_t size;
    peek_char(lexer, position, &size);
    uint64_t p = position + size;
    int32_t kind = peek_char(lexer, p, &size);
    p += size;

    bool result = false;
    if (kind == 'u' || kind == 'U')
    {
        uint32_t digits_wanted = kind == 'u' ? 4 : 8;
        bool delimited = false;
        bool valid = true;
        uint32_t count = 0;
        uint32_t value = 0;
        while (count != digits_wanted || delimited)
        {
            int32_t c = peek_char(lexer, p, &size);
            if (!delimited && count == 0 && c == '{')
            {
                delimited = true;
                p += size;
                continue;
            }
            if (delimited && c == '}')
            {
                p += size;
                break;
            }
            int32_t digit = hex_digit_value(c);
            if (digit < 0)
            {
                valid = !delimited;
                break;
            }
            if (value & 0xF0000000u)
            {
                valid = false;
                break;
            }
            value = (value << 4) | (uint32_t)digit;
            p += size;
            count += 1;
        }
        if (valid && count != 0 && !(delimited && kind == 'U') && (delimited || count == digits_wanted))
        {
            result = true;
            *end = p;
            *code_point = (value < 0xA0 || (value >= 0xD800 && value <= 0xDFFF)) ? 0 : value;
        }
    }
    else if (kind == 'N' && peek_char(lexer, p, &size) == '{')
    {
        p += size;
        uint64_t name_length = 0;
        bool found_end = false;
        bool valid_name = true;
        for (;;)
        {
            int32_t c = peek_char(lexer, p, &size);
            if (c == END_OF_FILE)
            {
                break;
            }
            p += size;
            if (c == '}')
            {
                found_end = true;
                break;
            }
            if (c == '\n' || c == '\r')
            {
                break;
            }
            name_length += 1;
            if (c == 0)
            {
                break;
            }
            valid_name &= (c >= 'A' && c <= 'Z') || is_digit(c) || c == ' ' || c == '-';
        }
        if (found_end && name_length != 0)
        {
            result = true;
            *end = p;
            *code_point = valid_name ? NAMED_UCN : 0;
        }
    }
    return result;
}

// clang's tryConsumeIdentifierUCN: `get_char` returns a backslash at `position`.
static bool try_consume_identifier_ucn(const ScalarLexer* lexer, uint64_t position, uint64_t* end)
{
    uint64_t ucn_end;
    uint32_t code_point;
    bool result = read_ucn(lexer, position, &ucn_end, &code_point) && code_point != 0 && is_identifier_continue_code_point(code_point);
    if (result)
    {
        *end = ucn_end;
    }
    return result;
}

// clang's tryConsumeIdentifierUTF8Char: `get_char` returns a non-ASCII byte at `position`.
static bool try_consume_identifier_utf8(const ScalarLexer* lexer, uint64_t position, uint64_t* end)
{
    uint64_t size;
    uint64_t char_start;
    get_char(lexer, position, &size, &char_start);
    uint32_t code_point;
    uint64_t length = decode_utf8(lexer, char_start, &code_point);
    bool result = length != 0 && is_identifier_continue_code_point(code_point);
    if (result)
    {
        *end = char_start + length;
    }
    return result;
}

// clang's LexIdentifierContinue: returns the end of the identifier whose first character ends at `position`.
static uint64_t lex_identifier_continue(const ScalarLexer* lexer, uint64_t position)
{
    uint64_t p = position;
    for (;;)
    {
        uint64_t size;
        int32_t c = peek_char(lexer, p, &size);
        if (is_ascii_identifier_continue(c) || c == '$')
        {
            p += size;
        }
        else if (!((c == '\\' && try_consume_identifier_ucn(lexer, p, &p)) || (c >= 0x80 && try_consume_identifier_utf8(lexer, p, &p))))
        {
            break;
        }
    }
    return p;
}

// clang's LexNumericConstant: returns the end of the pp-number whose first character (and the digit after a leading
// '.') end at `position`.
static uint64_t lex_numeric_constant(const ScalarLexer* lexer, uint64_t position)
{
    uint64_t p = position;
    for (;;)
    {
        uint64_t size;
        int32_t c = peek_char(lexer, p, &size);
        int32_t previous = 0;
        while (is_ascii_identifier_continue(c) || c == '.' || c == '$')
        {
            p += size;
            previous = c;
            c = peek_char(lexer, p, &size);
        }

        if ((c == '-' || c == '+') && (previous == 'e' || previous == 'E' || previous == 'p' || previous == 'P'))
        {
            p += size;
            continue;
        }
        if (c == '\'')
        {
            uint64_t next_size;
            if (is_ascii_identifier_continue(peek_char(lexer, p + size, &next_size)))
            {
                p += size + next_size;
                continue;
            }
        }
        if ((c == '\\' && try_consume_identifier_ucn(lexer, p, &p)) || (c >= 0x80 && try_consume_identifier_utf8(lexer, p, &p)))
        {
            continue;
        }
        break;
    }
    return p;
}

// The suffixes of the standard library's string literal operators, which (unlike others) don't start with '_'.
static bool is_standard_suffix(const char* suffix, uint32_t length)
{
    static const char* const suffixes[] = { "d", "h", "i", "if", "il", "min", "ms", "ns", "s", "sv", "us", "y" };
    bool result = false;
    for (uint32_t i = 0; i < ARRAY_LENGTH(suffixes); i += 1)
    {
        result |= strlen(suffixes[i]) == length && memcmp(suffixes[i], suffix, length) == 0;
    }
    return result;
}

// clang's LexUDSuffix (C++): the end of the user-defined literal suffix after the literal that ends at `position`
// (`position` itself if it has none).
static uint64_t lex_ud_suffix(const ScalarLexer* lexer, uint64_t position, bool is_string)
{
    uint64_t p = position;
    uint64_t size;
    int32_t c = peek_char(lexer, p, &size);
    bool consumed = false;
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'))
    {
        if (!((c == '\\' && try_consume_identifier_ucn(lexer, p, &p)) || (c >= 0x80 && try_consume_identifier_utf8(lexer, p, &p))))
        {
            return position;
        }
        consumed = true;
    }
    if (!consumed)
    {
        // A suffix must start with '_', unless it's one of the standard library's (after a string literal).
        bool is_suffix = c == '_';
        if (!is_suffix && is_string)
        {
            char suffix[3] = { (char)c };
            uint32_t length = 1;
            uint64_t q = p + size;
            for (;;)
            {
                uint64_t next_size;
                int32_t next = peek_char(lexer, q, &next_size);
                if (!is_ascii_identifier_continue(next) && next != '$')
                {
                    is_suffix = is_standard_suffix(suffix, length);
                    break;
                }
                if (length == 3)
                {
                    break;
                }
                suffix[length] = (char)next;
                length += 1;
                q += next_size;
            }
        }
        if (!is_suffix)
        {
            return position;
        }
        p += size;
    }
    for (;;)
    {
        c = peek_char(lexer, p, &size);
        if (is_ascii_identifier_continue(c) || c == '$')
        {
            p += size;
        }
        else if (!((c == '\\' && try_consume_identifier_ucn(lexer, p, &p)) || (c >= 0x80 && try_consume_identifier_utf8(lexer, p, &p))))
        {
            break;
        }
    }
    return p;
}

// clang's LexRawStringLiteral (C++): returns the end of the raw string literal whose `R"` ends at `position`, and
// turns `kind` into `C_TOKEN_UNKNOWN` if its delimiter is invalid or it is unterminated. Inside of it, trigraphs and
// line splices are undone, so it is read from the source bytes.
static uint64_t lex_raw_string(const ScalarLexer* lexer, uint64_t position, uint8_t* kind)
{
    const uint8_t* s = lexer->source;
    const uint64_t n = lexer->length;
    // The delimiter: up to 16 printable characters other than space, '(', ')' and '\\', then '('.
    uint64_t delimiter_length = 0;
    while (delimiter_length != 16 && position + delimiter_length < n && s[position + delimiter_length] > ' ' && s[position + delimiter_length] < 0x7F &&
           s[position + delimiter_length] != '(' && s[position + delimiter_length] != ')' && s[position + delimiter_length] != '\\')
    {
        delimiter_length += 1;
    }
    uint64_t p = position;
    if (position + delimiter_length >= n || s[position + delimiter_length] != '(')
    {
        // An unknown token, up to the next '"'.
        while (p < n && s[p] != '"')
        {
            p += 1;
        }
        *kind = C_TOKEN_UNKNOWN;
        return p < n ? p + 1 : n;
    }
    p = position + delimiter_length + 1;
    for (;;)
    {
        if (p >= n)
        {
            *kind = C_TOKEN_UNKNOWN;
            return n;
        }
        const uint8_t c = s[p];
        p += 1;
        if (c == ')' && p + delimiter_length < n && memcmp(s + p, s + position, delimiter_length) == 0 && s[p + delimiter_length] == '"')
        {
            p += delimiter_length + 1;
            break;
        }
    }
    return lex_ud_suffix(lexer, p, true);
}

// clang's LexCharConstant and LexStringLiteral: returns the end of the literal whose opening quote ends at
// `position`, and turns `kind` into `C_TOKEN_UNKNOWN` if the literal is unterminated (or an empty character constant).
// In C++, a user-defined suffix is part of the literal.
static uint64_t lex_char_or_string(const ScalarLexer* lexer, uint64_t position, int32_t quote, uint8_t* kind)
{
    uint64_t p = position;
    uint64_t size;
    int32_t c = peek_char(lexer, p, &size);
    p += size;
    if (quote == '\'' && c == '\'')
    {
        *kind = C_TOKEN_UNKNOWN;
        return p;
    }
    while (c != quote)
    {
        if (c == '\\')
        {
            c = peek_char(lexer, p, &size);
            p += size;
        }
        if (c == '\n' || c == '\r')
        {
            *kind = C_TOKEN_UNKNOWN;
            return p - 1;
        }
        if (c == END_OF_FILE)
        {
            *kind = C_TOKEN_UNKNOWN;
            return lexer->length;
        }
        c = peek_char(lexer, p, &size);
        p += size;
    }
    return lexer->cplusplus ? lex_ud_suffix(lexer, p, quote == '"') : p;
}

// Skips the block comment whose "/*" ends at `position`. Returns false if it is unterminated.
static bool skip_block_comment(const ScalarLexer* lexer, uint64_t position, uint64_t* end)
{
    uint64_t p = position;
    int32_t previous = 0;
    for (;;)
    {
        uint64_t size;
        int32_t c = peek_char(lexer, p, &size);
        if (c == END_OF_FILE)
        {
            return false;
        }
        p += size;
        if (c == '/' && previous == '*')
        {
            *end = p;
            return true;
        }
        previous = c;
    }
}

// Skips the line comment whose "//" ends at `position`, up to (not including) the newline that ends it.
static uint64_t skip_line_comment(const ScalarLexer* lexer, uint64_t position)
{
    uint64_t p = position;
    for (;;)
    {
        uint64_t size;
        int32_t c = peek_char(lexer, p, &size);
        if (c == END_OF_FILE || c == '\n' || c == '\r')
        {
            return p;
        }
        p += size;
    }
}

static void emit(TokenOutput* output, uint8_t kind, uint64_t start, uint64_t end)
{
    output->kinds[output->count] = kind | (output->start_of_line ? C_TOKEN_START_OF_LINE : 0);
    output->starts[output->count] = (uint32_t)start;
    output->ends[output->count] = (uint32_t)end;
    output->count += 1;
    output->start_of_line = false;
}

// After a one-character punctuator `one` at `position`: if `get_char` returns `second` next, the token is `two`.
static uint8_t punctuator2(const ScalarLexer* lexer, uint64_t* position, int32_t second, uint8_t two, uint8_t one)
{
    uint64_t size;
    uint8_t result = one;
    if (peek_char(lexer, *position, &size) == second)
    {
        *position += size;
        result = two;
    }
    return result;
}

// Lexes from `position`, where a token may start (`start_of_line` if no token has started its logical line yet),
// writing the tokens to `kinds`, `starts` and `ends` from `count` on. Stops at the end of the source, or after the
// first newline at or past `stop_at` that is outside of tokens and comments: `resume` is then the position after it
// (else the length). Returns the new count. c_tokenize uses this for the parts its vectorized path can't do.
uint64_t c_tokenize_scalar_range(const uint8_t* source, uint64_t length, uint64_t position, bool start_of_line, uint64_t stop_at,
                                 uint8_t* kinds, uint32_t* starts, uint32_t* ends, uint64_t count, uint32_t flags, uint64_t* resume)
{
    ScalarLexer lexer_value = {
        .source = source,
        .length = length,
        .trigraphs = (flags & C_TOKENIZER_TRIGRAPHS) != 0,
        .cplusplus = (flags & C_TOKENIZER_CPLUSPLUS) != 0,
    };
    const ScalarLexer* lexer = &lexer_value;
    TokenOutput output = { .kinds = kinds, .starts = starts, .ends = ends, .count = count, .start_of_line = start_of_line };
    uint64_t p = position;
    *resume = length;

    for (;;)
    {
        uint64_t size;
        uint64_t start;
        int32_t c = get_char(lexer, p, &size, &start);
        if (c == END_OF_FILE)
        {
            break;
        }
        p += size;

        uint64_t size2;
        int32_t c2;
        uint8_t kind = C_TOKEN_UNKNOWN;
        switch (c)
        {
            case ' ': case '\t': case '\f': case '\v': case 0:
                continue;
            case '\n': case '\r':
                output.start_of_line = true;
                if (start >= stop_at)
                {
                    *resume = p;
                    return output.count;
                }
                continue;
            case '/':
                c2 = peek_char(lexer, p, &size2);
                if (c2 == '/')
                {
                    p = skip_line_comment(lexer, p + size2);
                    continue;
                }
                if (c2 == '*')
                {
                    if (skip_block_comment(lexer, p + size2, &p))
                    {
                        continue;
                    }
                    p = length;
                    kind = C_TOKEN_UNKNOWN;
                }
                else
                {
                    kind = punctuator2(lexer, &p, '=', C_TOKEN_SLASHEQUAL, C_TOKEN_SLASH);
                }
                break;
            case '"':
                kind = C_TOKEN_STRING_LITERAL;
                p = lex_char_or_string(lexer, p, '"', &kind);
                break;
            case '\'':
                kind = C_TOKEN_CHAR_CONSTANT;
                p = lex_char_or_string(lexer, p, '\'', &kind);
                break;
            case 'L': case 'u': case 'U': case 'R':
            {
                // Prefixed character constants and string literals: L, u8, u and U, in the order of the kinds. In
                // C++, R (after one of those or alone) starts a raw string literal.
                uint8_t prefix = c == 'L' ? 1 : c == 'u' ? 3 : c == 'U' ? 4 : 0;
                c2 = peek_char(lexer, p, &size2);
                uint64_t quote_end = p + size2;
                if (c == 'u' && c2 == '8')
                {
                    uint64_t size3;
                    int32_t c3 = peek_char(lexer, quote_end, &size3);
                    if (c3 == '"' || c3 == '\'' || (c3 == 'R' && lexer->cplusplus))
                    {
                        prefix = 2;
                        c2 = c3;
                        quote_end += size3;
                    }
                }
                bool raw = false;
                if (lexer->cplusplus && (c == 'R' || c2 == 'R'))
                {
                    uint64_t size3;
                    if (c == 'R')
                    {
                        raw = c2 == '"';
                    }
                    else if (peek_char(lexer, quote_end, &size3) == '"')
                    {
                        raw = true;
                        c2 = '"';
                        quote_end += size3;
                    }
                }
                if (raw)
                {
                    kind = (uint8_t)(C_TOKEN_STRING_LITERAL + prefix);
                    p = lex_raw_string(lexer, quote_end, &kind);
                }
                else if (c != 'R' && (c2 == '"' || c2 == '\''))
                {
                    kind = (uint8_t)((c2 == '"' ? C_TOKEN_STRING_LITERAL : C_TOKEN_CHAR_CONSTANT) + prefix);
                    p = lex_char_or_string(lexer, quote_end, c2, &kind);
                }
                else
                {
                    kind = C_TOKEN_RAW_IDENTIFIER;
                    p = lex_identifier_continue(lexer, p);
                }
                break;
            }
            case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                kind = C_TOKEN_NUMERIC_CONSTANT;
                p = lex_numeric_constant(lexer, p);
                break;
            case '.':
                c2 = peek_char(lexer, p, &size2);
                if (is_digit(c2))
                {
                    kind = C_TOKEN_NUMERIC_CONSTANT;
                    p = lex_numeric_constant(lexer, p + size2);
                }
                else if (c2 == '*' && lexer->cplusplus)
                {
                    kind = C_TOKEN_PERIODSTAR;
                    p += size2;
                }
                else
                {
                    uint64_t size3;
                    kind = C_TOKEN_PERIOD;
                    if (c2 == '.' && peek_char(lexer, p + size2, &size3) == '.')
                    {
                        kind = C_TOKEN_ELLIPSIS;
                        p += size2 + size3;
                    }
                }
                break;
            case '[': kind = C_TOKEN_L_SQUARE; break;
            case ']': kind = C_TOKEN_R_SQUARE; break;
            case '(': kind = C_TOKEN_L_PAREN; break;
            case ')': kind = C_TOKEN_R_PAREN; break;
            case '{': kind = C_TOKEN_L_BRACE; break;
            case '}': kind = C_TOKEN_R_BRACE; break;
            case '~': kind = C_TOKEN_TILDE; break;
            case '?': kind = C_TOKEN_QUESTION; break;
            case ';': kind = C_TOKEN_SEMI; break;
            case ',': kind = C_TOKEN_COMMA; break;
            case '-':
                c2 = peek_char(lexer, p, &size2);
                kind = C_TOKEN_MINUS;
                if (c2 == '-' || c2 == '>' || c2 == '=')
                {
                    kind = c2 == '-' ? C_TOKEN_MINUSMINUS : c2 == '>' ? C_TOKEN_ARROW : C_TOKEN_MINUSEQUAL;
                    p += size2;
                    uint64_t size3;
                    if (c2 == '>' && lexer->cplusplus && peek_char(lexer, p, &size3) == '*')
                    {
                        kind = C_TOKEN_ARROWSTAR;
                        p += size3;
                    }
                }
                break;
            case '+':
                c2 = peek_char(lexer, p, &size2);
                kind = C_TOKEN_PLUS;
                if (c2 == '+' || c2 == '=')
                {
                    kind = c2 == '+' ? C_TOKEN_PLUSPLUS : C_TOKEN_PLUSEQUAL;
                    p += size2;
                }
                break;
            case '&':
                c2 = peek_char(lexer, p, &size2);
                kind = C_TOKEN_AMP;
                if (c2 == '&' || c2 == '=')
                {
                    kind = c2 == '&' ? C_TOKEN_AMPAMP : C_TOKEN_AMPEQUAL;
                    p += size2;
                }
                break;
            case '|':
                c2 = peek_char(lexer, p, &size2);
                kind = C_TOKEN_PIPE;
                if (c2 == '|' || c2 == '=')
                {
                    kind = c2 == '|' ? C_TOKEN_PIPEPIPE : C_TOKEN_PIPEEQUAL;
                    p += size2;
                }
                break;
            case '*': kind = punctuator2(lexer, &p, '=', C_TOKEN_STAREQUAL, C_TOKEN_STAR); break;
            case '^': kind = punctuator2(lexer, &p, '=', C_TOKEN_CARETEQUAL, C_TOKEN_CARET); break;
            case '!': kind = punctuator2(lexer, &p, '=', C_TOKEN_EXCLAIMEQUAL, C_TOKEN_EXCLAIM); break;
            case '=': kind = punctuator2(lexer, &p, '=', C_TOKEN_EQUALEQUAL, C_TOKEN_EQUAL); break;
            case '#': kind = punctuator2(lexer, &p, '#', C_TOKEN_HASHHASH, C_TOKEN_HASH); break;
            case '<':
                c2 = peek_char(lexer, p, &size2);
                kind = C_TOKEN_LESS;
                if (c2 == '<')
                {
                    p += size2;
                    kind = punctuator2(lexer, &p, '=', C_TOKEN_LESSLESSEQUAL, C_TOKEN_LESSLESS);
                }
                else if (c2 == '=' || c2 == ':' || c2 == '%')
                {
                    uint64_t size3;
                    uint64_t size4;
                    const int32_t c3 = peek_char(lexer, p + size2, &size3);
                    // In C++, "<::" is '<' and "::" unless a ':' or '>' follows.
                    if (c2 == ':' && lexer->cplusplus && c3 == ':' && peek_char(lexer, p + size2 + size3, &size4) != ':' &&
                        peek_char(lexer, p + size2 + size3, &size4) != '>')
                    {
                        break;
                    }
                    kind = c2 == '=' ? C_TOKEN_LESSEQUAL : c2 == ':' ? C_TOKEN_L_SQUARE : C_TOKEN_L_BRACE;
                    p += size2;
                    if (c2 == '=' && lexer->cplusplus && c3 == '>')
                    {
                        kind = C_TOKEN_SPACESHIP;
                        p += size3;
                    }
                }
                break;
            case '>':
                c2 = peek_char(lexer, p, &size2);
                kind = C_TOKEN_GREATER;
                if (c2 == '>')
                {
                    p += size2;
                    kind = punctuator2(lexer, &p, '=', C_TOKEN_GREATERGREATEREQUAL, C_TOKEN_GREATERGREATER);
                }
                else if (c2 == '=')
                {
                    kind = C_TOKEN_GREATEREQUAL;
                    p += size2;
                }
                break;
            case '%':
                c2 = peek_char(lexer, p, &size2);
                kind = C_TOKEN_PERCENT;
                if (c2 == '=' || c2 == '>')
                {
                    kind = c2 == '=' ? C_TOKEN_PERCENTEQUAL : C_TOKEN_R_BRACE;
                    p += size2;
                }
                else if (c2 == ':')
                {
                    // "%:" is '#', and "%:%:" is "##".
                    uint64_t size3;
                    uint64_t size4;
                    p += size2;
                    kind = C_TOKEN_HASH;
                    if (peek_char(lexer, p, &size3) == '%' && peek_char(lexer, p + size3, &size4) == ':')
                    {
                        kind = C_TOKEN_HASHHASH;
                        p += size3 + size4;
                    }
                }
                break;
            case ':':
                c2 = peek_char(lexer, p, &size2);
                kind = C_TOKEN_COLON;
                if (c2 == '>' || c2 == ':')
                {
                    kind = c2 == '>' ? C_TOKEN_R_SQUARE : C_TOKEN_COLONCOLON;
                    p += size2;
                }
                break;
            case '\\':
            {
                // A UCN that starts an identifier, or else an unknown token (the whole UCN if there is one).
                uint64_t end;
                uint32_t code_point;
                kind = C_TOKEN_UNKNOWN;
                if (read_ucn(lexer, start, &end, &code_point))
                {
                    p = end;
                    if (code_point != 0 && is_identifier_start_code_point(code_point))
                    {
                        kind = C_TOKEN_RAW_IDENTIFIER;
                        p = lex_identifier_continue(lexer, p);
                    }
                }
                break;
            }
            default:
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '$')
                {
                    kind = C_TOKEN_RAW_IDENTIFIER;
                    p = lex_identifier_continue(lexer, p);
                }
                else if (c >= 0x80)
                {
                    // A UTF-8 identifier, or else an unknown token: the whole character if it is valid UTF-8.
                    uint32_t code_point;
                    uint64_t utf8_length = decode_utf8(lexer, start, &code_point);
                    kind = C_TOKEN_UNKNOWN;
                    if (utf8_length != 0)
                    {
                        p = start + utf8_length;
                        if (is_identifier_start_code_point(code_point))
                        {
                            kind = C_TOKEN_RAW_IDENTIFIER;
                            p = lex_identifier_continue(lexer, p);
                        }
                    }
                }
                break;
        }
        emit(&output, kind, start, p);
    }

    return output.count;
}

int64_t c_tokenize_scalar(const uint8_t* source, uint64_t length, uint8_t* kinds, uint32_t* starts, uint32_t* ends, uint32_t flags)
{
    if (length > UINT32_MAX - 128)
    {
        return C_TOKENIZER_ERROR_FILE_TOO_BIG;
    }
    const uint64_t begin = length >= 3 && source[0] == 0xEF && source[1] == 0xBB && source[2] == 0xBF ? 3 : 0;
    uint64_t resume;
    return (int64_t)c_tokenize_scalar_range(source, length, begin, true, UINT64_MAX, kinds, starts, ends, 0, flags, &resume);
}
