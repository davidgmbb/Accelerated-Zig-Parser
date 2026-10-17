// Tests the vectorized C tokenizer (c_tokenizer.c) against the scalar lexer token by token, then benchmarks it.
//
// Usage: c_tokenizer_test [--fuzz N] [--seed N] [--no-bench] path...
// Paths are files or directories (searched for .c and .h files). The checks are: every file of the corpus, a snippet
// slid across the 64- and 512-byte boundaries, and N random inputs (default 200000). Run it pinned to one CPU
// (`taskset -c 2 ...`) for stable numbers. The scalar lexer itself is checked against clang by c/c_tokenizer_oracle.py.
#define _GNU_SOURCE
#include "c_tokenizer.h"

#include <dirent.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define PASSES (21)

typedef struct Source Source;
struct Source
{
    char* path;
    uint8_t* bytes;
    uint64_t length;
};

typedef struct Corpus Corpus;
struct Corpus
{
    Source* sources;
    uint64_t count;
    uint64_t capacity;
    uint64_t total_bytes;
    uint64_t max_length;
};

typedef struct Output Output;
struct Output
{
    uint8_t* kinds;
    uint32_t* starts;
    uint32_t* ends;
};

static bool has_c_extension(const char* name)
{
    uint64_t n = strlen(name);
    return n >= 2 && name[n - 2] == '.' && (name[n - 1] == 'c' || name[n - 1] == 'h');
}

static void add_file(Corpus* corpus, const char* path)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd == -1)
    {
        return;
    }
    struct stat file_stats;
    if (fstat(fd, &file_stats) == 0 && S_ISREG(file_stats.st_mode))
    {
        uint64_t length = (uint64_t)file_stats.st_size;
        uint8_t* bytes = aligned_alloc(64, (length + 64) & ~(uint64_t)63);
        uint64_t read_bytes = 0;
        while (read_bytes < length)
        {
            ssize_t n = read(fd, bytes + read_bytes, length - read_bytes);
            if (n <= 0)
            {
                break;
            }
            read_bytes += (uint64_t)n;
        }
        if (corpus->count == corpus->capacity)
        {
            corpus->capacity = corpus->capacity ? corpus->capacity * 2 : 1024;
            corpus->sources = realloc(corpus->sources, corpus->capacity * sizeof(Source));
        }
        corpus->sources[corpus->count] = (Source){ .path = strdup(path), .bytes = bytes, .length = read_bytes };
        corpus->count += 1;
        corpus->total_bytes += read_bytes;
        if (read_bytes > corpus->max_length)
        {
            corpus->max_length = read_bytes;
        }
    }
    close(fd);
}

static void add_path(Corpus* corpus, const char* path)
{
    struct stat path_stats;
    if (stat(path, &path_stats) != 0)
    {
        fprintf(stderr, "can't open %s\n", path);
        exit(1);
    }
    if (!S_ISDIR(path_stats.st_mode))
    {
        add_file(corpus, path);
        return;
    }
    DIR* directory = opendir(path);
    if (directory == 0)
    {
        return;
    }
    struct dirent* entry;
    while ((entry = readdir(directory)) != 0)
    {
        if (entry->d_name[0] == '.')
        {
            continue;
        }
        char child[4096];
        snprintf(child, sizeof(child), "%s/%s", path, entry->d_name);
        struct stat child_stats;
        if (lstat(child, &child_stats) != 0)
        {
            continue;
        }
        if (S_ISDIR(child_stats.st_mode))
        {
            add_path(corpus, child);
        }
        else if (S_ISREG(child_stats.st_mode) && has_c_extension(entry->d_name))
        {
            add_file(corpus, child);
        }
    }
    closedir(directory);
}

static Output output_alloc(uint64_t length)
{
    uint64_t capacity = C_TOKENIZER_CAPACITY(length);
    Output result = {
        .kinds = calloc(capacity, 1),
        .starts = calloc(capacity, sizeof(uint32_t)),
        .ends = calloc(capacity, sizeof(uint32_t)),
    };
    return result;
}

static void print_token(const uint8_t* source, uint64_t length, const Output* output, int64_t i, int64_t count)
{
    if (i < 0 || i >= count)
    {
        printf("-");
        return;
    }
    uint32_t start = output->starts[i];
    uint32_t end = output->ends[i];
    printf("%s%s [%u, %u) \"", c_token_kind_name(output->kinds[i]), output->kinds[i] & C_TOKEN_START_OF_LINE ? " SOL" : "", start, end);
    for (uint32_t p = start; p < end && p < start + 40 && p < length; p += 1)
    {
        uint8_t c = source[p];
        if (c >= 32 && c < 127 && c != '\\')
        {
            putchar(c);
        }
        else
        {
            printf("\\x%02X", c);
        }
    }
    printf("\"");
}

typedef enum Comparison
{
    COMPARISON_SAME,
    COMPARISON_DIFFERENT,
    COMPARISON_FELL_BACK,
} Comparison;

// Compares the vectorized path with the scalar lexer on `source`, printing the difference if `name` isn't null.
static Comparison compare(const uint8_t* source, uint64_t length, Output* vectorized, Output* scalar, const char* name)
{
    int64_t vectorized_count = c_tokenize_vectorized(source, length, vectorized->kinds, vectorized->starts, vectorized->ends, 0);
    if (vectorized_count == C_TOKENIZER_ERROR_NEEDS_SCALAR)
    {
        return COMPARISON_FELL_BACK;
    }
    int64_t scalar_count = c_tokenize_scalar(source, length, scalar->kinds, scalar->starts, scalar->ends, 0);
    int64_t i = 0;
    while (i < vectorized_count && i < scalar_count && vectorized->kinds[i] == scalar->kinds[i] && vectorized->starts[i] == scalar->starts[i] &&
           vectorized->ends[i] == scalar->ends[i])
    {
        i += 1;
    }
    if (i == vectorized_count && i == scalar_count)
    {
        return COMPARISON_SAME;
    }
    if (name != 0)
    {
        printf("%s: token %ld differs (vectorized has %ld tokens, scalar %ld)\n", name, (long)i, (long)vectorized_count, (long)scalar_count);
        for (int64_t j = i - 2; j <= i + 2; j += 1)
        {
            if (j < 0)
            {
                continue;
            }
            printf("  %s vectorized: ", j == i ? ">" : " ");
            print_token(source, length, vectorized, j, vectorized_count);
            printf("\n      scalar:     ");
            print_token(source, length, scalar, j, scalar_count);
            printf("\n");
        }
    }
    return COMPARISON_DIFFERENT;
}

// Tricky C that the vectorized path handles itself (no fallback): literals with escapes, prefixes and digit
// separators, comments of all kinds, line splices, digraphs, punctuator chains, unterminated literals and stray
// characters.
static const char boundary_snippet[] =
    "#include <stdio.h> // a \"comment\" with 'quotes' and /* in it\n"
    "  # define X(a, b) a ## b %: y %:%: z \\\n"
    "    (x) /* block\n"
    " * comment with \"quotes\" and 'apostrophes' don't\n"
    " */ int main(void) { return 0; } /**/ /* not closed by / or * / */ x /***/\n"
    "char *s = \"string with \\\" escaped \\\\ and // not a comment\", c = 'x', d = '\\'', e = '\\\\', f = '\"';\n"
    "wchar_t *w = L\"wide\"; char8_t *u = u8\"utf8\" u\"utf16\" U\"utf32\" L'w' u8'8' u'u' U'U' LR\"(no raw)\" Lx\"y\" uu\"z\";\n"
    "x = a+++b - -c --d ->e <<= f >>= g <= h >= i == j != k && l || m *= n /= o %= p += q -= r &= s ^= t |= u;\n"
    "a<::>b <% %> <: :> ... .. . .... a.b ...5 ..5 .5.5 1.2.3 x.e+5 1e+e+5 0x1p-3 1'000'000 1'a 1'e+5 0b1010 12.5DF 1$x;\n"
    "'' 'unterminated\n\"unterminated string\nok = \"multi \\\nline string\" 'a\\\nb' @ ` $x a$ ::: <=> ->* .* ## #\r\n"
    "s = \"a\\\\\\\\\" \"tab\tin\" /* \"unterminated\n*/ 'x' // line comment \\\ncontinued\nend;\r\n";

static uint64_t random_state = 0x9E3779B97F4A7C15ull;

static uint64_t random_next(void)
{
    random_state ^= random_state << 13;
    random_state ^= random_state >> 7;
    random_state ^= random_state << 17;
    return random_state;
}

// Pieces of C that random inputs are made of, weighted towards what makes the bitstrings interact.
static const char* const fuzz_pieces[] = {
    "a", "b", "L", "u", "u8", "U", "e", "E", "p", "x", "_", "$", "0", "1", "9", ".", "..", "...", "'", "'", "\"", "\"", "\"",
    "\\", "\\\\", "\\\"", "\\'", "\\\n", "\\\r\n", "/", "*", "/*", "*/", "//", "\n", "\n", "\r\n", "\r", " ", " ", "  ", "\t",
    "+", "-", "<", ">", "=", "!", "&", "|", "^", "%", ":", "#", "?", "(", ")", "[", "]", "{", "}", ";", ",", "~", "@", "`",
    "e+", "1e+", "0x", "'1'", "'a'", "%:", "<:", ":>", "<%", "%>", "->", "++", "<<=", ">>=", "##", "::", "\v", "\f",
};

// Pieces with a backslash, which often need the scalar lexer outside of literals: they're only used inside of literals
// in half of the random inputs.
static bool has_backslash(const char* piece)
{
    return strchr(piece, '\\') != 0;
}

static uint64_t fuzz_input(uint8_t* buffer, uint64_t capacity)
{
    uint64_t pieces = 1 + random_next() % 300;
    uint64_t length = 0;
    bool careful = random_next() % 2 == 0;
    bool in_literal = false;
    for (uint64_t i = 0; i < pieces; i += 1)
    {
        const char* piece = fuzz_pieces[random_next() % (sizeof(fuzz_pieces) / sizeof(fuzz_pieces[0]))];
        if (careful)
        {
            if (strcmp(piece, "\"") == 0)
            {
                in_literal = !in_literal;
            }
            else if (strchr(piece, '\n') != 0 || strchr(piece, '\r') != 0)
            {
                in_literal = false;
            }
            if (has_backslash(piece) && !in_literal)
            {
                continue;
            }
        }
        uint64_t n = strlen(piece);
        if (length + n > capacity)
        {
            break;
        }
        memcpy(buffer + length, piece, n);
        length += n;
        // Long runs of plain code between the interesting bits, to move them across the chunk boundaries.
        if (random_next() % 16 == 0)
        {
            uint64_t run = random_next() % 200;
            for (uint64_t j = 0; j < run && length < capacity; j += 1)
            {
                buffer[length] = "abc xyz 123 (); "[random_next() % 16];
                length += 1;
            }
        }
    }
    // Close a block comment that may be open: an unterminated one needs the scalar lexer.
    if (careful && length + 3 <= capacity)
    {
        memcpy(buffer + length, "\n*/", 3);
        length += 3;
    }
    return length;
}

typedef struct PerfCounters PerfCounters;
struct PerfCounters
{
    int cycles_fd;
    int instructions_fd;
};

static int open_counter(uint64_t config)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.type = PERF_TYPE_HARDWARE;
    attr.config = config;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    return (int)syscall(SYS_perf_event_open, &attr, 0, -1, -1, PERF_FLAG_FD_CLOEXEC);
}

static uint64_t read_counter(int fd)
{
    uint64_t value = 0;
    if (fd >= 0 && read(fd, &value, sizeof(value)) != sizeof(value))
    {
        value = 0;
    }
    return value;
}

typedef struct Sample Sample;
struct Sample
{
    uint64_t nanoseconds;
    uint64_t cycles;
    uint64_t instructions;
};

static Sample sample_now(const PerfCounters* counters)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    Sample result = {
        .nanoseconds = (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec,
        .cycles = read_counter(counters->cycles_fd),
        .instructions = read_counter(counters->instructions_fd),
    };
    return result;
}

static int compare_u64(const void* a, const void* b)
{
    uint64_t x = *(const uint64_t*)a;
    uint64_t y = *(const uint64_t*)b;
    return x < y ? -1 : x > y;
}

typedef int64_t (*TokenizeFunction)(const uint8_t*, uint64_t, uint8_t*, uint32_t*, uint32_t*, uint32_t);

int main(int argc, char** argv)
{
    Corpus corpus = {0};
    uint64_t fuzz_count = 200000;
    bool bench = true;
    for (int i = 1; i < argc; i += 1)
    {
        if (strcmp(argv[i], "--fuzz") == 0 && i + 1 < argc)
        {
            fuzz_count = strtoull(argv[i + 1], 0, 10);
            i += 1;
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            random_state = strtoull(argv[i + 1], 0, 10) * 0x9E3779B97F4A7C15ull + 1;
            i += 1;
        }
        else if (strcmp(argv[i], "--no-bench") == 0)
        {
            bench = false;
        }
        else
        {
            add_path(&corpus, argv[i]);
        }
    }
    printf("%lu files, %lu bytes\n", (unsigned long)corpus.count, (unsigned long)corpus.total_bytes);

    uint64_t max_length = corpus.max_length > 8192 ? corpus.max_length : 8192;
    Output vectorized = output_alloc(max_length);
    Output scalar = output_alloc(max_length);

    // 1. Every file of the corpus.
    uint64_t different = 0;
    uint64_t fell_back = 0;
    uint64_t fell_back_bytes = 0;
    for (uint64_t i = 0; i < corpus.count; i += 1)
    {
        Comparison comparison = compare(corpus.sources[i].bytes, corpus.sources[i].length, &vectorized, &scalar, different < 10 ? corpus.sources[i].path : 0);
        different += comparison == COMPARISON_DIFFERENT;
        if (comparison == COMPARISON_FELL_BACK)
        {
            fell_back += 1;
            fell_back_bytes += corpus.sources[i].length;
        }
    }
    printf("corpus: %lu files differ, %lu files (%.2f%% of bytes) need the scalar lexer\n", (unsigned long)different, (unsigned long)fell_back,
           corpus.total_bytes ? 100.0 * (double)fell_back_bytes / (double)corpus.total_bytes : 0.0);

    // 2. The snippet at every alignment, after 0 to 1099 bytes of padding.
    {
        uint64_t snippet_different = 0;
        uint64_t snippet_fell_back = 0;
        uint64_t snippet_length = sizeof(boundary_snippet) - 1;
        uint8_t* buffer = aligned_alloc(64, 4096);
        char name[128];
        for (uint64_t misalignment = 0; misalignment < 64; misalignment += 1)
        {
            for (uint64_t padding = 0; padding < 1100; padding += 1)
            {
                memset(buffer, 0, 4096);
                for (uint64_t p = 0; p < padding; p += 1)
                {
                    buffer[misalignment + p] = p % 61 == 60 ? '\n' : ' ';
                }
                memcpy(buffer + misalignment + padding, boundary_snippet, snippet_length);
                snprintf(name, sizeof(name), "snippet (misalignment %lu, padding %lu)", (unsigned long)misalignment, (unsigned long)padding);
                Comparison comparison = compare(buffer + misalignment, padding + snippet_length, &vectorized, &scalar, snippet_different < 3 ? name : 0);
                snippet_different += comparison == COMPARISON_DIFFERENT;
                snippet_fell_back += comparison == COMPARISON_FELL_BACK;
            }
        }
        printf("snippet: %lu of %u placements differ, %lu need the scalar lexer\n", (unsigned long)snippet_different, 64 * 1100,
               (unsigned long)snippet_fell_back);
        free(buffer);
        different += snippet_different;
    }

    // 3. Random inputs.
    {
        uint64_t fuzz_different = 0;
        uint64_t fuzz_fell_back = 0;
        uint8_t* buffer = aligned_alloc(64, 8192 + 64);
        char name[64];
        for (uint64_t i = 0; i < fuzz_count; i += 1)
        {
            uint64_t misalignment = random_next() % 64;
            uint64_t length = fuzz_input(buffer + misalignment, 8000);
            snprintf(name, sizeof(name), "random input %lu", (unsigned long)i);
            Comparison comparison = compare(buffer + misalignment, length, &vectorized, &scalar, fuzz_different < 3 ? name : 0);
            if (comparison == COMPARISON_DIFFERENT && fuzz_different < 3)
            {
                char path[64];
                snprintf(path, sizeof(path), "fuzz_failure_%lu.c", (unsigned long)fuzz_different);
                FILE* f = fopen(path, "wb");
                if (f != 0)
                {
                    fwrite(buffer + misalignment, 1, length, f);
                    fclose(f);
                    printf("  (saved as %s)\n", path);
                }
            }
            fuzz_different += comparison == COMPARISON_DIFFERENT;
            fuzz_fell_back += comparison == COMPARISON_FELL_BACK;
        }
        printf("random: %lu of %lu inputs differ, %lu need the scalar lexer\n", (unsigned long)fuzz_different, (unsigned long)fuzz_count,
               (unsigned long)fuzz_fell_back);
        free(buffer);
        different += fuzz_different;
    }

    // 4. Throughput, with the implementations taking turns. The first pass is a warm-up.
    if (bench && corpus.total_bytes != 0)
    {
        PerfCounters counters = { .cycles_fd = open_counter(PERF_COUNT_HW_CPU_CYCLES), .instructions_fd = open_counter(PERF_COUNT_HW_INSTRUCTIONS) };
        const char* const names[2] = { "c_tokenize", "c_tokenize_scalar" };
        TokenizeFunction functions[2] = { c_tokenize, c_tokenize_scalar };
        Sample samples[2][PASSES];
        uint64_t tokens = 0;
        for (uint64_t pass = 0; pass <= PASSES; pass += 1)
        {
            for (uint64_t which = 0; which < 2; which += 1)
            {
                Output* output = which == 0 ? &vectorized : &scalar;
                Sample before = sample_now(&counters);
                uint64_t pass_tokens = 0;
                for (uint64_t i = 0; i < corpus.count; i += 1)
                {
                    pass_tokens += (uint64_t)functions[which](corpus.sources[i].bytes, corpus.sources[i].length, output->kinds, output->starts, output->ends, 0);
                }
                Sample after = sample_now(&counters);
                tokens = pass_tokens;
                if (pass > 0)
                {
                    samples[which][pass - 1] = (Sample){
                        .nanoseconds = after.nanoseconds - before.nanoseconds,
                        .cycles = after.cycles - before.cycles,
                        .instructions = after.instructions - before.instructions,
                    };
                }
            }
        }
        printf("%d interleaved passes over %lu tokens, medians (cycles/B also min-max):\n", PASSES, (unsigned long)tokens);
        for (uint64_t which = 0; which < 2; which += 1)
        {
            uint64_t ns[PASSES];
            uint64_t cycles[PASSES];
            uint64_t instructions[PASSES];
            for (uint64_t p = 0; p < PASSES; p += 1)
            {
                ns[p] = samples[which][p].nanoseconds;
                cycles[p] = samples[which][p].cycles;
                instructions[p] = samples[which][p].instructions;
            }
            qsort(ns, PASSES, sizeof(uint64_t), compare_u64);
            qsort(cycles, PASSES, sizeof(uint64_t), compare_u64);
            qsort(instructions, PASSES, sizeof(uint64_t), compare_u64);
            double bytes = (double)corpus.total_bytes;
            double median_cycles = (double)cycles[PASSES / 2];
            printf("  %-18s %8.3f ms  %6.2f GB/s  %6.3f cycles/B [%.3f-%.3f]  %6.3f instructions/B  IPC %.2f\n", names[which],
                   (double)ns[PASSES / 2] / 1e6, bytes / (double)ns[PASSES / 2], median_cycles / bytes, (double)cycles[0] / bytes,
                   (double)cycles[PASSES - 1] / bytes, (double)instructions[PASSES / 2] / bytes, (double)instructions[PASSES / 2] / median_cycles);
        }
    }

    return different == 0 ? 0 : 1;
}
