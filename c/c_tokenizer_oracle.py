#!/usr/bin/env python3
"""Checks the C tokenizer token by token against clang's raw lexer (`clang -cc1 -std=c23 -dump-raw-tokens`).

Usage: python3 c/c_tokenizer_oracle.py [--impl simd|scalar] [--clang clang] [-j N] [--show N] [--edge-cases]
                                      [--cplusplus] [--trigraphs] [path...]
Paths can be files or directories (searched for .c and .h files, or with --cplusplus for C++ sources and headers,
including extensionless ones). --cplusplus checks C++ against `clang -x c++ -std=c++26`. --edge-cases adds the
tricky inputs below, and --trigraphs checks the trigraph option against `clang -std=c17` on inputs with trigraphs.
Build zig-out/c_tokenizer/libc_tokenizer.so first with c/c_tokenizer_build.sh.

Clang's dump is normalized to what the tokenizer emits: whitespace and comments are dropped (passing a start-of-line flag on
to the next token), and a token starts after any line splices.
"""
import argparse
import ctypes
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile

LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zig-out", "c_tokenizer", "libc_tokenizer.so")
START_OF_LINE = 0x80
WHITESPACE = frozenset(b" \t\n\r\v\f\0")

library = None
options = None

# Inputs that exercise the corners of C's lexical grammar (and of the vectorized path).
EDGE_CASES = [
    # Stray characters, literals and their prefixes.
    b"a\x00b ''' c 'a'b' d\n1$x 1.5e+ 0x1.p+5 x. \xc3\xa9x \xf0\x9f\x98\x80 y \xff z \xe2\x80\x8b w\n\"s\\\r\nt\" q\\\r\nr //c\\\rd\nk\n",
    b"x /* unterminated\nmore",
    b"y \\   ",
    b"\"abc",
    b"@@ `` \x01\x01 \\\\ $x a$ '\\' \"\\\"\nL'x' u8\"x\" u8'x' U'x' u'x' LR\"x\" a...b .. . ->* .* <=> ::: a::b <::> <% %> %:%: %:% ## # ?? ??= ??/\n"
    b" x\n/\\\n/ c\n'\\\n' \"a\\\\\n\" q\n'ab 1'2' 0x1p-3 1.e+5 .5e-3f 1'000'000 0b1010 1ULL 12.5DF\n",
    b"#define X 1\n  # define Y\n/* c */ #z\n/*\n*/ #w\nx # y\n\\\n#v\n",
    b"L'' u8'' U\"\" u8\"x L'\nLx\"y\" uu\"z\" u8u'x' 8\"x\" Lu'x'\n'' 'x 'xy' '\\'' '\\\\' \"\\\"\" \"a\\\nb\" '\n'\n",
    # UCNs and UTF-8.
    b"\\u{e9}x \\N{LATIN SMALL LETTER E WITH ACUTE}y \\u0041 \\u0024 \\u00a8 \\ud800 \\u0300x a\\u0300 \\U00010000 1\\u00e9 x\\u00e9y \\u00e9\n"
    b"L'' u8'' '\\\n' \"\\\n\" x/*/ y */ z a*/*b*/c\n\xcc\x80x a\xcc\x80 \xef\xbb\xbf \xc2\xa0 \xe2\x80\x8bq 1\xc3\xa9 \xc3 x\n#include <a'b> \"\n",
    b"x\\u0041 x\\ud800 x\\u00a8 x\\u12 \\u12 x\\u{} x\\u{110000} x\\u{e9 x\\U0010FFFF x\\U00110000\n"
    b"1'$ 1'\\u00e9 1'e+5 1e'+5 1'_ 0x1'p+2 1\\u0041 x$ $1 1$\n"
    b"\xe2\x88\x82x x\xe2\x82\x81 \xe2\x88\x9e \xc2\xb7x x\xc2\xb7 \xe3\x80\x80 \xef\xbc\xa1\n"
    b"\\ x\\\t\ny \\\v\fz \\\n\rw \\\r\n\rv\nx = 1 /\\\n/ c\nk /\\\n* c */ m\n",
    b"#include <CppCoreCheck\\Warnings.h>\n\\scratch3, \\in_2 \\\\x \\\"y\" \\'z' \\Ux \\u12345678 \\N{GREEK SMALL LETTER ALPHA}x\n",
    # Line splices everywhere.
    b"ab\\\ncd 12\\\n34 +\\\n+ -\\\n> /\\\n* x *\\\n/ \"st\\\nr\" 'c\\\nd' // c\\\nd\n#define M(a) \\\n  a \\\n  + 1\n",
    b"x\\\r\ny\\\rz\\\n\rw\\  \nv \\\t\nu\n1e+\\\n5 .\\\n5 1.\\\n'2 /**\\\n\\\n/ a\n'1'\\\n1 \"\\\r\r\n::\n",
    # Comments.
    b"/**/ /***/ /*/ */ /* * / */ //\n/* a\n * b \" ' // \n */ x /* \"*/ y\n// a /* b\nc */ d\n",
    b"/*\n/**/;*/ a /* x */*/ b\n/* it's\n don't */ 'a' \"/*\" '/*' x\n/*\n\"*/ /* \"\n*/ e\n",
    # pp-numbers and punctuators.
    b"1.2.3 x.e+5 1e+e+5 .5.5 ..5 1'a 1'_ x1'2' ...5 ....5 .....5 a+++b a--->b <<= >>= <<<= %:%:%: <::> :>>\n",
    # A byte order mark, CR line endings, no newline at the end.
    b"\xef\xbb\xbfint x;\rint y;\r\n#if 1\r#endif",
]

# C++ inputs (checked with --cplusplus --edge-cases, with the C ones).
EDGE_CASES_CPP = [
    # User-defined literal suffixes.
    b"\"abc\"sv \"abc\"s \"abc\"if \"abc\"_x 'a'_x 'a'x \"\"_x \"a\"PRId64 \"a\"min \"a\"d \"a\"y u8\"x\"_s \"a\"_ \"a\"__x L\"a\"s "
    b"\"a\"il \"a\"ms \"a\"sss \"a\"i \"a\"h \"x\"_1 \"x\"1 ''_x \"x\"_a$b \"x\"s1 \"x\"sv_ 'x'sv \"x\"\\\n_y \"x\"\\u00e9 1.0_km 10ms\n",
    # Raw string literals.
    b"R\"(a)\" R\"d(a)d\" R\"d(a)x)d\" LR\"(a)\" u8R\"(a)\" uR\"(a)\" UR\"(a)\" xR\"(a)\" R\"(x)\"_s R\"(x)\"s R\"a b(x)a b\" z\n"
    b"R\"(a\\\nb)\" R\"(multi\nline // not a comment /* nor this\n\"raw)\" R\"abcdefghijklmnopq(x)abcdefghijklmnopq\" y\n"
    b"R\"$@`(x)$@`\" R\"'(x)'\" R\"x\"(z)x\"\" R\"*/(c)*/\" /* R\"( */ // R\"(\nw R\n\"(x)\" u8R \"y\" LR\"\n(x)\"\n",
    b"R\"(unterminated\nraw",
    # Punctuators.
    b"a <=> b a->*b a.*b 1.*2 ...* <::x <::: <::> a<::b <:: c ^^ :: .** ->** <=>= <==> <=>> ==> :>* %>* .*. x.*=\n"
    b"std::vector<::std::string> v; template<::T> <:\\\n:x <%::\n",
]

# Inputs with trigraphs (checked with --trigraphs), without C23's lexical additions.
TRIGRAPH_CASES = [
    b"??=define X(a) a ??=??= b\n??=if 1\nint a??(3??) = ??< 1, 2 ??>; x = a ??! b ??- c ??' d;\n??=endif\n"
    b"\"str??/\"ing\" ??/\nfoo ???= ??x\n// line ??/\ncontinued\nint y; /* *??/\n/ z\n",
]


def escaped_newline_size(data, position):
    """clang's getEscapedNewLineSize."""
    size = 0
    while position + size < len(data) and data[position + size] in b" \t\f\v\n\r":
        size += 1
        c = data[position + size - 1]
        if c in b"\n\r":
            if position + size < len(data) and data[position + size] in b"\n\r" and data[position + size] != c:
                size += 1
            return size
    return 0


def skip_splices(data, position, end):
    """Skips the line splices at `position` (with trigraphs, "??/" is a backslash)."""
    while position < end:
        if data[position] == 0x5C:
            backslash = 1
        elif options.trigraphs and data[position:position + 3] == b"??/":
            backslash = 3
        else:
            break
        size = escaped_newline_size(data, position + backslash)
        if size == 0:
            break
        position += backslash + size
    return position


def line_offsets(data):
    """The offset of each line, counted like clang's SourceManager ("\\r\\n" is one line break)."""
    offsets = [0]
    i = 0
    n = len(data)
    while i < n:
        c = data[i]
        if c == 0x0A or c == 0x0D:
            if c == 0x0D and i + 1 < n and data[i + 1] == 0x0A:
                i += 1
            offsets.append(i + 1)
        i += 1
    return offsets


def clang_tokens(path, data):
    """Clang's tokens for `path` as (kind, start, end, start_of_line), normalized, or None if clang fails."""
    language = ["-x", "c++", "-std=c++26"] if options.cplusplus else ["-std=c17" if options.trigraphs else "-std=c23"]
    result = subprocess.run([options.clang, "-cc1", *language, "-dump-raw-tokens", path], capture_output=True)
    record = re.compile(rb"^([a-z0-9_]+) +'(.*?)' +Loc=<" + re.escape(path.encode()) + rb":(\d+):(\d+)>(.*)$")
    offsets = line_offsets(data)
    raw = []
    for line in result.stderr.split(b"\n"):
        match = record.match(line)
        if match:
            start = offsets[int(match.group(3)) - 1] + int(match.group(4)) - 1
            raw.append((match.group(1).decode(), start, b"[StartOfLine]" in match.group(5)))
    if result.returncode != 0 and not raw:
        return None

    tokens = []
    pending_start_of_line = False
    for i, (kind, start, start_of_line) in enumerate(raw):
        end = raw[i + 1][1] if i + 1 < len(raw) else len(data)
        if kind == "comment":
            pending_start_of_line |= start_of_line
            continue
        clean_start = skip_splices(data, start, end)
        if kind == "unknown" and all(c in WHITESPACE for c in data[clean_start:end]):
            pending_start_of_line |= start_of_line
            continue
        tokens.append((kind, clean_start, end, start_of_line or pending_start_of_line))
        pending_start_of_line = False
    return tokens


def our_tokens(data):
    capacity = len(data) + 128
    kinds = (ctypes.c_uint8 * capacity)()
    starts = (ctypes.c_uint32 * capacity)()
    ends = (ctypes.c_uint32 * capacity)()
    tokenize = library.c_tokenize if options.impl == "simd" else library.c_tokenize_scalar
    count = tokenize(data, len(data), kinds, starts, ends, (1 if options.trigraphs else 0) | (2 if options.cplusplus else 0))
    if count < 0:
        return None
    name = library.c_token_kind_name
    return [(name(kinds[i]).decode(), starts[i], ends[i], bool(kinds[i] & START_OF_LINE)) for i in range(count)]


def initialize(parsed_options):
    global library, options
    options = parsed_options
    library = ctypes.CDLL(options.library)
    signature = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    for function in (library.c_tokenize, library.c_tokenize_scalar):
        function.argtypes = signature
        function.restype = ctypes.c_int64
    library.c_token_kind_name.argtypes = [ctypes.c_uint8]
    library.c_token_kind_name.restype = ctypes.c_char_p


def describe(data, token):
    kind, start, end, start_of_line = token
    text = data[start:end][:60]
    return f"{kind} [{start}, {end}){' SOL' if start_of_line else ''} {text!r}"


def check(path):
    with open(path, "rb") as f:
        data = f.read()
    expected = clang_tokens(path, data)
    if expected is None:
        return path, "clang failed", 0, None
    actual = our_tokens(data)
    if actual is None:
        return path, "tokenizer failed", 0, None
    if actual == expected:
        return path, "ok", len(expected), None
    i = 0
    while i < min(len(actual), len(expected)) and actual[i] == expected[i]:
        i += 1
    lines = [f"  first difference at token {i} of {len(expected)} (we have {len(actual)}):"]
    for j in range(max(0, i - 2), min(i + 3, max(len(actual), len(expected)))):
        ours = describe(data, actual[j]) if j < len(actual) else "-"
        theirs = describe(data, expected[j]) if j < len(expected) else "-"
        lines.append(f"  {'>' if j == i else ' '} ours:  {ours}\n      clang: {theirs}")
    return path, "mismatch", len(expected), "\n".join(lines)


C_EXTENSIONS = (".c", ".h")
CPP_EXTENSIONS = (".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++", ".h", ".inc", ".def", ".ipp", ".tcc", ".tpp")


def is_source(path, cplusplus):
    """Is `path` a C (or C++) source file? In C++, a file without an extension counts if it is text (like <vector>)."""
    if os.path.islink(path) or not os.path.isfile(path):
        return False
    name = os.path.basename(path)
    if not cplusplus:
        return name.endswith(C_EXTENSIONS)
    if name.endswith(CPP_EXTENSIONS):
        return True
    if "." in name:
        return False
    with open(path, "rb") as f:
        head = f.read(4096)
    return len(head) > 0 and b"\0" not in head


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--impl", choices=("simd", "scalar"), default="simd")
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--library", default=LIBRARY)
    parser.add_argument("-j", type=int, default=os.cpu_count())
    parser.add_argument("--show", type=int, default=10, help="mismatching files to print")
    parser.add_argument("--edge-cases", action="store_true")
    parser.add_argument("--trigraphs", action="store_true")
    parser.add_argument("--cplusplus", action="store_true")
    parsed = parser.parse_args()

    files = []
    for path in parsed.paths:
        if os.path.isdir(path):
            for directory, _, names in os.walk(path):
                files.extend(os.path.join(directory, name) for name in sorted(names) if is_source(os.path.join(directory, name), parsed.cplusplus))
        else:
            files.append(path)
    if parsed.edge_cases or parsed.trigraphs:
        directory = tempfile.mkdtemp(prefix="c_tokenizer_cases_")
        cases = TRIGRAPH_CASES if parsed.trigraphs else EDGE_CASES + (EDGE_CASES_CPP if parsed.cplusplus else [])
        for i, case in enumerate(cases):
            files.append(os.path.join(directory, f"case_{i}.c"))
            with open(files[-1], "wb") as f:
                f.write(case)

    counts = {"ok": 0, "mismatch": 0, "clang failed": 0, "tokenizer failed": 0}
    tokens = 0
    shown = 0
    with multiprocessing.Pool(parsed.j, initializer=initialize, initargs=(parsed,)) as pool:
        for path, status, token_count, details in pool.imap_unordered(check, files, chunksize=16):
            counts[status] += 1
            tokens += token_count
            if status != "ok" and shown < parsed.show:
                shown += 1
                print(f"{status}: {path}")
                if details:
                    print(details)
    print(f"{len(files)} files, {tokens} tokens: " + ", ".join(f"{count} {status}" for status, count in counts.items()))
    sys.exit(0 if counts["mismatch"] == 0 and counts["tokenizer failed"] == 0 else 1)


if __name__ == "__main__":
    main()
