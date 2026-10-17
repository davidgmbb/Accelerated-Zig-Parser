#!/bin/sh
# Builds the C tokenizer's test and benchmark program, and the shared library that c/c_tokenizer_oracle.py loads,
# into zig-out/c_tokenizer/. Run it from the repository root. MARCH picks the target CPU (default: native).
set -eu
MARCH=${MARCH:-native}
OUT=zig-out/c_tokenizer
mkdir -p "$OUT"
FLAGS="-std=c11 -march=$MARCH -O3 -g -fwrapv -funsigned-char -fno-strict-aliasing -Wall -Wextra -Werror"
SOURCES="c/c_tokenizer.c c/c_tokenizer_scalar.c"
clang $FLAGS -fPIC -shared $SOURCES -o "$OUT/libc_tokenizer.so"
clang $FLAGS $SOURCES c/c_tokenizer_test.c -o "$OUT/c_tokenizer_test"
