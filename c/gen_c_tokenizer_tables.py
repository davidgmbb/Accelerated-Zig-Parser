#!/usr/bin/env python3
"""Generates c/c_tokenizer_tables.h, the lookup tables of the vectorized C tokenizer (c/c_tokenizer.c).

Usage: python3 c/gen_c_tokenizer_tables.py   (run from the repository root)
"""
import re

with open("c/c_tokenizer.h") as f:
    kind_names = re.findall(r"^\s*X\((\w+), \w+\)", f.read(), re.M)
KIND = {name: i for i, name in enumerate(kind_names)}

# Property bits (`c_char_properties`), tested to make the per-chunk bitstrings.
PROPERTIES = {
    "IDENTIFIER": 0x01,  # [A-Za-z0-9_$]
    "DIGIT": 0x02,  # [0-9]
    "WHITESPACE": 0x04,  # ' ' \t \v \f \0 (not newlines)
    "NEWLINE": 0x08,  # \n \r
    "SEPARATOR_FOLLOWER": 0x10,  # [A-Za-z0-9_]: what may follow a digit separator
}

# Multi-character punctuators are found with "channels": each byte gets the set of channels in which it can be the
# first, second or third character, looked up by the low 4 bits of its operator class. A pair of adjacent bytes
# forms a 2-character punctuator if their first-of and second-of sets intersect. The SELF, SHIFT and DOTS channels
# only count if the two characters are the same.
CHANNELS = {
    "EQ": 0x01,  # X=, for X in + - * / % & | ^ < > = !
    "SELF": 0x02,  # ++ -- && || == ## ::
    "SHIFT": 0x04,  # << >>, which become <<= >>= with a third '='
    "DOTS": 0x08,  # .., which isn't a punctuator but becomes ... with a third '.'
    "GT": 0x10,  # -> :> %>
    "LTDIG": 0x20,  # <: <%
    "HASHDIG": 0x40,  # %:
    "EXPSIGN": 0x80,  # e+ e- E+ E- p+ p- P+ P-, a sign that continues a pp-number
}
# The operator class of each character: low 4 bits pick the channel sets, the high bits tell apart characters that
# share them (for the same-character channels). Class 0 is in no channel.
OPERATOR_CLASSES = {}
SLOTS = [
    # characters, first-of channels, second-of channels, third-of channels
    ("/^!", "EQ", "", ""),
    ("*", "EQ", "", ""),
    ("&|", "EQ SELF", "SELF", ""),
    ("+", "EQ SELF", "SELF EXPSIGN", ""),
    ("-", "EQ SELF GT", "SELF EXPSIGN", ""),
    ("%", "EQ GT HASHDIG", "LTDIG", ""),
    ("<", "EQ SHIFT LTDIG", "SHIFT", ""),
    (">", "EQ SHIFT", "SHIFT GT", ""),
    ("=", "EQ SELF", "EQ SELF", "SHIFT"),
    ("#", "SELF", "SELF", ""),
    (":", "SELF GT", "SELF LTDIG HASHDIG", ""),
    (".", "DOTS", "DOTS", "DOTS"),
    ("eEpP", "EXPSIGN", "", ""),
]
first, second, third = [0] * 16, [0] * 16, [0] * 16


def channel_set(names):
    return sum(CHANNELS[name] for name in names.split())


for slot, (characters, first_of, second_of, third_of) in enumerate(SLOTS, start=1):
    for i, c in enumerate(characters):
        OPERATOR_CLASSES[c] = slot | (i << 4)
    first[slot], second[slot], third[slot] = channel_set(first_of), channel_set(second_of), channel_set(third_of)

# C++ adds "<=>" and "->*": a '>' after a pair in the EQ channel and a '*' after one in the GT channel are triple
# candidates, which the tokenizer keeps if they start with '<' and '-'. (".*" is found from the '.' and '*' masks.)
third_cxx = list(third)
third_cxx[OPERATOR_CLASSES[">"] & 15] |= CHANNELS["EQ"]
third_cxx[OPERATOR_CLASSES["*"] & 15] |= CHANNELS["GT"]

ONE_CHARACTER = {
    "[": "L_SQUARE", "]": "R_SQUARE", "(": "L_PAREN", ")": "R_PAREN", "{": "L_BRACE", "}": "R_BRACE", ".": "PERIOD",
    "&": "AMP", "*": "STAR", "+": "PLUS", "-": "MINUS", "~": "TILDE", "!": "EXCLAIM", "/": "SLASH", "%": "PERCENT",
    "<": "LESS", ">": "GREATER", "^": "CARET", "|": "PIPE", "?": "QUESTION", ":": "COLON", ";": "SEMI", "=": "EQUAL",
    ",": "COMMA", "#": "HASH", '"': "STRING_LITERAL", "'": "CHAR_CONSTANT",
}
MULTI_CHARACTER = {
    "->": "ARROW", "++": "PLUSPLUS", "--": "MINUSMINUS", "<<": "LESSLESS", ">>": "GREATERGREATER", "<=": "LESSEQUAL",
    ">=": "GREATEREQUAL", "==": "EQUALEQUAL", "!=": "EXCLAIMEQUAL", "&&": "AMPAMP", "||": "PIPEPIPE",
    "*=": "STAREQUAL", "/=": "SLASHEQUAL", "%=": "PERCENTEQUAL", "+=": "PLUSEQUAL", "-=": "MINUSEQUAL",
    "&=": "AMPEQUAL", "^=": "CARETEQUAL", "|=": "PIPEEQUAL", "##": "HASHHASH", "::": "COLONCOLON", "<:": "L_SQUARE",
    ":>": "R_SQUARE", "<%": "L_BRACE", "%>": "R_BRACE", "%:": "HASH", "...": "ELLIPSIS", "<<=": "LESSLESSEQUAL",
    ">>=": "GREATERGREATEREQUAL", "%:%:": "HASHHASH",
}
MULTI_CHARACTER_CXX = {"<=>": "SPACESHIP", "->*": "ARROWSTAR", ".*": "PERIODSTAR"}


def is_pair(a, b):
    ca, cb = OPERATOR_CLASSES.get(a, 0), OPERATOR_CLASSES.get(b, 0)
    channels = first[ca & 15] & second[cb & 15]
    if ca != cb:
        channels &= ~(CHANNELS["SELF"] | CHANNELS["SHIFT"] | CHANNELS["DOTS"])
    return channels


# Check the channels against the punctuator list: every pair of printable characters.
OP2 = sum(CHANNELS[n] for n in ("EQ", "SELF", "SHIFT", "GT", "LTDIG", "HASHDIG"))
for a in map(chr, range(33, 127)):
    for b in map(chr, range(33, 127)):
        channels = is_pair(a, b)
        assert bool(channels & OP2) == (a + b in MULTI_CHARACTER), a + b
        assert bool(channels & CHANNELS["DOTS"]) == (a + b == ".."), a + b
        assert bool(channels & CHANNELS["EXPSIGN"]) == (a in "eEpP" and b in "+-"), a + b
        for c in map(chr, range(33, 127)):
            triple = third[OPERATOR_CLASSES.get(c, 0) & 15] & channels
            assert bool(triple) == (a + b + c in ("<<=", ">>=", "...")), a + b + c
            triple_cxx = third_cxx[OPERATOR_CLASSES.get(c, 0) & 15] & channels
            assert bool(triple_cxx & ~(CHANNELS["EQ"] | CHANNELS["GT"])) == bool(triple), a + b + c
            kept = bool(triple_cxx & (CHANNELS["EQ"] | CHANNELS["GT"])) and a == ("<" if c == ">" else "-")
            assert kept == (a + b + c in ("<=>", "->*")), a + b + c

# A punctuator's kind is looked up by a perfect hash of its first three characters (the third is 0 for 2-character
# punctuators, and "%:%" stands for "%:%:").
def hash_value(op):
    b = op.encode()[:3]
    return b[0] | b[1] << 8 | (b[2] << 16 if len(b) > 2 else 0)


def op_hash(op, multiplier):
    return ((hash_value(op) * multiplier) & 0xFFFFFFFF) >> 26


ALL_MULTI_CHARACTER = {**MULTI_CHARACTER, **MULTI_CHARACTER_CXX}
HASH_MULTIPLIER = 0x5DCA45B
random_state = 1
while len({op_hash(op, HASH_MULTIPLIER) for op in ALL_MULTI_CHARACTER}) != len(ALL_MULTI_CHARACTER):
    random_state = (random_state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    HASH_MULTIPLIER = (random_state >> 32) | 1
op_kinds = [KIND["UNKNOWN"]] * 64
for op, kind in ALL_MULTI_CHARACTER.items():
    op_kinds[op_hash(op, HASH_MULTIPLIER)] = KIND[kind]

properties, operator_classes, char_kinds = [0] * 128, [0] * 128, [KIND["UNKNOWN"]] * 128
for i in range(128):
    c = chr(i)
    if c.isalnum() or c in "_$":
        properties[i] |= PROPERTIES["IDENTIFIER"]
        char_kinds[i] = KIND["NUMERIC_CONSTANT" if c.isdigit() else "RAW_IDENTIFIER"]
    if c.isalnum() or c == "_":
        properties[i] |= PROPERTIES["SEPARATOR_FOLLOWER"]
    if c.isdigit():
        properties[i] |= PROPERTIES["DIGIT"]
    if c in " \t\v\f\0":
        properties[i] |= PROPERTIES["WHITESPACE"]
    if c in "\n\r":
        properties[i] |= PROPERTIES["NEWLINE"]
    if c in ONE_CHARACTER:
        char_kinds[i] = KIND[ONE_CHARACTER[c]]
    operator_classes[i] = OPERATOR_CLASSES.get(c, 0)


def table(name, values, per_line=16):
    lines = [f"static const uint8_t {name}[{len(values)}] __attribute__((aligned(64))) =", "{"]
    for i in range(0, len(values), per_line):
        lines.append("    " + " ".join(f"0x{v:02X}," for v in values[i:i + per_line]))
    lines.append("};")
    return "\n".join(lines)


with open("c/c_tokenizer_tables.h", "w") as out:
    out.write("// Generated by c/gen_c_tokenizer_tables.py: the lookup tables of the vectorized C tokenizer.\n#pragma once\n\n#include <stdint.h>\n\n")
    for name, value in PROPERTIES.items():
        out.write(f"#define CHAR_{name} 0x{value:02X}\n")
    out.write("\n")
    for name, value in CHANNELS.items():
        out.write(f"#define CHANNEL_{name} 0x{value:02X}\n")
    out.write(f"#define OP_HASH_MULTIPLIER 0x{HASH_MULTIPLIER:X}u\n\n")
    out.write("// Indexed by ASCII character.\n")
    out.write(table("c_char_properties", properties) + "\n" + table("c_operator_classes", operator_classes) + "\n" + table("c_char_kinds", char_kinds) + "\n\n")
    out.write("// Indexed by the low 4 bits of an operator class.\n")
    out.write(table("c_first_char_channels", first) + "\n" + table("c_second_char_channels", second) + "\n" + table("c_third_char_channels", third) + "\n")
    out.write("// In C++.\n" + table("c_third_char_channels_cxx", third_cxx) + "\n\n")
    out.write("// Indexed by the hash of a multi-character punctuator.\n" + table("c_op_hash_kinds", op_kinds) + "\n")
