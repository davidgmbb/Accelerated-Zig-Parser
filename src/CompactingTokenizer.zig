// The 512-byte-chunk compacting tokenizer by Niles Salter (@Validark), presented in
// "Accelerating the Zig Tokenizer (with AVX-512)" at Software You Can Love Vancouver 2026.
//
// Extracted from the source embedded in the talk's slides (`<g id="compacting-tokenizer">` in
// https://github.com/Validark/presentations/blob/cc335a81f66756d3cd4c2035ab866bf894d370ad/sycl.html).
// Copyright (c) 2024 Niles Salter, MIT License (same terms as this repository's LICENSE).
//
// The slides target a newer Zig than this repository. Every change made to build with Zig 0.14.1,
// or to fill in pieces the slides left out, is marked with a `PORT:` comment.
const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

pub const Token = struct {
    kind: Tag,
    start: u32,
    end: u32,
};

pub const TokenList = std.MultiArrayList(Token);
pub const empty = @This(){ .token_list = .empty };

token_list: TokenList,

pub fn deinit(self: *@This(), gpa: Allocator) void {
    self.token_list.deinit(gpa);
}

/// PORT: The slides' `tokenizeAssumeCapacityWithOptions` returns errors from a `void` function, so its signature now says so.
pub const Error = error{ InvalidCarriageReturn, InvalidUtf8 };

pub fn tokenize(
    self: *@This(),
    gpa: Allocator,
    source: [:0]const u8,
) (Error || error{ FileTooBig, OutOfMemory })!void {
    try self.ensureTokenCapacity(gpa, source.len);
    try self.tokenizeAssumeCapacity(source);
}

pub fn tokenizeAndShrink(
    self: *@This(),
    gpa: Allocator,
    source: [:0]const u8,
) (Error || error{ FileTooBig, OutOfMemory })!void {
    try self.tokenize(gpa, source);
    self.shrinkAndFree(gpa);
}

pub fn ensureTokenCapacity(self: *@This(), gpa: Allocator, source_len_raw: usize) error{ FileTooBig, OutOfMemory }!void {
    const source_len = std.math.cast(u32, source_len_raw) orelse return error.FileTooBig;
    const over_allocated_token_buf_cap = std.math.add(u32, source_len, 64) catch return error.FileTooBig;
    try self.token_list.ensureTotalCapacity(gpa, over_allocated_token_buf_cap);
}

pub fn shrinkAndFree(self: *@This(), gpa: Allocator) void {
    self.token_list.shrinkAndFree(gpa, self.token_list.len);
}

pub fn tokenizeAssumeCapacity(self: *@This(), source: [:0]const u8) Error!void {
    return self.tokenizeAssumeCapacityWithOptions(source, .{});
}

pub const Tag = enum(u8) {
    null = 0,
    @"*" = 1,
    @"=" = 2,
    @"+" = 3,
    @"<" = 4,
    @"?" = 5,
    _unused_multi_char_symbol_participant = 6,
    @">" = 7,
    @"|" = 8,
    @"^" = 9,
    @"-" = 10,
    @"&" = 11,
    @"!" = 12,
    @"/" = 13,
    @"%" = 14,
    @"." = 15,

    @">>=" = 16,
    @"-%=" = 19,
    @"//" = 20,
    @"!=" = 23,
    @"||" = 24,
    @"-|" = 25,
    @"**" = 27,
    @"<<|" = 28,
    @"*^" = 29,
    @"=>" = 32,
    @".**" = 33,
    @"-%" = 37,
    @"/=" = 40,
    @".." = 44,
    @">>" = 52,
    @"*|=" = 54,
    @"<=" = 55,
    @".?" = 59,
    @"..." = 62,
    @"<<=" = 63,
    @"*=" = 64,
    @"*%=" = 66,
    @"+|=" = 75,
    @"==" = 76,
    @"//!" = 78,
    @"->" = 82,
    @"+=" = 85,
    @"+%=" = 87,
    @"%=" = 88,
    @"*|" = 90,
    @".*" = 93,
    @"///" = 95,
    @"<<|=" = 96,
    @">=" = 97,
    @"<<" = 99,
    @"*%" = 103,
    @"|=" = 108,
    @"&=" = 109,
    @"+|" = 111,
    @"++" = 114,
    @"-|=" = 116,
    @"^=" = 123,
    @"+%" = 124,
    @"-=" = 127,
    @"(" = 128,
    @")" = 129,
    @"extern" = 130,
    @"," = 131,
    @":" = 132,
    @"while" = 133,
    @";" = 134,
    @"[" = 135,
    @"]" = 136,
    @"align" = 137,
    @"else" = 138,
    @"{" = 139,
    @"or" = 140,
    @"break" = 141,
    @"}" = 142,
    @"anytype" = 143,
    @"~" = 144,
    @"volatile" = 147,
    string_identifier = 148,
    builtin = 149,
    @"errdefer" = 155,
    double_backslash = 159,
    unknown = 160,
    @"catch" = 161,
    whitespace = 162,
    @"test" = 163,
    @"struct" = 164,
    @"const" = 165,
    @"return" = 168,
    @"resume" = 171,
    string = 172,
    @"opaque" = 173,
    @"packed" = 177,
    @"orelse" = 178,
    char_literal = 179,
    invalid = 180,
    @"var" = 185,
    identifier = 187,
    @"unreachable" = 194,
    @"threadlocal" = 195,
    @"anyframe" = 196,
    number = 197,
    @"noinline" = 199,
    @"defer" = 200,
    @"try" = 203,
    @"pub" = 204,
    @"fn" = 208,
    @"and" = 209,
    @"allowzero" = 210,
    @"error" = 211,
    @"addrspace" = 214,
    @"if" = 215,
    @"nosuspend" = 217,
    @"linksection" = 221,
    @"inline" = 222,
    @"export" = 224,
    @"asm" = 225,
    @"noalias" = 226,
    @"suspend" = 227,
    @"switch" = 233,
    @"union" = 234,
    @"enum" = 245,
    @"continue" = 247,
    @"for" = 249,
    @"callconv" = 250,
    @"comptime" = 252,
};

const Keywords = struct {
    // We can halve the number of cache lines by using two-byte slices into a dense superstring:
    const superstring: []const u8 =
        "addrspacerrdeferrorelsenumswitchunreachablepackedforeturnunionwhilecontinueconstructcomptimevolatileifnpubreaktestrycatch" ++ "linksectionosuspendanytypeanyframeandallowzeropaquexporthreadlocallconvaresumexternoinlinealignoaliasm";

    const kw_tags = [_]Tag{
        .@"addrspace", .@"align", .@"allowzero",   .@"and",    .@"anyframe",    .@"anytype",  .@"asm",       .@"break",  .@"callconv", .@"catch",
        .@"comptime",  .@"const", .@"continue",    .@"defer",  .@"else",        .@"enum",     .@"errdefer",  .@"error",  .@"export",   .@"extern",
        .@"fn",        .@"for",   .@"if",          .@"inline", .@"noalias",     .@"noinline", .@"nosuspend", .@"opaque", .@"or",       .@"orelse",
        .@"packed",    .@"pub",   .@"resume",      .@"return", .@"linksection", .@"struct",   .@"suspend",   .@"switch", .@"test",     .@"threadlocal",
        .@"try",       .@"union", .@"unreachable", .@"var",    .@"volatile",    .@"while",
    };

    const super_string_starts = blk: {
        @setEvalBranchQuota(100000); // PORT: Zig 0.14's comptime `indexOf` needs more
        var starts: @Vector(128, u8) = @splat(0);
        for (kw_tags) |tag| {
            const name = @tagName(tag);
            const start_offset = std.mem.indexOf(u8, superstring, name) orelse
                @compileError(std.fmt.comptimePrint("{} should be present in `unpadded_kws`", .{name}));

            // PORT: `hash` returns a vector, and only its low 7 bits are used as an index (like `perm2b` does at runtime).
            const i: u7 = @truncate(hash(
                1,
                @splat(@intCast(name.len)),
                @bitCast(@as([2]u8, name[0..2].*)),
                @bitCast(@as([2]u8, name[name.len - 2 ..][0..2].*)), // PORT: drop the sentinel before @bitCast
            )[0]);

            starts[i] = std.math.cast(u8, start_offset) orelse
                @compileError(std.fmt.comptimePrint("index {} doesn't fit in a u8, we're gonna need a bigger indexing integer", .{}));
        }
        break :blk starts;
    };

    const super_string_lens = blk: {
        var lens: @Vector(128, u8) = @splat(0);
        for (kw_tags) |tag| {
            const name = @tagName(tag);
            // PORT: `hash` returns a vector, and only its low 7 bits are used as an index (like `perm2b` does at runtime).
            const i: u7 = @truncate(hash(
                1,
                @splat(@intCast(name.len)),
                @bitCast(@as([2]u8, name[0..2].*)),
                @bitCast(@as([2]u8, name[name.len - 2 ..][0..2].*)), // PORT: drop the sentinel before @bitCast
            )[0]);
            lens[i] = name.len;
        }
        break :blk lens;
    };

    pub fn hash(
        comptime N: comptime_int,
        identifier_lens: @Vector(N, u16),
        first_two_chars: @Vector(N, u16),
        last_two_chars: @Vector(N, u16),
    ) @Vector(2 * N, u8) {
        return @bitCast(
            (((identifier_lens << @splat(14)) ^ first_two_chars) *% last_two_chars) >> @splat(8),
        );
    }
};

const CharClasses = enum(u8) {
    null = 0,
    @"*" = 1,
    @"=" = 2,
    @"+" = 3,
    @"<" = 4,
    @"?" = 5,
    _unused_multi_char_symbol_participant = 6,
    @">" = 7,
    @"|" = 8,
    @"^" = 9,
    @"-" = 10,
    @"&" = 11,
    @"!" = 12,
    @"/" = 13,
    @"%" = 14,
    @"." = 15,

    @"(" = 128,
    @")" = 129,
    @"," = 131,
    @":" = 132,
    @";" = 134,
    @"[" = 135,
    @"]" = 136,
    @"{" = 139,
    @"}" = 142,
    @"~" = 144,

    string_identifier = 148,
    at = 149,
    backslash = 159,
    unknown = 160,
    whitespace = 162,
    quote = 172,
    apostrophe = 179,
    bad_control = 180,
    alpha_underscore = 187,
    digit = 197,

    pub const FIRST_NON_SYMBOL = CharClasses.string_identifier;
};

// This lookup table:
//  1. Is used to map all characters than can be in multi-character operators to the range [1,15].
//     Once the relevant characters are in the range [1, 15], it can be fed into a `vpshufb` to
//     determine where the multi-char operator are. Everything not mapped to the range [1, 15] is
//     mapped to the range [0x80, 0xFF] instead, that way, when passed into `vpshufb`, the result will be zeroed.
//  2. Simplifies the checks we have to do to produce bitstrings such that only a single instruction
//     is necessary to produce any bitstring that contains some interesting set of characters.
//     We can take advantage of signed and unsigned comparisons by making our interesting clusters
//     in the range [0x80, 0x80+c] and/or [0xFF-k, 0xFF] for some value of `c` or `k`
//  3. Is used as the base of the `kinds` vector, which holds the uncompressed token `kind`s. Things
//     like '"' and '\'' will be prefilled with the kind for a quote or a character-literal.
//     That way, we don't have to place those kinds manually when constructing the vector of `kind`s.
const classifier: @Vector(128, u8) = blk: {
    var v: [128]CharClasses = undefined;
    for (0..128) |c|
        v[c] = switch (@as(u7, c)) {
            0...'\t' - 1, '\n' + 1...'\r' - 1, '\r' + 1...' ' - 1, 0x7F => .bad_control,
            '\n', '\r', '\t', ' ' => .whitespace,
            '0'...'9' => .digit,
            'a'...'z', 'A'...'Z', '_' => .alpha_underscore,
            '\'' => .apostrophe,
            '"' => .quote,
            '\\' => .backslash,
            '@' => .at,
            '#', '$', '`' => .unknown,

            '!' => .@"!",
            '%' => .@"%",
            '&' => .@"&",
            '(' => .@"(",
            ')' => .@")",
            '*' => .@"*",
            '+' => .@"+",
            ',' => .@",",
            '-' => .@"-",
            '.' => .@".",
            '/' => .@"/",
            ':' => .@":",
            ';' => .@";",
            '<' => .@"<",
            '=' => .@"=",
            '>' => .@">",
            '?' => .@"?",
            '[' => .@"[",
            ']' => .@"]",
            '^' => .@"^",
            '{' => .@"{",
            '|' => .@"|",
            '}' => .@"}",
            '~' => .@"~",
        };
    break :blk @bitCast(v);
};

fn comptimeChecker() void {
    comptime {
        @setEvalBranchQuota(10000);
        for (0..16) |i| {
            const a = @tagName(@as(CharClasses, @enumFromInt(i)));
            const b = @tagName(@as(Tag, @enumFromInt(i)));
            if (!std.mem.eql(u8, a, b))
                @compileError(std.fmt.comptimePrint("Discrepancy: CharClasses.{s} and Tag.{s} are both {}. Either they should have the same name or there is a mismatch.", .{ a, b, i }));
        }

        for (@typeInfo(CharClasses).@"enum".fields) |field| {
            const name = switch (@field(std.meta.FieldEnum(CharClasses), field.name)) {
                .backslash => "double_backslash",
                .alpha_underscore => "identifier",
                .digit => "number",
                .bad_control => "invalid",
                .quote => "string",
                .apostrophe => "char_literal",
                .at => "builtin",
                else => field.name,
            };

            if (!@hasField(Tag, name))
                @compileError(std.fmt.comptimePrint(
                    "Tag should have member {s} with value {}",
                    .{ name, field.value },
                ));

            const tag_value = @intFromEnum(@field(Tag, name));
            if (tag_value != field.value)
                @compileError(std.fmt.comptimePrint(
                    "Tag.{s} is {}, but it should be {} if you want to match CharClasses.{s}",
                    .{ name, tag_value, field.value, field.name },
                ));
        }
    }
}

pub fn tokenizeAssumeCapacityWithOptions(
    self: *@This(),
    source: [:0]const u8,

    // This is configurable fortesting purposes.
    // Just use `fill`.
    comptime options: struct {
        V: type = @Vector(64, u8),
        uint: type = @Vector(8, u64),
        VLEN: comptime_int = 512,
        NUM_MASK_PRODUCING_ITERS: comptime_int = 8,
        USE_ASM: bool = true,
    },
) Error!void {
    comptimeChecker();

    const andn = simd.andn;
    const V = options.V;
    const uint = options.uint;
    const VLEN = options.VLEN;
    const NUM_MASK_PRODUCING_ITERS = options.NUM_MASK_PRODUCING_ITERS;
    // PORT: Only the default configuration is implemented. `USE_ASM` only applied to the removed `Helper.Carry` (see below).
    comptime assert(V == @Vector(64, u8) and uint == @Vector(8, u64) and VLEN == 512 and NUM_MASK_PRODUCING_ITERS == 8);

    const MaskNames = [_][]const u8{
        "carriages",
        "slashes",
        "backslashes",
        "newlines_or_eof",
        "quotes",
        "apostrophes",
        "bad_controls",
        "ats",
        "whitespaces",
        "alpha_underscores",
        "alpha_numeric_underscores",
        "possible_one_char_symbols",
        "prelim_double_char_ends",
        "prelim_triple_char_ends",
    };

    // PORT: The slides were in the middle of moving the loop-carried state into vector registers (`Helper.Carry`, which
    // also produced the `prev_*` mask names). That code didn't agree with its call sites (most of the fields they
    // used didn't exist), so the previous chunk's bitstrings are carried in this plain struct instead. When only
    // the last bit of a bitstring matters, `topBit` moves it to bit 0.
    const zero: uint = @splat(0);
    var carried: struct {
        carriages: uint = zero,
        slashes: uint = zero,
        backslashes: uint = zero,
        ats: uint = zero,
        next_is_escaped: uint = zero,
        alpha_numeric_underscores: uint = zero,
        identifier_or_number_or_builtin_ends: uint = zero,
        double_char: uint = zero,
        delete_triple_char_pos: uint = zero,
        local_word_ends: uint = zero,
        all_starts: uint = zero,
        // Which construct the previous chunk ended inside of, at bit 0.
        inside_quote_string: uint = zero,
        inside_char_literal: uint = zero,
        inside_comment: uint = zero,
        inside_line_string: uint = zero,
    } = .{};

    const SubMasks = ExternStruct(&MaskNames, @Vector(64, bool)); // PORT: was `@Struct(...)`
    var o = std.mem.zeroes(ExternStruct(&MaskNames, uint)); // PORT: was `@Struct(...)`

    // PORT: Bytes past the end of the file read as newlines (see `safe_load` below), so a 64-byte chunk that lies
    // entirely past the end must look like newlines too. The slides zeroed these, which let an unterminated
    // comment or string at the end of a file run to the end of the 512-byte chunk.
    var eof_sub_masks = std.mem.zeroes(SubMasks);
    eof_sub_masks.newlines_or_eof = @splat(true);
    eof_sub_masks.whitespaces = @splat(true);

    self.token_list.len = self.token_list.capacity;
    errdefer self.token_list.len = 0; // PORT
    const slice = self.token_list.slice();

    // PORT: The slides mixed slices with pointer arithmetic here. Many-item pointers make the rewinding below simple.
    const kinds_base: [*]u8 = @ptrCast(slice.items(.kind).ptr);
    var cur_kind = kinds_base;
    var cur_start = slice.items(.start).ptr;
    var cur_end = slice.items(.end).ptr;

    var maybe_aligned_cur_in = source;
    if (std.mem.startsWith(u8, maybe_aligned_cur_in, "\xEF\xBB\xBF"))
        maybe_aligned_cur_in = maybe_aligned_cur_in[3..];

    const misalignment: u6 = @truncate(@intFromPtr(maybe_aligned_cur_in.ptr));
    const load_mask = (if (maybe_aligned_cur_in.len > 64) ~@as(u64, 0) else simd.bzhi(~@as(u64, 0), maybe_aligned_cur_in.len)) << misalignment;
    var cur_in = @as([*:0]align(64) const u8, @ptrFromInt(@intFromPtr(maybe_aligned_cur_in.ptr) & ~@as(u64, 63)))[0 .. maybe_aligned_cur_in.len + misalignment];

    // PORT: Padded with '\n' instead of ' ', so that bytes past the end of the file terminate comments and strings,
    // like the padding of the last chunk does. Bytes before the start of the file are whitespace either way.
    var chunk: V = simd.masked_load(cur_in, load_mask, '\n');
    var prev_chunk_for_pt3: V = @splat(0);

    var prev_classified: V = @splat(0);
    var prev_double_char_ends: V = @splat(0);

    // PORT: The span of 64-byte chunks that contain non-ASCII bytes, for the UTF-8 validation after the loop.
    var non_ascii_start: usize = std.math.maxInt(usize);
    var non_ascii_end: usize = 0;

    // This loop has 3 phases.
    //   I. Vectorized Classification: Read input, then produce bitstrings that tell us where different character classes exist in a chunk.
    //  II. Analysis: Do bit-manipulation to figure out where all tokens start and end
    // III. Write-out: Use the information produced in Phase II to produce the output `starts` and/or `ends` and/or `lens` vectors.
    // MAYBE:     Phase III is also where we handle classification of operators and keywords.
    while (true) {
        var sub_masks = (std.MultiArrayList(SubMasks){ .bytes = @ptrCast(&o), .len = NUM_MASK_PRODUCING_ITERS, .capacity = NUM_MASK_PRODUCING_ITERS }).slice();

        // PORT: The position of this 512-byte chunk in `source` (wrapping, because the first chunk starts before
        // `source.ptr` when the source is misaligned).
        const chunk_pos: u32 = @truncate(@intFromPtr(cur_in.ptr) -% @intFromPtr(source.ptr));
        var reached_eof = false;

        var i: u32 = 0;
        var chunks: [NUM_MASK_PRODUCING_ITERS]V = undefined;
        var classified_chunks: [NUM_MASK_PRODUCING_ITERS]V = undefined;

        while (true) {
            chunks[i] = chunk;
            const non_ASCII = chunk >= @as(V, @splat(0x80));

            // See the documentation on `classifier`
            var classified = simd.perm2b(classifier, chunk);

            if (@as(u64, @bitCast(non_ASCII)) != 0) {
                @branchHint(.unlikely);

                // `classified` has undefined bytes corresponding to the non-ascii bytes in `chunk`. Let's fix that.
                classified = @select(
                    u8,
                    non_ASCII,
                    @as(V, @splat(@intFromEnum(CharClasses.unknown))),
                    classified,
                );

                // PORT: was `utf8_validator.chunk(chunk);`, which the slides never defined.
                const pos = @intFromPtr(cur_in.ptr) -| @intFromPtr(source.ptr);
                non_ascii_start = @min(non_ascii_start, pos);
                non_ascii_end = pos + 64;
            }

            classified_chunks[i] = classified;
            const prev1_classified = shiftInPrevious(1, classified, prev_classified);
            const prev2_classified = shiftInPrevious(2, classified, prev_classified);

            const Channels = packed struct {
                bar_mod: u1 = 1,
                eql: u1 = 1,
                dot: u1 = 1,
                arrow: u1 = 1,
                @"<": u1 = 1,
                @">": u1 = 1,
                dot_question: u1 = 1,
                self: u1 = 1, // TODO: move this field to whichever slot allows us to reuse a vector
            };

            const char_data = comptime char_data: {
                var first_char_data: @Vector(16, u8) = @splat(0);
                var second_char_data: @Vector(16, u8) = @splat(0);
                var third_char_data: @Vector(16, u8) = @splat(0);

                for (std.meta.fields(Channels)) |f| {
                    // Note: this assumes that the prefix string of all operators defined here are also an operator.
                    // If you want a new operator that has a prefix string which is not a valid operator,
                    // you need to either check it yourself without using the vector shuffle,
                    // or just write a line that augments the logic done by the vector shuffle.
                    // The augmentation technique would look similar to what we do below for the "self" matcher.

                    for (@as([3][]const u8, switch (@field(std.meta.FieldEnum(Channels), f.name)) {
                        // zig fmt: off
                        .bar_mod      => .{ "-+*",             "%|",            "="  }, // Matches [-+*][%|]=?
                        .eql          => .{ "|=!%^+*/&<>-",    "=",             ""   }, // Matches [|=!%^+*/&<>-]=
                        .self         => .{ "|.+*",          "|.+*",            "."  }, // Matches ([|.+*])(\1)|...
                        .dot          => .{ ".",               "*",             "*"  }, // Matches .**|.*
                        .arrow        => .{ "-=",              ">",             ""   }, // Matches [-=]>
                        .@"<"         => .{ "<",               "<",             "|=" }, // Matches <<[|=]
                        .@">"         => .{ ">",               ">",             "="  }, // Matches >>=
                        .dot_question => .{ ".",               "?",             ""   }, // Matches .?
                        // zig fmt: on
                    }), .{ &first_char_data, &second_char_data, &third_char_data }) |string, dest_char_data| {
                        var state = std.mem.zeroes(Channels);
                        @field(state, f.name) = 1;
                        for (string) |c|
                            dest_char_data[@intFromEnum(@field(CharClasses, &[1]u8{c}))] |= @bitCast(state); // PORT: `@intFromEnum`
                    }
                }

                for (0..128) |k| {
                    if (classifier[k] == 0)
                        @compileError(
                            "If you are just mapping a multi-char-symbol to 0:\n" ++
                                "Make sure that `Tag.eof` is non-0, because `kinds` starts out as `classified`, meaning that your symbol will become a Tag backed by 0. Alternatively, one could manually updoot those 0's in `kinds` to something else, or apply an XOR to `classified` before using it as `kinds`. If that's done, you should be good to delete this.\n\n" ++
                                "If you are mapping 0 to 0, you have to update the `single_char_ends` declaration after this line of code.",
                        );
                }

                // PORT: The slides mutated `comptime var`s declared inside this runtime loop, which Zig 0.14 rejects.
                break :char_data .{ first_char_data, second_char_data, third_char_data };
            };
            const first_char_data, const second_char_data, const third_char_data = char_data;

            const double_char_ends = simd.vpshufb(second_char_data, classified) &
                simd.vpshufb(first_char_data, prev1_classified) &
                @select(
                    u8,
                    prev1_classified == classified,
                    @as(V, @splat(@bitCast(Channels{ .self = 1 }))),
                    @as(V, @splat(@bitCast(Channels{ .self = 0 }))),
                );

            const triple_char_ends = simd.vpshufb(third_char_data, classified) &
                shiftInPrevious(1, double_char_ends, prev_double_char_ends) &
                @select(
                    u8,
                    prev2_classified == prev1_classified,
                    @as(V, @splat(@bitCast(Channels{ .self = 1 }))),
                    @as(V, @splat(@bitCast(Channels{ .self = 0 }))),
                );

            // PORT: `@intFromEnum` on the `CharClasses` below.
            sub_masks.set(i, .{
                // zig fmt: off
                .carriages                 = chunk == @as(V, @splat('\r')),
                .newlines_or_eof           = chunk == @as(V, @splat('\n')),
                .slashes                   = chunk == @as(V, @splat('/')),
                .backslashes               = chunk == @as(V, @splat('\\')),
                .quotes                    = chunk == @as(V, @splat('"')),
                .apostrophes               = chunk == @as(V, @splat('\'')),
                .ats                       = chunk == @as(V, @splat('@')),
                .bad_controls              = classified == @as(V, @splat(@intFromEnum(CharClasses.bad_control))),
                .whitespaces               = classified == @as(V, @splat(@intFromEnum(CharClasses.whitespace))),
                .alpha_underscores         = classified == @as(V, @splat(@intFromEnum(CharClasses.alpha_underscore))),
                .alpha_numeric_underscores = classified >= @as(V, @splat(@intFromEnum(CharClasses.alpha_underscore))),
                .possible_one_char_symbols = classified < @as(V, @splat(@intFromEnum(CharClasses.FIRST_NON_SYMBOL))),
                .prelim_double_char_ends   = double_char_ends != @as(V, @splat(0)),
                .prelim_triple_char_ends   = triple_char_ends != @as(V, @splat(0)),
                // zig fmt: on
            });

            comptime for (@typeInfo(CharClasses).@"enum".fields) |field|
                if (@field(std.meta.FieldEnum(CharClasses), field.name) != .digit and field.value > @intFromEnum(CharClasses.alpha_underscore))
                    @compileError("Currently `mapped_char >= CharClasses.alpha_underscore` is used to compute alpha_numeric_underscore. " ++
                        "Either adjust the enum or the comparison used above.");

            prev_classified = classified;
            prev_double_char_ends = double_char_ends;

            i += 1;

            // PORT: The slides only advanced `cur_in` between the 64-byte chunks of one 512-byte chunk, so each 512-byte chunk
            // started by re-reading the previous chunk's last 64 bytes, and the outer loop had no exit (it ended in an
            // unconditional `break` after the first 512 bytes). We advance after every 64-byte chunk and note when the
            // file ended.
            if (cur_in.len <= 64) {
                reached_eof = true;
                while (i < NUM_MASK_PRODUCING_ITERS) : (i += 1) {
                    chunks[i] = @splat('\n');
                    classified_chunks[i] = @splat(@intFromEnum(CharClasses.whitespace));
                    sub_masks.set(i, eof_sub_masks);
                }
                break;
            }

            cur_in = cur_in[64..];
            chunk = if (cur_in.len < 64) simd.safe_load(64, cur_in, '\n') else cur_in[0..64].*;
            if (i >= NUM_MASK_PRODUCING_ITERS) break;
        }

        // Matches `(?<=\r)([^\n]|$)`
        const invalid_carriage_followers = ~o.newlines_or_eof & shld(o.carriages, 1, carried.carriages);
        // PORT: The slides computed this but left it unused. Like the 64-byte tokenizer (and the check after this loop),
        // a '\r' must be followed by '\n'.
        if (!isZero(invalid_carriage_followers)) return error.InvalidCarriageReturn;

        // Matches `(?<=(^|[^\\])(\\\\)*\\)([^\\]|$)`. I.e. we match the character (or eof) after any odd number of backslashes
        const escaped_characters = escaped_characters: {
            // This code in this block brought to you courtesy of simdjson, licensed
            // under the Apache 2.0 license which is included at the bottom of main.zig.
            // Credit to John Keiser (@jkeiser) for designing this algorithm.
            // See https://github.com/simdjson/simdjson/pull/2042

            const ODD_BITS: uint = @bitCast(@as([@sizeOf(uint)]u8, @splat(0xaa)));
            // |                                | Mask (shows characters instead of 1's) | Depth | Instructions        |
            // |--------------------------------|----------------------------------------|-------|---------------------|
            // | string                         | `\\n_\\\n___\\\n___\\\\___\\\\__\\\`   |       |                     |
            // |                                | `    even   odd    even   odd   odd`   |       |                     |
            // | potential_escape               | ` \  \\\    \\\    \\\\   \\\\  \\\`   | 1     | 1 (backslashes & ~first_is_escaped)
            // | escape_and_terminal_code       | ` \n \ \n   \ \n   \ \    \ \   \ \`   | 5     | 5 (next_escape_and_terminal_code())
            // | escaped                        | `\    \ n    \ n    \ \    \ \   \ ` X | 6     | 7 (escape_and_terminal_code ^ (potential_escape | first_is_escaped))
            // | escape                         | `    \ \    \ \    \ \    \ \   \ \`   | 6     | 8 (escape_and_terminal_code & backslashes)
            // | first_is_escaped               | `\                                 `   | 7 (*) | 9 (escape >> 63) ()
            //                                                                               (*) this is not needed until the next iteration

            const next_is_escaped = topBit(carried.next_is_escaped);
            const potential_escape = o.backslashes & ~next_is_escaped;

            // If we were to just shift and mask out any odd bits, we'd actually get a *half* right answer:
            // any even-aligned backslashes runs would be correct! Odd-aligned backslashes runs would be
            // inverted (\\\ would be 010 instead of 101).
            //
            // ```
            // string:              | ____\\\\_\\\\_____ |
            // maybe_escaped | ODD  |     \ \   \ \      |
            //               even-aligned ^^^  ^^^^ odd-aligned
            // ```
            //
            // Taking that into account, our basic strategy is:
            //
            // 1. Use subtraction to produce a mask with 1's for even-aligned runs and 0's for
            //    odd-aligned runs.
            // 2. XOR all odd bits, which masks out the odd bits in even-aligned runs, and brings IN the
            //    odd bits in odd-aligned runs.
            // 3. & with backslashes to clean up any stray bits.
            // runs are set to 0, and then XORing with "odd":
            //
            // |                                | Mask (shows characters instead of 1's) | Instructions        |
            // |--------------------------------|----------------------------------------|---------------------|
            // | string                         | `\\n_\\\n___\\\n___\\\\___\\\\__\\\`   |
            // |                                | `    even   odd    even   odd   odd`   |
            // | maybe_escaped                  | `  n  \\n    \\n    \\\_   \\\_  \\` X | 1 (potential_escape << 1)
            // | maybe_escaped_and_odd          | ` \n_ \\n _ \\\n_ _ \\\__ _\\\_ \\\`   | 1 (maybe_escaped | odd)
            // | even_series_codes_and_odd      | `  n_\\\  _    n_ _\\\\ _     _    `   | 1 (maybe_escaped_and_odd - potential_escape)
            // | escape_and_terminal_code       | ` \n \ \n   \ \n   \ \    \ \   \ \`   | 1 (^ odd)
            //

            // Escaped characters are characters following an escape.
            const maybe_escaped = shl(potential_escape, 1);

            // To distinguish odd from even escape sequences, therefore, we turn on any *starting*
            // escapes that are on an odd byte. (We actually bring in all odd bits, for speed.)
            // - Odd runs of backslashes are 0000, and the code at the end ("n" in \n or \\n) is 1.
            // - Odd runs of backslashes are 1111, and the code at the end ("n" in \n or \\n) is 0.
            // - All other odd bytes are 1, and even bytes are 0.
            const even_series_codes_and_odd_bits = sub(maybe_escaped | ODD_BITS, potential_escape);

            // Now we flip all odd bytes back with xor. This:
            // - Makes odd runs of backslashes go from 0000 to 1010
            // - Makes even runs of backslashes go from 1111 to 1010
            // - Sets actually-escaped codes to 1 (the n in \n and \\n: \n = 11, \\n = 100)
            // - Resets all other bytes to 0
            const escape_and_terminal_code = even_series_codes_and_odd_bits ^ ODD_BITS;
            const escaped = escape_and_terminal_code ^ (o.backslashes | next_is_escaped);
            carried.next_is_escaped = escape_and_terminal_code & o.backslashes;
            break :escaped_characters escaped;
        };

        const unescaped_quotes = o.quotes & ~escaped_characters;
        const unescaped_apostrophes = o.apostrophes & ~escaped_characters;
        // Matches `\/(?=\/)`
        const double_slash_starts = o.slashes & (shr(o.slashes, 1) | topBit(carried.slashes));
        // Matches `\\(?=\\)`
        const double_backslash_starts = o.backslashes & (shr(o.backslashes, 1) | topBit(carried.backslashes));

        // PORT: The slides read the previous chunk's open construct from `source[last_start]`, with `last_start = 2` as a
        // placeholder. It is carried explicitly instead (see where `carried.inside_*` is set).
        const ended_inside_quote_string = carried.inside_quote_string;
        const ended_inside_char_literal = carried.inside_char_literal;
        const ended_inside_comment = carried.inside_comment;
        const ended_inside_line_string = carried.inside_line_string;
        const ended_inside_string_or_comment_or_char_literal =
            ended_inside_quote_string | ended_inside_char_literal | ended_inside_comment | ended_inside_line_string;

        // PORT: If we start inside of something, the first character can only continue (or close) that thing. Without this,
        // a comment carried over from the previous chunk whose next character is `"` would end at the next `"`, etc.
        const not_carried_into_quote = ended_inside_string_or_comment_or_char_literal & ~ended_inside_quote_string;
        const not_carried_into_char_literal = ended_inside_string_or_comment_or_char_literal & ~ended_inside_char_literal;
        const not_carried_into_comment = ended_inside_string_or_comment_or_char_literal & ~ended_inside_comment;
        const not_carried_into_line_string = ended_inside_string_or_comment_or_char_literal & ~ended_inside_line_string;

        const quote_bounds_incl_carry = (unescaped_quotes & ~not_carried_into_quote) ^ ended_inside_quote_string;
        const apostrophe_bounds_incl_carry = (unescaped_apostrophes & ~not_carried_into_char_literal) ^ ended_inside_char_literal;
        const comment_bounds_incl_carry = (double_slash_starts & ~not_carried_into_comment) | ended_inside_comment;
        const line_string_bounds_incl_carry = (double_backslash_starts & ~not_carried_into_line_string) | ended_inside_line_string;

        // We might have the end of some string/char_literal at the beginning of the chunk,
        // and in this case we need the (pseudo) start and end character to be the same.
        // That's why we handle this separately from the loop that follows, which only permits
        // starts to be different from ends.
        const first_char_ends_string = disjoint_or(
            unescaped_quotes & ended_inside_quote_string,
            unescaped_apostrophes & ended_inside_char_literal,
        );

        // Bitstrings that mark where each token begins and ends
        var all_starts: uint = first_char_ends_string;
        var all_ends: uint = first_char_ends_string;
        const MSb: uint = @bitCast(@as(std.meta.Int(.unsigned, @bitSizeOf(uint)), 1) << (VLEN - 1));

        // Figure out which characters are inside a quote, comment, line-string, or, to a lesser extent, a character literal.
        // The fundamental problem is that any of these could be arbitrarily nested inside of each other, meaning that a
        // 100% parallel tokenize is probably not going to be developed.
        // We could, however, operate in parallel on comments+line-strings because they both end on a '\n' (using pext & pdep).
        // " //'d' " // # | source
        // 1.......1..... | double quotes
        // ...1.......1.. | double slashes
        // ....1.1....... | single quotes
        // .............1 | newlines
        // We want to end with:
        // 1..........1.. | token starts
        // ........1....1 | token ends

        // Algorithm: Create an `iter` bitstring that has 1 bits as cursors at the first non-newline in each group of non-newlines (henceforth called a "line")
        // Take the OR of all the starting positions of { quotes, apostrophes, comments, line_strings }, call it `all_bounds_incl_carry`
        // For each bit in OR, let it heat-seek/advance to the next bit in `all_bounds_incl_carry` which comes after its position, stopping at newlines too.
        // This will gives us a bitstring that tells us, for each line, where the first comment/quote/char-literal/line-string begins.
        // Then we want to advance to the end of whatever we found in each position. If we found a string, we should stop at the next quote
        // character. If we found a character literal, we stop at this next apostrophe. Everything else, of course, stops at a newline, and it
        // is an error if the quotes/apostrophes terminate at a newline.
        // To accomplish this, we replace the bits from the starting position to the end of the line with whatever we hoped to find.
        // Then we advance the cursors forward, hopefully matching the right characters, hitting newlines otherwise.
        // By tracking all of our starting and ending positions, we have computed which pieces are nested within each other.

        // Note: We do not throw an error for a quote or character literal that is missing a terminator. That should happen in Sema.zig
        const carriages_or_newlines_or_eof = o.carriages | o.newlines_or_eof;

        {
            var all_bounds_incl_carry = quote_bounds_incl_carry | apostrophe_bounds_incl_carry | comment_bounds_incl_carry | line_string_bounds_incl_carry;
            var iter = ~carriages_or_newlines_or_eof & shld(carriages_or_newlines_or_eof, 1, MSb);

            while (true) {
                // Advance each cursor in `iter` to next 1 bit in `all_bounds_incl_carry | carriages_or_newlines_or_eof`,
                // but don't count `carriages_or_newlines_or_eof` as a valid start
                const starts = all_bounds_incl_carry & ~sub(all_bounds_incl_carry | carriages_or_newlines_or_eof, iter);
                all_starts |= starts;

                // From each bit in `starts` to the next newline, fold in the pieces of the target bitstring
                // that contains the end-characters we want to match against.
                // E.g. "...\n
                //      ^ we start a quote here
                // E.g. "...\n
                //       ^^^ so we put the characters from the `quote_bounds_incl_carry` bitstring in there
                var interleaved = carriages_or_newlines_or_eof;
                interleaved |= quote_bounds_incl_carry & sub(carriages_or_newlines_or_eof, starts & quote_bounds_incl_carry);
                interleaved |= apostrophe_bounds_incl_carry & sub(carriages_or_newlines_or_eof, starts & apostrophe_bounds_incl_carry);
                interleaved &= ~starts;

                // Advance cursor within interleaved bitstring
                const ends = interleaved & ~sub(interleaved, starts);
                all_ends |= ends;

                // Make `ends` the new iters, but if we ended on a newline, delete that bit.
                iter = ends & ~carriages_or_newlines_or_eof;
                all_bounds_incl_carry &= ~ends;

                if (isZero(iter)) break;
            }
        }

        const inside_strings_and_comments_including_start = sub(all_ends, all_starts);

        // PORT: Record which construct we end inside of, if any. (Replaces `ended_on.set(.inside_string_or_comment_or_char_literal, ...)`.)
        // It's whichever one the last start belongs to.
        carried.inside_quote_string = zero;
        carried.inside_char_literal = zero;
        carried.inside_comment = zero;
        carried.inside_line_string = zero;
        if (!isZero(inside_strings_and_comments_including_start & MSb)) {
            const last_start = highestBit(all_starts);
            const bit_0 = topBit(MSb);
            if (!isZero(last_start & quote_bounds_incl_carry))
                carried.inside_quote_string = bit_0
            else if (!isZero(last_start & apostrophe_bounds_incl_carry))
                carried.inside_char_literal = bit_0
            else if (!isZero(last_start & comment_bounds_incl_carry))
                carried.inside_comment = bit_0
            else
                carried.inside_line_string = bit_0;
        }

        // Revise `all_ends` such that tokens do not include the carriages_or_newlines_or_eof
        all_ends = (all_ends & ~carriages_or_newlines_or_eof) | shr(all_ends & carriages_or_newlines_or_eof, 1);

        // Add a dummy end if we ended inside a string/comment/line_string/char_literal
        all_ends = disjoint_or(all_ends, inside_strings_and_comments_including_start & MSb);

        const is_first_char_inside_string_or_comment = ended_inside_string_or_comment_or_char_literal &
            ~first_char_ends_string &
            ~carriages_or_newlines_or_eof;

        // zig fmt: off
        const ats                       = ~inside_strings_and_comments_including_start & o.ats;
        const alpha_underscores         = ~inside_strings_and_comments_including_start & o.alpha_underscores;
        const alpha_numeric_underscores = ~inside_strings_and_comments_including_start & o.alpha_numeric_underscores;
        // zig fmt: on

        const string_identifier_started_before_ext_chunk = unescaped_quotes & topBit(carried.ats);
        const builtin_started_before_ext_chunk = alpha_underscores & topBit(carried.ats);
        const is_comment_opener_split_between_ext_chunks = o.slashes & topBit(carried.slashes) & ~ended_inside_string_or_comment_or_char_literal;
        const is_line_string_opener_split_between_ext_chunks = o.backslashes & topBit(carried.backslashes) & ~ended_inside_string_or_comment_or_char_literal;
        const identifier_or_number_or_builtin_with_letters_started_before_ext_chunk =
            topBit(carried.identifier_or_number_or_builtin_ends) & alpha_numeric_underscores;

        // Why do we not pull in the last bit from the previous chunk when shifting by 1? Why do we unconditionally shift in a zero bit?
        // Because we need to have a start or pseudo-start at the beginning of a chunk when we have a token that crosses chunks.
        const follows_at = shl(ats, 1);
        const identifier_or_number_starts = alpha_numeric_underscores & ~shl(alpha_numeric_underscores, 1) & ~follows_at;
        const builtin_starts = (ats & shr(alpha_underscores, 1)) | builtin_started_before_ext_chunk;
        const string_identifier_starts = (ats & shr(unescaped_quotes, 1)) | string_identifier_started_before_ext_chunk;

        const identifier_or_number_or_builtin_ends = alpha_numeric_underscores & ~shr(alpha_numeric_underscores, 1);

        all_starts &= ~(follows_at & unescaped_quotes);
        all_starts |= identifier_or_number_starts | builtin_starts | string_identifier_starts;
        all_ends |= identifier_or_number_or_builtin_ends;

        // These may indicate the first char of a chunk, when a multi-char symbol started in the previous chunk
        const double_char_ends, const triple_char_ends = multi_char_ends: {
            const triple_char_ends = ~inside_strings_and_comments_including_start & o.prelim_triple_char_ends;

            // The rule for multi-char symbol matching is that we always want to match the longest possible symbol that we can.
            // That means that if there is a 3-char-end in the next position from a two-char-end, we can unset the two-char-end bit.
            // E.g.                 a +|= b
            //  double_char_ends <- 0001000 (we unset the 1 bit)
            //  triple_char_ends <- 0000100
            const double_char_ends = ~inside_strings_and_comments_including_start & o.prelim_double_char_ends & ~shr(triple_char_ends, 1);
            // ^ We could use XOR instead of ANDN in the above expression, so long as all 2-char prefixes of 3-char symbols are valid.

            // The idea is to iterate over a bitstring which is the bitwise-OR of `double_char_ends` and `triple_char_ends`.
            // We iteratively find the lowest 1 bit, unset that bit, then unset one, possibly two bits after that bit, and repeat.
            // We would unset two bits after if that bit corresponds to the end of a triple-char symbol. E.g.
            //
            //                                   a =>>= b
            //  double_char_ends | triple_char_ends <- 00011100
            //                                       ^ always unset this bit, because we shouldn't match `=>` and `>>`
            //                                        ^ unset this bit if there's a potential 3-char-symbol here. We shouldn't match `=>` and `>>=`.
            var s = double_char_ends | triple_char_ends;

            // Will be a subset of `double_char_ends` | `triple_char_ends`, but with the boundaries properly figured out.
            var ends: uint = std.mem.zeroes(uint);

            // If a triple-char-symbol-end is at the beginning of the next chunk, we unset it if the last chunk ended with a multi-char symbol within two chars. E.g.:
            //                                                                     a =>>= b
            // ends <- 0000000000000000000000000000000000000000000000000000000000000001
            //                                                           next chunk -> 1100000000000000000000000000000000000000000000000000000000000000
            //                                                                          ^ unset this one since it's a triple-char-end within two positions of the previous symbol-end
            //                                                                    a =>>= b
            // ends <- 0000000000000000000000000000000000000000000000000000000000000010
            //                                                           next chunk -> 1000000000000000000000000000000000000000000000000000000000000000
            //                                                                         ^ unset this one since it's a triple-char-end within two positions of the previous symbol-end

            s &= ~(carried.delete_triple_char_pos & ~double_char_ends);

            // This handles the case where a chunk ends on a double-char symbol, only to find in the next chunk that it's a triple-char symbol.
            //                                                                     a +%= b
            // ends <- 0000000000000000000000000000000000000000000000000000000000000001
            //                                                           next chunk -> 100000000000000000000000000000000000000000000000000000000000000
            //                                                                         ^ this is valid because it's a triple-char-end
            //                                                                     a ++= b
            // ends <- 0000000000000000000000000000000000000000000000000000000000000001
            //                                                           next chunk -> 100000000000000000000000000000000000000000000000000000000000000
            //                                                                         ^ unset because it's a double-char-end
            s &= ~andn(topBit(carried.double_char), triple_char_ends);

            while (true) {
                // Optimization: Instead of iterating over a single position at once, we iterate over multiple positions at once
                // that have at least two "spaces" (not-possible-to-be-multi-char-symbol chars) in between them. (Or at the start.)
                // For a "pathological" input of randomly assorted 2 and 3-char symbols that overlap a lot, we have to iterate from
                // start to finish, in order, because we cannot tell in parallel where all the symbols start.
                // However, in real Zig code, you typically have "spaces" in between the symbols, sufficient that you can guarantee that
                // the first character in a group is the start of a multi-char symbol.
                const iter = s & ~shl(s, 1) & ~shl(s, 2);
                ends |= iter;

                // We want to unset the current bit(s) pointed to by `iter`, as well as the next bit, given by `iter << 1`,
                // because you can't have a 2 or 3 character symbol end in the next byte after the end of a 2 or 3 character symbol.
                // E.g.         a  b
                //      iter <- 00010000
                //      s    <- 00011100
                //                  ^ always unset this bit, because we can't match both `=>` and `>>` since they overlap

                // We also want to zero two bits after, if that corresponds to a non-2-char symbol `(iter << 2) & ~double_char_ends`.
                // E.g.                    .....
                //                 iter <- 00100
                //                 s    <- 00111
                //     double_char_ends <- 00001
                //     triple_char_ends <- 00111
                //                           ^ always unset
                //                            ^ not unset because this happens to be a valid 2-char-symbol
                // E.g.                    ......
                //                 iter <- 001000
                //                 s    <- 001111
                //     double_char_ends <- 000001
                //     triple_char_ends <- 001111
                //                            ^ always unset
                //                             ^ unset because this is a valid 3-char-symbol but not a valid 2-char-symbol (due to the next char making a 3-char sequence, see `double_char_ends &= ...` above)
                // E.g.         a =>>= b
                //      iter <- 00010000
                //      s    <- 00011100
                //                   ^ unset this bit because there's a potential 3-char-symbol-end within two chars of one of the `iter` bits.
                //                     I.e., we can't match both `=>` and `>>=` because they overlap.

                s &= ~(disjoint_or(iter, shl(iter, 1)) | andn(shl(iter, 2), double_char_ends));
                //           ^ All 1's in `iter` must have at least two 0's between them. The compiler gives us the shift+add in a single `lea` instruction :)

                if (isZero(s)) {
                    @branchHint(.likely);
                    break;
                }

                // We could advance each 1 bit in `iter` to the next 1 bit in `s` using `iter = s & ~(s -% iter);`,
                // but it's pretty much impossible for this loop to branch in practice, so this optimization would be pointless.
            }

            // 3-char sequences take precedence over two-char sequences, except when they are just two characters apart.
            const double_char_ends_final = (ends & ~triple_char_ends) | (ends & shl(ends, 2));
            const triple_char_ends_final = ends ^ double_char_ends_final;
            // Alternative implementation:
            //const triple_char_ends_final = ((ends & triple_char_ends) & ~(ends & (ends << 2)));
            //const double_char_ends_final = ends ^ triple_char_ends_final;

            // PORT: these two were `defer`s writing to `self.delete_triple_char_pos_carry` and `ended_on`, which didn't exist.
            carried.delete_triple_char_pos = shr(ends, VLEN - 2);
            carried.double_char = ends & double_char_ends;

            break :multi_char_ends .{ double_char_ends_final, triple_char_ends_final };
        };

        const single_char_ends =
            o.possible_one_char_symbols &
            ~inside_strings_and_comments_including_start &
            // Delete single_char_ends that are already covered by `double_char_ends` and `triple_char_ends`
            ~(double_char_ends | shr(double_char_ends, 1)) &
            ~(triple_char_ends | shr(triple_char_ends, 1) | shr(triple_char_ends, 2));

        const multi_char_symbol_ends = double_char_ends | triple_char_ends;
        const all_symbol_ends = single_char_ends | multi_char_symbol_ends;

        const multi_char_symbol_starts = shr(double_char_ends, 1) | shr(triple_char_ends, 2);
        const _1: uint = @bitCast(@as(std.meta.Int(.unsigned, @bitSizeOf(uint)), 1));
        const multi_char_has_two_chars_in_prev_chunk = _1 & triple_char_ends;
        const multi_char_has_one_char_in_prev_chunk = _1 & (double_char_ends | shr(triple_char_ends, 1));
        const multi_char_starts_before_chunk = multi_char_has_one_char_in_prev_chunk | multi_char_has_two_chars_in_prev_chunk;
        const all_multi_char_starts = multi_char_symbol_starts | multi_char_starts_before_chunk;

        // PORT: Phase III classifies multi-char symbols one 64-byte chunk at a time, from their ends. For that it needs, per
        // 64-byte chunk, the starts of the symbols that *end* in it, with a pseudo-start at bit 0 for a symbol that started
        // in the previous 64-byte chunk. (The slides only handled that across 512-byte chunks.)
        const lane_bit_0: uint = @splat(1);
        const multi_char_crosses_into_lane = lane_bit_0 & (double_char_ends | triple_char_ends | (triple_char_ends >> @splat(1)));
        const multi_char_starts_by_end_lane = (double_char_ends >> @splat(1)) | (triple_char_ends >> @splat(2)) | multi_char_crosses_into_lane;

        all_starts |= single_char_ends | all_multi_char_starts;
        all_ends |= all_symbol_ends;

        // PORT: The slides computed `invalids` but never turned them into tokens (and the expression also included
        // characters that were already part of a token). Like the 64-byte tokenizer, every character that is neither
        // whitespace nor part of a token becomes a 1-character `invalid` token.
        const invalids = ~(sub(all_ends, all_starts) | all_ends | o.whitespaces);
        all_starts |= invalids;
        all_ends |= invalids;

        // Bad-controls that aren't already in `invalids` are inside of a string/comment/line_string/char_literal
        // " abc __ abd"
        // 1.....11..... <- starts
        // ......11....1 <- ends

        // ----------------------------------------------------------------------------------------------------- //
        // Note: For these variables, we don't know if we are inside a number from the previous chunk, so our analysis
        // shouldn't assume we know these are identifiers. E.g.
        // 2|e5
        //  ^ chunk boundary
        // Conversely, this will not match identifiers like:
        // if|2e5
        //   ^ chunk boundary
        // This behavior is just fine, however, for keyword matching, because a number can't be anywhere in a keyword.

        // There is trade-off when deciding which things to prune from being checked by the
        // algorithm. Up to 8 identifier/number looking things can be checked with no impact to speed. So in cases where there are more than that,
        // early pruning could help if it takes us below that threshold. But if our solution makes every other chunk (where this situation doesn't
        // matter) slower, than the overall cost we pay for ALL chunks is usually going to be higher than slightly reducing the cost of a single
        // chunk's tokenization. I.e. if we make one iteration 5% faster, but all other iterations are 0.1% slower, then the cumulative effect of
        // the 0.1% pessimization per iteration may be significantly higher if the 5% faster case is rare enough to not overcome the general
        // pessimization. We can make "good enough" decisions in this category by just speedtesting different options. It's possible that we
        // observe compiler artifacts rather than inherent issues. For example, if we use `identifier_or_number_starts`, that makes the common case
        // not need an additional computation and perhaps not need an additional register. But maybe the compiler is going to spill
        // `identifier_or_number_starts` to the stack no matter what, and so whatever we do that prunes identifiers/numbers from being keywords
        // is perhaps more likely to be worth it. There are also compiler issues where the compiler will spill `a` and `b` at the same time,
        // then read them back in to do an AND. What it should have done is do the AND, then spill. That way you only need 1 spill and one fill
        // rather than 2. Unfortunately this can happen even for expressions specified in a single line of code.
        // See: https://github.com/llvm/llvm-project/issues/113275
        // Some CPU's have register aliasing which impacts the situation too. That's where spilling and filling to memory can be speculatively
        // backed by internal CPU registers. This makes the cost of spilling and filling to memory just 1 cycle (or less) in cases where the CPU
        // can employ this trick. To my knowledge, this optimization only exists on certain micro-architectures and only works on general-purpose
        // registers. I know of no reason why the same trick couldn't be applied to SIMD registers, but it simply hasn't been done on general-
        // purpose CPU's to my knowledge. Not that I've read all of the materials in full that I should have.

        // PORT: In the slides, any letter at bit 0 of a 64-byte chunk started a keyword candidate, so the tail of a number
        // or builtin that crossed into this 64-byte chunk could be classified as a keyword (e.g. `1|or`, `@|export`). A
        // candidate now can't follow an `@`, and bit 0 only starts one if it really starts a word, or if it continues a
        // candidate from the previous 64-byte chunk.
        const prev_chunk_ended_on_alpha_numeric_underscore = std.simd.mergeShift(carried.alpha_numeric_underscores, alpha_numeric_underscores, 7) >> @splat(63);
        const local_word_starts =
            alpha_underscores &
            ~(alpha_numeric_underscores << @splat(1)) &
            ~shld(ats, 1, carried.ats) &
            ~prev_chunk_ended_on_alpha_numeric_underscore;

        // While it is acceptable for `possible_kw_starts` to benefit from whatever degree of cross-chunk knowledge we can easily provide it,
        // we want everything else to understand itself only in terms of 64 byte chunks and not see across those boundaries. Hence
        // we do not use the generic cross-chunk constructs for this analysis.
        const local_identifier_or_number_or_builtin_ends = alpha_numeric_underscores & ~(alpha_numeric_underscores >> @splat(1));
        // Candidates that start in this 64-byte chunk. (A word that fills a whole 64-byte chunk can't be a keyword anyway.)
        const local_word_ends = local_identifier_or_number_or_builtin_ends & ~(local_identifier_or_number_or_builtin_ends -% local_word_starts);

        // There are at most 32 identifiers in 64-byte chunk.
        // There are at most 22 multi-character identifiers in a 64-byte chunk, including a carry-over from the previous one
        // i|f(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if
        //  ^ break between previous chunk and current chunk
        const prev_chunk_ended_on_possible_kw = std.simd.mergeShift(carried.local_word_ends, local_word_ends, 7) >> @splat(63);
        const possible_kw_started_before_chunk = prev_chunk_ended_on_possible_kw & alpha_numeric_underscores;
        const possible_kw_starts = local_word_starts | possible_kw_started_before_chunk;
        const possible_kw_ends = local_identifier_or_number_or_builtin_ends & ~(local_identifier_or_number_or_builtin_ends -% possible_kw_starts);

        // Alternatively, we could try having `possible_kw_started_before_chunk` be ` & lower_alpha` instead of ` & alpha_numeric_underscores`
        const possible_kw_starts_longer_than_1 = possible_kw_starts & (~possible_kw_ends | prev_chunk_ended_on_possible_kw);
        // `first_two_chars_mask` can grab a dummy bit in the case where we have an identifier split between chunks
        // and only 1 character in this chunk. E.g. i|f
        const first_two_chars_mask = disjoint_or(possible_kw_starts_longer_than_1, possible_kw_starts_longer_than_1 << @splat(1));

        // Add dummy bit to second bit position if we have only one character of a keyword started in the previous chunk
        const possible_kw_with_one_char_in_chunk = prev_chunk_ended_on_possible_kw & possible_kw_ends;
        const possible_kw_ends_longer_than_1 = (possible_kw_ends & ~possible_kw_starts) | (possible_kw_with_one_char_in_chunk << @splat(1));
        // `last_two_chars_mask` can grab a dummy bit in the case where we have an identifier split between chunks
        // nd only 1 character in this chunk. E.g. i|f
        const last_two_chars_mask = disjoint_or(possible_kw_ends_longer_than_1, possible_kw_ends_longer_than_1 >> @splat(1));

        const prev_carried_lens = @as(uint, @splat(1)) + @as(uint, @clz(std.simd.mergeShift(carried.all_starts, all_starts, 7)));
        const possible_kw_carried_lens = prev_carried_lens & -%possible_kw_started_before_chunk;

        // Assert that there are an equal number of starts and ends, and that they are paired up in non-overlapping ways
        assert(@popCount(@as(u512, @bitCast(all_ends))) == @popCount(@as(u512, @bitCast(all_starts))));
        assert(isZero((all_ends & ~sub(all_ends, all_starts)) ^ all_ends));

        // if a builtin or a number or an identifier that is too long
        const first_token_starts_before_ext_chunk =
            is_first_char_inside_string_or_comment |
            first_char_ends_string |
            string_identifier_started_before_ext_chunk |
            builtin_started_before_ext_chunk |
            is_comment_opener_split_between_ext_chunks |
            is_line_string_opener_split_between_ext_chunks |
            (_1 & multi_char_starts_before_chunk) |
            // If identifier is short enough to be a keyword, we handle it in the keyword-specific logic
            identifier_or_number_or_builtin_with_letters_started_before_ext_chunk;

        // If the first token started before the current chunk, go backwards and save its start and kind.
        // PORT: The slides loaded these into vectors through `cur_start[0].start`, which doesn't fit the
        // `kind`/`start`/`end` layout; we read the previous token's start and kind as scalars.
        const first_token_crosses: u1 = @intCast(first_token_starts_before_ext_chunk[0]);
        cur_kind -= first_token_crosses;
        cur_start -= first_token_crosses;
        cur_end -= first_token_crosses;
        const carried_start: u32 = if (first_token_crosses == 1) cur_start[0] else 0;
        const carried_kind: u8 = if (first_token_crosses == 1) cur_kind[0] else 0;

        var carry_kind_mask = @as(
            @Vector(64, u8),
            // The ANDN here is because we don't want to carry an `invalid` type across boundaries.
            @bitCast(first_token_starts_before_ext_chunk & ~is_line_string_opener_split_between_ext_chunks),
        ) == @as(@Vector(64, u8), @splat(0));

        // In "reasonable" code, the maximum number of multi-char symbols in a 64-byte chunk is 13.
        // "*% b *% b *% b *% b *% b *% b *% b *% b *% b *% b *% b *% b *% b"
        // Therefore, we can hide anything crazier than that behind a branch.
        // We just use a single condition here because we expect this to be so rare.
        const contains_multi_char_spam = @reduce(.Or, @popCount(multi_char_symbol_ends) >= @as(@Vector(8, u7), @splat(16)));

        // PORT: Which 64-byte chunks must overwrite the kind of the token that crossed into them from the previous
        // 64-byte chunk: a keyword candidate or multi-char symbol, whose kind isn't known until we see its end. The first
        // 64-byte chunk instead rewinds the whole token (see `first_token_crosses`).
        const kind_rewrites = (possible_kw_started_before_chunk | multi_char_crosses_into_lane) & ~_1;

        const k_masks = struct {
            builtin_starts: [8]@Vector(64, bool),
            string_identifier_starts: [8]@Vector(64, bool),
            comment_bounds_incl_carry: [8]@Vector(64, bool),
            invalids: [8]@Vector(64, bool),
        }{
            .builtin_starts = @bitCast(builtin_starts),
            .string_identifier_starts = @bitCast(string_identifier_starts),
            .comment_bounds_incl_carry = @bitCast(comment_bounds_incl_carry),
            .invalids = @bitCast(invalids),
        };

        // PORT: The slides indexed these 512-bit bitstrings with the runtime `j` below; arrays make that explicit.
        // zig fmt: off
        const all_starts_lanes: [8]u64                       = all_starts;
        const all_ends_lanes: [8]u64                         = all_ends;
        const multi_char_symbol_ends_lanes: [8]u64           = multi_char_symbol_ends;
        const triple_char_ends_lanes: [8]u64                 = triple_char_ends;
        const multi_char_starts_by_end_lane_lanes: [8]u64    = multi_char_starts_by_end_lane;
        const first_two_chars_mask_lanes: [8]u64             = first_two_chars_mask;
        const last_two_chars_mask_lanes: [8]u64              = last_two_chars_mask;
        const possible_kw_starts_longer_than_1_lanes: [8]u64 = possible_kw_starts_longer_than_1;
        const possible_kw_carried_lens_lanes: [8]u64         = possible_kw_carried_lens;
        const possible_kw_with_one_char_lanes: [8]u64        = possible_kw_with_one_char_in_chunk;
        const kind_rewrites_lanes: [8]u64                    = kind_rewrites;
        // zig fmt: on

        for (
            chunks,
            classified_chunks,
            0..,
        ) |loaded_chunk, classified_chunk, j| {
            // This is a vector that holds the `kind` of each token.
            // It's okay to have tags broadcasted to places that will not be extracted in the end
            var kinds = classified_chunk; // PORT: was `classified_chunks`
            kinds = @select(u8, carry_kind_mask, kinds, @as(@Vector(64, u8), @splat(carried_kind)));

            inline for (
                [_]struct { @Vector(64, bool), @Vector(64, u8) }{
                    // zig fmt: off
                    .{ k_masks.builtin_starts[j],            @splat(@intFromEnum(Tag.builtin)) },
                    .{ k_masks.string_identifier_starts[j],  @splat(@intFromEnum(Tag.string_identifier)) },
                    .{ k_masks.comment_bounds_incl_carry[j], @splat(@intFromEnum(Tag.@"//")) },
                    .{ k_masks.invalids[j],                  @splat(@intFromEnum(Tag.invalid)) },
                    // zig fmt: on
                },
            ) |mask_and_tags| {
                const mask, const tags = mask_and_tags;
                kinds = @select(u8, mask, tags, kinds);
            }

            const multi_char_end_indices = simd.bitsToIndices(multi_char_symbol_ends_lanes[j]);

            // Maximally, we can have 32 2 character symbols in a 64-byte chunk.
            const which_multi_char_symbols_are_triple_chars_compacted = simd.pext(triple_char_ends_lanes[j], multi_char_symbol_ends_lanes[j]);

            const helper = struct {
                fn getMultiCharKinds(
                    prev_chunk_: @Vector(64, u8),
                    chunk_: @Vector(64, u8),
                    multi_char_end_indices_: @Vector(16, u8),
                    which_multi_char_symbols_are_triple_chars_compacted_: @Vector(16, bool),
                ) @Vector(16, u8) {
                    const adjustment_vec: @Vector(64, u8) = @bitCast(@select(
                        u32,
                        which_multi_char_symbols_are_triple_chars_compacted_,
                        [_]u32{@bitCast([4]u8{ 2, 1, 0, 0x80 })} ** 16, // 3 characters, zero the uppermost byte
                        [_]u32{@bitCast([4]u8{ 1, 0, 0x80, 0x80 })} ** 16, // 2 characters, zero the 2 upper bytes
                    ));

                    // PORT: `undefined` instead of a zero-length vector as @shuffle's second operand.
                    const successive_indices = @shuffle(u8, multi_char_end_indices_, undefined, [_]i32{
                        0,  0,  0,  0,
                        1,  1,  1,  1,
                        2,  2,  2,  2,
                        3,  3,  3,  3,

                        4,  4,  4,  4,
                        5,  5,  5,  5,
                        6,  6,  6,  6,
                        7,  7,  7,  7,

                        8,  8,  8,  8,
                        9,  9,  9,  9,
                        10, 10, 10, 10,
                        11, 11, 11, 11,

                        12, 12, 12, 12,
                        13, 13, 13, 13,
                        14, 14, 14, 14,
                        15, 15, 15, 15,
                    }) -% adjustment_vec;

                    // PORT: was an `@shuffle` with a bool vector as its first operand; a zero-masked lookup was meant.
                    const multi_char_symbol_strs_zero_padded = @select(
                        u8,
                        adjustment_vec < @as(V, @splat(0x80)),
                        simd.perm2b(std.simd.join(chunk_, prev_chunk_), successive_indices),
                        @as(V, @splat(0)),
                    );

                    var hash: @Vector(16, u8) = @truncate(
                        @as(@Vector(16, u32), @bitCast(multi_char_symbol_strs_zero_padded)) *% ([1]u32{698839068} ** 16) >> @splat(25),
                    );

                    // It would be cool if we changed our hash function so that it didn't require these post-processing steps.
                    // Everytime we do a 7-bit addition or subtraction, there is implicitly a bitwise AND that must be inserted.
                    // This hash function is from the old version, which had more inputs because it had to accept single-character symbols.
                    // We could probably find a better one
                    hash -%= @as(@TypeOf(hash), @splat(29));
                    hash = @as(@Vector(16, u7), @truncate(hash));

                    // TODO: On ARM, we can fold this into a single TBX instruction (plus a vector constant load)
                    hash = @select(
                        u8,
                        hash < @as(@TypeOf(hash), @splat(16)),
                        hash +% @as(@Vector(16, u8), @splat(18)),
                        hash,
                    );
                    return hash;
                }
            };

            const multi_char_kinds_compressed: @Vector(32, u8) = std.simd.join(
                helper.getMultiCharKinds(
                    prev_chunk_for_pt3,
                    loaded_chunk,
                    std.simd.extract(multi_char_end_indices, 0, 16),
                    @bitCast(@as(u16, @truncate(which_multi_char_symbols_are_triple_chars_compacted))),
                ),

                if (!contains_multi_char_spam)
                    @as(@Vector(16, u8), undefined)
                else blk: {
                    @branchHint(.unlikely);
                    break :blk helper.getMultiCharKinds(
                        @splat(0),
                        loaded_chunk,
                        std.simd.extract(multi_char_end_indices, 16, 16),
                        @bitCast(@as(u16, @truncate(which_multi_char_symbols_are_triple_chars_compacted >> 16))),
                    );
                },
            );

            // PORT: `simd.expand` takes the fallback first, and the symbols are located by the lane they end in.
            kinds = simd.expand(kinds, simd.pad_with_undefineds(@Vector(64, u8), multi_char_kinds_compressed), multi_char_starts_by_end_lane_lanes[j]);

            { // Get keyword extents
                const first_two_bytes_only = simd.padZeros(@Vector(64, bool), @Vector(2, bool){ true, true });

                // PORT: The carried lengths are read for this 64-byte chunk directly, rather than shifted down a lane per iteration.
                const start_indices = simd.bitsToIndices(first_two_chars_mask_lanes[j]) -% @select(
                    u8,
                    first_two_bytes_only,
                    @as(@Vector(64, u8), @splat(@truncate(possible_kw_carried_lens_lanes[j]))),
                    @as(@Vector(64, u8), @splat(0)),
                );
                const end_indices = simd.bitsToIndices(last_two_chars_mask_lanes[j]) -% @select(
                    u8,
                    first_two_bytes_only,
                    @as(@Vector(64, u8), @splat(@truncate(possible_kw_with_one_char_lanes[j]))),
                    @as(@Vector(64, u8), @splat(0)),
                );
                const identifier_lens = end_indices -% start_indices +% @as(@Vector(64, u8), @splat(2));

                const first_two_chars: @Vector(32, u16) = @bitCast(simd.perm2b(std.simd.join(loaded_chunk, prev_chunk_for_pt3), start_indices));
                const last_two_chars: @Vector(32, u16) = @bitCast(simd.perm2b(std.simd.join(loaded_chunk, prev_chunk_for_pt3), end_indices));

                // Hash is just the upper byte of the u16's, ignoring the uppermost bit.
                // We do a `>> splat(8)` to move it to the lower byte, that way we can reuse `first8Interlacedbroadcast` below

                const hash = Keywords.hash(32, @bitCast(identifier_lens), first_two_chars, last_two_chars);

                const super_looked_up_starts = simd.perm2b(Keywords.super_string_starts, hash);
                const super_looked_up_lens = simd.perm2b(Keywords.super_string_lens, hash);

                // TODO: remove this hack?
                // Written in a weird way because of https://github.com/llvm/llvm-project/issues/111431
                var matched1: @Vector(8, bool) = undefined;
                var matched2: @Vector(16, bool) = undefined;
                var matched: @Vector(32, bool) = undefined;

                // We can match 8 keywords at a time, because that's how many subdivisions we get in a 64-byte vector.
                // I.e. the maximum granularity of a compare-equal is 8 bytes, and we can fit 8 of those in a 64-byte vector.
                // There are at most 22 multi-character identifiers in a chunk, including a carry-over from the previous chunk
                // i|f(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if
                //  ^ break between previous chunk and current chunk
                const REQUIRED_ITERATIONS = comptime try std.math.divCeil(usize, 22, 8);

                // PORT: `inline`, because the @shuffle masks below depend on `k`.
                outer: inline for (0..REQUIRED_ITERATIONS) |k| {
                    const first8_interlaced_broadcast = @Vector(64, u8){
                        0,  0,  0,  0,  0,  0,  0,  0,
                        2,  2,  2,  2,  2,  2,  2,  2,
                        4,  4,  4,  4,  4,  4,  4,  4,
                        6,  6,  6,  6,  6,  6,  6,  6,
                        8,  8,  8,  8,  8,  8,  8,  8,
                        10, 10, 10, 10, 10, 10, 10, 10,
                        12, 12, 12, 12, 12, 12, 12, 12,
                        14, 14, 14, 14, 14, 14, 14, 14,
                    } + @as(@Vector(64, u8), @splat(16 * k));

                    const lookup_starts_group = @shuffle(u8, super_looked_up_starts, undefined, first8_interlaced_broadcast);
                    const lookup_lens_group = @shuffle(u8, super_looked_up_lens, undefined, first8_interlaced_broadcast);

                    const starts_group = @shuffle(u8, start_indices, undefined, first8_interlaced_broadcast);
                    const lens_group = @shuffle(u8, identifier_lens, undefined, first8_interlaced_broadcast);

                    var cur_matched: @Vector(8, bool) = @splat(true);

                    inline for ([2]@Vector(64, u8){
                        comptime std.simd.repeat(64, [8]u8{ 0, 1, 2, 3, 4, 5, 6, 7 }),
                        comptime std.simd.repeat(64, [8]u8{ 8, 9, 10, 11, 12, 13, 14, 15 }),
                    }) |indices| {
                        const keyword_candidates_from_chunk_half = @select(
                            u8,
                            indices < lens_group,
                            simd.perm2b(std.simd.join(loaded_chunk, prev_chunk_for_pt3), starts_group +% indices),
                            lens_group,
                        );

                        const keywords_from_table2_half = @select(
                            u8,
                            indices < lookup_lens_group,
                            simd.perm4b(
                                comptime @bitCast(Keywords.superstring[0..Keywords.superstring.len].* ++ [1]u8{0} ** (256 - Keywords.superstring.len)), // PORT: `@bitCast`
                                lookup_starts_group +% indices,
                            ),
                            lookup_lens_group,
                        );

                        // PORT: bool vectors don't support `&=` in Zig 0.14.
                        cur_matched = @select(
                            bool,
                            cur_matched,
                            @as(@Vector(8, u64), @bitCast(keyword_candidates_from_chunk_half)) ==
                                @as(@Vector(8, u64), @bitCast(keywords_from_table2_half)),
                            @as(@Vector(8, bool), @splat(false)),
                        );
                    }

                    if (k == 0) {
                        if (@popCount(possible_kw_starts_longer_than_1_lanes[j]) <= 8) {
                            @branchHint(.likely);
                            matched = simd.padZeros(@Vector(32, bool), cur_matched);
                            break :outer;
                        }
                    }

                    // TODO: remove this hack?
                    // Written in a weird way because of https://github.com/llvm/llvm-project/issues/111431
                    switch (k) {
                        0 => matched1 = cur_matched,
                        1 => matched2 = std.simd.join(matched1, cur_matched),
                        2 => matched = std.simd.join(matched2, simd.padZeros(@Vector(16, bool), cur_matched)),
                        else => unreachable,
                    }
                }

                // The hash is 7 bits, but in our vector each one still occupies 2 bytes of space.
                // Reduce it to a single byte.
                // Set the highest bit of each byte because that is how we fit the keywords into the `Tag` enum.
                const potential_kinds =
                    @shuffle(u8, hash, undefined, std.simd.iota(u8, 32) << @splat(1)) |
                    @as(@Vector(32, u8), @splat(128));

                const matched_kinds = @select(
                    u8,
                    matched,
                    potential_kinds,
                    @as(@Vector(32, u8), @splat(@intFromEnum(Tag.identifier))),
                );

                // PORT: `simd.expand` takes the fallback first (the slides also had `kinds.*`, but `kinds` isn't a pointer).
                kinds = simd.expand(kinds, simd.pad_with_undefineds(@Vector(64, u8), matched_kinds), possible_kw_starts_longer_than_1_lanes[j]);
            }

            // PORT: The write-out below targets the `kind`/`start`/`end` layout; the slides still wrote interleaved
            // `cur_extents`. Starts and ends are written independently, so a token that crosses into the next 64-byte
            // chunk simply has its end written there. When the crossing token's kind is only known now, we go back one
            // slot and write over the kind that the previous 64-byte chunk guessed.
            const kind_rewrite: u1 = @intCast(kind_rewrites_lanes[j] & 1);
            const kind_dest = cur_kind - kind_rewrite;
            kind_dest[0..64].* = simd.compact_z(kinds, all_starts_lanes[j] | kind_rewrite);

            const lane_pos = chunk_pos +% @as(u32, @intCast(64 * j));
            const start_indices: @Vector(64, u32) = simd.bitsToIndices(all_starts_lanes[j]);
            const end_indices: @Vector(64, u32) = simd.bitsToIndices(all_ends_lanes[j]);
            cur_start[0..64].* = start_indices +% @as(@Vector(64, u32), @splat(lane_pos));
            cur_end[0..64].* = end_indices +% @as(@Vector(64, u32), @splat(lane_pos +% 1)); // Exclusive, like std.zig.Token.Loc
            if (j == 0 and first_token_crosses == 1) cur_start[0] = carried_start;

            const num_starts = @popCount(all_starts_lanes[j]);
            cur_kind += num_starts;
            cur_start += num_starts;
            cur_end += @popCount(all_ends_lanes[j]);

            prev_chunk_for_pt3 = loaded_chunk;
            carry_kind_mask = @splat(true);
        }

        // PORT: was `carry.set(...)` calls on a `carry` that didn't exist.
        carried.carriages = o.carriages;
        carried.slashes = o.slashes;
        carried.backslashes = o.backslashes;
        carried.ats = ats;
        carried.alpha_numeric_underscores = alpha_numeric_underscores;
        carried.identifier_or_number_or_builtin_ends = identifier_or_number_or_builtin_ends;
        carried.local_word_ends = local_word_ends;
        carried.all_starts = all_starts;

        if (reached_eof) break;
    }

    // Because the main loop incorrectly allows a trailing carriage return, throw here if applicable.
    if (source.len != 0 and source[source.len - 1] == '\r') return error.InvalidCarriageReturn;

    // PORT: The slides called an undefined `utf8_validator` for chunks with non-ASCII bytes. Those are rare in source
    // code, so we validate them afterwards with the standard library. Only the span from the first to the last chunk
    // with non-ASCII bytes needs checking: no UTF-8 sequence can start or end in a chunk that is all ASCII.
    if (non_ascii_start < non_ascii_end and
        !std.unicode.utf8ValidateSlice(source[non_ascii_start..@min(non_ascii_end, source.len)]))
        return error.InvalidUtf8;

    self.token_list.len = @intFromPtr(cur_kind) - @intFromPtr(kinds_base);
}

/// PORT: The slides build these types with Zig 0.16's `@Struct`; Zig 0.14 spells it with `@Type`.
fn ExternStruct(comptime names: []const []const u8, comptime T: type) type {
    var fields: [names.len]std.builtin.Type.StructField = undefined;
    for (&fields, names) |*field, name| field.* = .{
        .name = std.fmt.comptimePrint("{s}", .{name}),
        .type = T,
        .default_value_ptr = null,
        .is_comptime = false,
        .alignment = @alignOf(T),
    };
    return @Type(.{ .@"struct" = .{
        .layout = .@"extern",
        .fields = &fields,
        .decls = &.{},
        .is_tuple = false,
    } });
}

/// PORT: The last bit of a 512-bit bitstring, moved to bit 0.
fn topBit(x: @Vector(8, u64)) @Vector(8, u64) {
    return .{ x[7] >> 63, 0, 0, 0, 0, 0, 0, 0 };
}

/// PORT: The highest set bit of a 512-bit bitstring (or 0).
fn highestBit(x: @Vector(8, u64)) @Vector(8, u64) {
    const nonzero_lanes: u8 = @bitCast(x != @as(@Vector(8, u64), @splat(0)));
    if (nonzero_lanes == 0) return @splat(0);
    const lane = 7 - @clz(nonzero_lanes);
    const lanes: [8]u64 = x;
    var result: [8]u64 = @splat(0);
    result[lane] = @as(u64, 1) << @intCast(63 - @clz(lanes[lane]));
    return result;
}

fn shl(a: anytype, comptime i: std.math.Log2Int(std.meta.Int(.signed, @bitSizeOf(@TypeOf(a))))) @TypeOf(a) {
    switch (@typeInfo(@TypeOf(a))) {
        .int => return a << i,
        else => {
            if (i % 64 == 0) return std.simd.shiftElementsRight(a, i / 64, 0);
            return (std.simd.shiftElementsRight(a, 1 + i / 64, 0) >> @splat(64 - (i % 64))) |
                (std.simd.shiftElementsRight(a, i / 64, 0) << @splat(i % 64));
        },
    }
}

fn shr(a: anytype, comptime i: std.math.Log2Int(std.meta.Int(.signed, @bitSizeOf(@TypeOf(a))))) @TypeOf(a) {
    switch (@typeInfo(@TypeOf(a))) {
        .int => return a >> i,
        else => {
            if (i % 64 == 0) return std.simd.shiftElementsLeft(a, i / 64, 0);
            return std.simd.shiftElementsLeft(a, 1 + i / 64, 0) << @splat(64 - (i % 64)) |
                std.simd.shiftElementsLeft(a, i / 64, 0) >> @splat(i % 64);
        },
    }
}

fn cmhi(a: @Vector(2, u64), b: @Vector(2, u64)) @Vector(2, u64) {
    return @select(
        u64,
        a < b,
        ~@as(@TypeOf(a), @splat(0)),
        @as(@TypeOf(a), @splat(0)),
    );
}

/// Kogge-Stone Vector Addition/Subtraction
/// Source: http://www.numberworld.org/y-cruncher/internals/addition.html
fn sub(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return switch (@TypeOf(a)) {
        u32, u64 => a -% b,

        // movi    v3.2d, #0000000000000000
        // ...
        // sub     v2.2d, v0.2d, v1.2d
        // cmhi    v0.2d, v1.2d, v0.2d
        // zip1    v0.2d, v3.2d, v0.2d
        // add     v0.2d, v2.2d, v0.2d
        @Vector(2, u64) => (a -% b) +% @Vector(2, u64){ 0, cmhi(a, b)[0] },

        // vpcmpltuq    k0, zmm0, zmm1
        // vpcmpeqq     k1, zmm0, zmm1
        // vpsubq       zmm0, zmm0, zmm1
        // kaddb        k0, k0, k0
        // kaddb        k0, k0, k1
        // kxorb        k0, k0, k1
        // vpmovm2q     zmm1, k0
        // vpaddq       zmm0, zmm0, zmm1

        // (The last two instructions could be fused into a masked-add.)
        // Since we probably need to keep a `0xffffffffffffffff` vector lying around anyway, this can save us an instruction by consuming the mask directly.
        @Vector(8, u64) => blk: {
            const c: u8 = @as(u8, @bitCast(a < b));
            // assert(isZero(a & b));
            const m: u8 = @as(u8, @bitCast(a == b));
            const s = a -% b;

            // const ans = vpaddq_masked_asm(m ^ ((c << 1) +% m), s, @splat(0xffffffffffffffff));
            const ans = @select(u64, @as(@Vector(8, bool), @bitCast(m ^ ((c << 1) +% m))), s +% @as(@TypeOf(a), @splat(0xffffffffffffffff)), s);
            assert(@as(std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(a))), @bitCast(ans)) == @as(std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(a))), @bitCast(a)) -% @as(std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(a))), @bitCast(b)));
            break :blk ans;
        },
        else => @compileError("Sub not implemented for type"),
    };
}

fn disjoint_or(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    assert(isZero(a & b));
    switch (@TypeOf(a)) {
        u32, u64 => return a +% b,
        @Vector(8, u64) => return a | b,
        else => @compileError("Not implemented"),
    }
}

fn shld(a: anytype, comptime i: std.math.Log2Int(std.meta.Int(.signed, @bitSizeOf(@TypeOf(a)))), b: @TypeOf(a)) @TypeOf(a) {
    switch (@typeInfo(@TypeOf(a))) {
        .int => return (a << i) | (b >> (@bitSizeOf(@TypeOf(a)) - i)),
        else => {
            const y = std.simd.mergeShift(b, a, 8 - (i / 64));
            if (i % 64 == 0) return y;
            const x = std.simd.mergeShift(b, a, 8 - (1 + i / 64));

            return (x >> @splat(64 - (i % 64))) | (y << @splat(i % 64));
        },
    }
}

fn shiftInPrevious(comptime N: comptime_int, cur: anytype, prev: @TypeOf(cur)) @TypeOf(cur) {
    @setEvalBranchQuota(10000000);
    comptime assert(0 < @as(u4, N) and N < @sizeOf(@TypeOf(cur)));
    // if (!@inComptime() and comptime (!HAS_ARM_NEON and builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bw))) {
    //     // Workaround for https://github.com/llvm/llvm-project/issues/79799
    //     const c: @TypeOf(cur) = @bitCast(@shuffle(u64, @as(@Vector(@sizeOf(@TypeOf(cur)) / 8, u64), @bitCast(cur)), @as(@Vector(@sizeOf(@TypeOf(cur)) / 8, u64), @bitCast(prev)), std.simd.mergeShift(~std.simd.iota(i4, 8), std.simd.iota(i4, 8), 6)));
    //     return asm (
    //         \\vpalignr %[imm], %[c], %[cur], %[out]
    //         : [out] "=v" (-> @TypeOf(cur)),
    //         : [cur] "v" (cur),
    //           [c] "v" (c),
    //           [imm] "n" (16 - @as(u8, N)),
    //     );
    // } else {
    const a_shift = @as(comptime_int, N) * 8;
    const b_shift = (@sizeOf(@TypeOf(cur)) - N) * 8;

    const USE_SWAR = false;

    return if (USE_SWAR)
        switch (comptime builtin.cpu.arch.endian()) {
            .little => (cur << a_shift) | (prev >> b_shift),
            .big => (cur >> a_shift) | (prev << b_shift),
        }
    else
        std.simd.mergeShift(prev, cur, @sizeOf(@TypeOf(cur)) - N);
    // }
}

fn isZero(x: anytype) bool {
    switch (@TypeOf(x)) {
        @Vector(8, u64) => return 0 == ~@as(u8, @bitCast(x == @as(@TypeOf(x), @splat(0)))),
        else => return @as(std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(x))), @bitCast(x)) == 0,
    }
}

const simd = struct {
    pub fn andn(src: anytype, mask: @TypeOf(src)) @TypeOf(src) {
        // Only force `andn` when operating on GPR's

        if (!@inComptime() and switch (@TypeOf(src)) {
            u64, u32, usize => true,
            else => false,
        }) {
            switch (builtin.cpu.arch) {
                .x86_64 => if (TARGET_HAS_BMI2) {
                    return asm ("andn %[src], %[mask], %[ret]"
                        : [ret] "=r" (-> @TypeOf(src)),
                        : [src] "r" (src),
                          [mask] "r" (mask),
                        : "cc"); // PORT: Zig 0.14 clobber syntax
                },

                .aarch64, .aarch64_be => {
                    return asm ("bic %[ret], %[src], %[mask]"
                        : [ret] "=r" (-> @TypeOf(src)),
                        : [src] "r" (src),
                          [mask] "r" (mask),
                    );
                },

                else => {},
            }
        }

        if (!@inComptime() and @typeInfo(@TypeOf(src)) == .vector) {
            switch (builtin.cpu.arch) {
                // PORT: Switches on the bit size, because Zig 0.14 fails to match `switch (@sizeOf(T))` prongs for vector types.
                .x86_64 => if (comptime (switch (@bitSizeOf(@TypeOf(src))) {
                    512 => TARGET_HAS_AVX_512VL,
                    64, 128, 256 => TARGET_HAS_AVX,
                    else => |s| @compileError(std.fmt.comptimePrint("Invalid vector width passed to `andn`: {} bits", .{s})),
                })) {
                    if (@sizeOf(@TypeOf(src)) == 8) {
                        const VLEN = @typeInfo(@TypeOf(src)).vector.len;
                        const E = @typeInfo(@TypeOf(src)).vector.child;
                        return std.simd.extract(andn(
                            pad_with_undefineds(@Vector(VLEN * 2, E), src),
                            pad_with_undefineds(@Vector(VLEN * 2, E), mask),
                        ), 0, VLEN);
                    }

                    return asm ("vpandn" ++ (if (@sizeOf(@TypeOf(src)) != 64) "" else switch (@bitSizeOf(@typeInfo(@TypeOf(src)).vector.child)) {
                            32 => "d",
                            else => "q", // Matches LLVM behavior. Lane width doesn't really matter for this instruction anyway.
                        }) ++ " %[src], %[mask], %[ret]"
                        : [ret] "=v" (-> @TypeOf(src)),
                        : [src] "v" (src),
                          [mask] "v" (mask),
                    ); // PORT: Zig 0.14 clobber syntax
                },

                else => {},
            }
        }

        return src & ~mask;
    }

    fn expand_compact(
        comptime op: enum { expand, compress },
        comptime T: type,
        fallback: T,
        target: T,
        mask: std.meta.Int(.unsigned, @typeInfo(T).vector.len),
    ) T {
        if (@sizeOf(T) > 64)
            @compileError(std.fmt.comptimePrint("{} is too large for one instruction.", .{T}));

        if (@inComptime() or !TARGET_HAS_AVX_512VBMI2) {
            const vlen = @typeInfo(T).vector.len;
            const E = @typeInfo(T).vector.child;
            var result: [vlen]E = @bitCast(fallback);
            const target_array: [vlen]E = @bitCast(target);
            var i: std.math.Log2Int(@TypeOf(mask)) = 0;
            var j: std.math.Log2Int(@TypeOf(mask)) = 0;

            while (true) {
                if ((mask >> i) & 1 == 1) {
                    switch (op) {
                        .expand => result[i] = target_array[j],
                        .compress => result[j] = target_array[i],
                    }
                    j +%= 1;
                }

                i +%= 1;
                if (i == 0) break;
            }

            return @bitCast(result);
        }

        const vlen = @typeInfo(T).vector.len;
        if (comptime @bitSizeOf(T) < 128) {
            const V2 = @Vector(vlen * 2, @typeInfo(T).vector.child);
            return std.simd.extract(expand_compact(op, V2, pad_with_undefineds(V2, fallback), pad_with_undefineds(V2, target), mask), 0, vlen);
        }

        // I can't get it to work without this... maybe I will switch it to inline assembly at some point if I think it's worth it.
        if (comptime vlen < 8) {
            const V2 = @Vector(8, @typeInfo(T).vector.child);
            return std.simd.extract(
                expand_compact(op, V2, pad_with_undefineds(V2, fallback), pad_with_undefineds(V2, target), mask),
                0,
                vlen,
            );
        }

        return Delegator("x86.avx512.mask." ++ @tagName(op) ++ "." ++ switch (@bitSizeOf(child(T))) {
            8 => "b",
            16 => "w",
            32 => "d",
            64 => "q",
            else => @compileError("Invalid compact element size"),
        }, @TypeOf(target))(target, fallback, mask);
    }

    fn expand_compact_v_type(comptime func_name: []const u8, V: type, K: type) type {
        const v_child = switch (@typeInfo(V)) {
            .vector => |v| v.child,
            .array => |a| a.child,
            else => @compileError("Invalid argument passed to " ++ func_name),
        };

        const mask_len = switch (@typeInfo(K)) {
            .vector => |v| v.len,
            .int => |i| i.bits,
            else => @compileError("Invalid argument passed to " ++ func_name),
        };

        if (length(V) != mask_len)
            @compileError("Vector length and mask length mismatch to " ++ func_name);

        return @Vector(mask_len, v_child);
    }

    pub fn compact(
        fallback: anytype,
        compactable: anytype,
        mask: std.meta.Int(.unsigned, @typeInfo(@TypeOf(fallback)).vector.len),
    ) @TypeOf(fallback) {
        return expand_compact(.compress, @TypeOf(fallback), fallback, compactable, mask);
    }

    pub fn compact_z(compactable: anytype, mask: anytype) expand_compact_v_type("compact_z", @TypeOf(compactable), @TypeOf(mask)) {
        return compact(@as(expand_compact_v_type("compact_z", @TypeOf(compactable), @TypeOf(mask)), @splat(0)), compactable, @bitCast(mask));
    }

    pub fn expand_load(fallback: anytype, source: anytype, mask: anytype) @TypeOf(fallback) {
        const child_t = child(@TypeOf(fallback));
        comptime if (child_t != child(@TypeOf(source)))
            @compileError(std.fmt.comptimePrint("Child types should match, got {} vs {}", .{ @TypeOf(fallback), @TypeOf(source) }));

        const vlen = length(@TypeOf(fallback));
        const uint = std.meta.Int(.unsigned, vlen);
        if (@inComptime() or !TARGET_HAS_AVX_512F) {
            var cur = source;
            var msk: uint = @bitCast(mask);
            var arr: [vlen]child_t = fallback;

            while (msk != 0) : (msk &= msk -% 1) {
                arr[@ctz(msk)] = cur[0];
                cur = cur[1..];
            }
            return arr;
        } else return Delegator("masked.expandload", @TypeOf(fallback))(source.ptr, @bitCast(mask), fallback);
    }

    /// Compacts vec according to mask, and writes it into dest.
    pub fn compact_store(dest: anytype, vec: anytype, mask: anytype) void {
        switch (@typeInfo(@TypeOf(dest))) {
            .pointer => |p| {
                if (p.is_const)
                    @compileError(std.fmt.comptimePrint("Invalid `dest` type passed to compact_store: {} (cannot be const)", .{@TypeOf(dest)}));
                if (p.size != .slice)
                    @compileError(std.fmt.comptimePrint("Invalid `dest` type passed to compact_store: {}", .{@TypeOf(dest)}));
            },
            else => @compileError(std.fmt.comptimePrint("Invalid `dest` type passed to compact_store: {}", .{@TypeOf(dest)})),
        }

        const child_t = child(@TypeOf(vec));
        comptime if (child_t != child(@TypeOf(dest)))
            @compileError(std.fmt.comptimePrint("Child types should match, got {} vs {}", .{ @TypeOf(vec), @TypeOf(dest) }));

        const vlen = length(@TypeOf(vec));
        const uinty = std.meta.Int(.unsigned, vlen);

        if (@inComptime() or !TARGET_HAS_AVX_512F) {
            var cur = dest;
            var msk: uinty = @bitCast(mask);
            const arr: [vlen]child_t = vec;

            while (msk != 0) : (msk &= msk -% 1) {
                cur[0] = arr[@ctz(msk)];
                cur = cur[1..];
            }
        } else if (comptime std.mem.eql(u8, builtin.cpu.model.name, "znver4")) {
            // Zen 4 has a performance hazard on compress stores.
            // Do the compression separately from the store.
            const adv_amt = @popCount(@as(uinty, @bitCast(mask)));
            return masked_store(dest, bzhi(~@as(u64, 0), adv_amt), compact_z(vec, mask));
        } else {
            Delegator("masked.compressstore", @TypeOf(vec))(vec, dest.ptr, @as(@Vector(vlen, bool), @bitCast(mask)));
        }
    }

    pub fn expand(
        fallback: anytype,
        expandable: anytype,
        mask: std.meta.Int(.unsigned, @typeInfo(@TypeOf(fallback)).vector.len),
    ) @TypeOf(fallback) {
        return expand_compact(.expand, @TypeOf(fallback), fallback, expandable, mask);
    }

    pub fn expand_z(expandable: anytype, mask: anytype) expand_compact_v_type("expand_z", @TypeOf(expandable), @TypeOf(mask), u8) {
        return expand(@as(expand_compact_v_type("expand_z", @TypeOf(expandable), @TypeOf(mask)), @splat(0)), expandable, @bitCast(mask));
    }

    pub fn pdep_comptime_mask_assert_no_loss(a: anytype, comptime b: anytype) @TypeOf(b) {
        return pdep(@as(std.meta.Int(.unsigned, @popCount(b)), @intCast(a)), b);
    }

    pub fn pdep(a: anytype, b: anytype) if (@bitSizeOf(@TypeOf(a)) >= @bitSizeOf(@TypeOf(b))) @TypeOf(a) else @TypeOf(b) {
        const T = if (@bitSizeOf(@TypeOf(a)) >= @bitSizeOf(@TypeOf(b))) @TypeOf(a) else @TypeOf(b);

        if (@inComptime() or !TARGET_HAS_BMI2) {
            var src: @TypeOf(a, b) = a;
            var mask: @TypeOf(a, b) = b;
            var result: T = 0;

            while (true) {
                // 1. isolate the lowest set bit of mask
                const lowest: T = (-%mask & mask);

                if (lowest == 0) break;

                // 2. populate LSB from src
                const LSB: T = @bitCast(@as(std.meta.Int(.signed, @bitSizeOf(T)), @bitCast(src << (@bitSizeOf(T) - 1))) >> (@bitSizeOf(T) - 1));

                // 3. copy bit from mask
                result |= LSB & lowest;

                // 4. clear lowest bit
                mask &= ~lowest;

                // 5. prepare for next iteration
                src >>= 1;
            }

            return result;
        }

        if (@bitSizeOf(T) > 64)
            @compileError(std.fmt.comptimePrint("Cannot pdep a type larger than 64 bits. Got: {}", .{T}));

        const methods = struct {
            extern fn @"llvm.x86.bmi.pdep.32"(u32, u32) u32;
            extern fn @"llvm.x86.bmi.pdep.64"(u64, u64) u64;
            extern fn @"llvm.ppc.pdepd"(u64, u64) u64;
        };

        return switch (builtin.cpu.arch) {
            .powerpc64, .powerpc64le => methods.@"llvm.ppc.pdepd"(a, b),
            .x86, .x86_64 => switch (T) {
                u32 => methods.@"llvm.x86.bmi.pdep.32"(a, b),
                else => methods.@"llvm.x86.bmi.pdep.64"(a, b),
            },
            else => unreachable,
        };
    }

    pub fn pext_comptime_mask(a: anytype, comptime b: anytype) std.meta.Int(.unsigned, @popCount(b)) {
        return @intCast(pext(a, b));
    }

    pub fn pext(a: anytype, b: anytype) if (@bitSizeOf(@TypeOf(a)) >= @bitSizeOf(@TypeOf(b))) @TypeOf(a) else @TypeOf(b) {
        const T = if (@bitSizeOf(@TypeOf(a)) >= @bitSizeOf(@TypeOf(b))) @TypeOf(a) else @TypeOf(b);

        if (@bitSizeOf(T) > 64)
            @compileError(std.fmt.comptimePrint("Cannot pext a type larger than 64 bits. Got: {}", .{T}));

        if (@inComptime() or !HAS_FAST_PDEP_AND_PEXT) {
            // Implementation taken from https://www.chessprogramming.org/BMI2#PEXT
            // There are better polyfills out there (especially https://github.com/zwegner/zp7), especially with comptime-known masks,
            // but hopefully this is only used for comptime/dev-time testing.

            const val = a;
            var mask = b;

            var res: u64 = 0;
            var bb: u64 = 1;

            while (mask != 0) : (bb += bb) {
                if ((val & mask & -%mask) != 0)
                    res |= bb;
                mask &= mask -% 1;
            }

            return res;
        }

        const methods = struct {
            extern fn @"llvm.x86.bmi.pext.32"(u32, u32) u32;
            extern fn @"llvm.x86.bmi.pext.64"(u64, u64) u64;
            extern fn @"llvm.ppc.pextd"(u64, u64) u64;
        };

        return switch (builtin.cpu.arch) {
            .powerpc64, .powerpc64le => methods.@"llvm.ppc.pextd"(a, b),
            .x86, .x86_64 => switch (T) {
                u32 => methods.@"llvm.x86.bmi.pext.32"(a, b),
                else => methods.@"llvm.x86.bmi.pext.64"(a, b),
            },
            // TODO: SVE supports pext too
            else => @compileError("Missing pext intrinsic for platform"),
        };
    }

    // From a mask, produces a vector that tells us which indices correspond to set bits in the mask.
    // Above the real data is "don't care" data. Not undefined per se, we still want it to be within 0-63,
    // because we have a piece of code that does `cur.ptr+indices` that might operate on "don't care" data.
    // We could guarantee it to work for ANY byte, by just padding the source file more, but for now we operate
    // under the constraints of the current implementation-- that this returns values in the range of 0-63.
    fn bitsToIndices(mask: anytype) @Vector(@bitSizeOf(@TypeOf(mask)), u8) {
        if (!@inComptime() and comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi2)) {
            return struct {
                extern fn @"llvm.x86.avx512.mask.compress.b"(@Vector(64, u8), @Vector(64, u8), u64) @Vector(64, u8);
            }.@"llvm.x86.avx512.mask.compress.b"(std.simd.iota(u8, 64) + @as(@Vector(64, u8), @splat(0)), @splat(0), mask);
            // Alternatively:
            // }.@"llvm.x86.avx512.mask.compress.b"(std.simd.iota(u8, 64) + @as(@Vector(64, u8), @splat(start_index)), @splat(0), mask);
        } else if (HAS_FAST_PDEP_AND_PEXT) {
            const iota: u64 = @bitCast(std.simd.iota(u4, 16));
            var buffer = [_]u8{0} ** 64;

            inline for (0..4) |i|
                buffer[@popCount(@as(std.meta.Int(.unsigned, i * 16), @truncate(mask)))..][0..16].* =
                    expand8xu8To16xu4AsByteVector(@bitCast(pext(iota, pdep(mask >> (i * 16), 0x1111111111111111) * 0xF))) + @as(@Vector(16, u8), @splat(i * 16));

            return @bitCast(buffer);
        }
    }

    // Workaround until https://github.com/llvm/llvm-project/issues/79094 is solved.
    fn expand8xu8To16xu4AsByteVector(vec: @Vector(8, u8)) @Vector(16, u8) {
        return switch (comptime builtin.cpu.arch.endian()) {
            .little => switch (builtin.cpu.arch) {
                // Doesn't have shifts that operate at byte granularity.
                // To do a shift with byte-granularity, the compiler must insert an `&` operation.
                // Therefore, it's better to do a single `&` after interlacing, and get a 2-for-1.
                // We need to have all these bitCasts because of https://github.com/llvm/llvm-project/issues/89600
                .x86_64 => std.simd.interlace([2]@Vector(8, u8){ vec, @bitCast(@as(@Vector(4, u16), @bitCast(vec)) >> @splat(4)) }) & @as(@Vector(16, u8), @splat(0xF)),

                else => std.simd.interlace(.{ vec & @as(@Vector(8, u8), @splat(0xF)), vec >> @splat(4) }),
            },
            .big => std.simd.interlace(.{ vec >> @splat(4), vec & @as(@Vector(8, u8), @splat(0xF)) }),
        };
    }

    fn vpshufb(table: anytype, indices: anytype) if (@sizeOf(@TypeOf(indices)) > @sizeOf(@TypeOf(table))) @TypeOf(indices) else @TypeOf(table) {
        if (@inComptime()) {
            var result: @TypeOf(indices) = undefined;
            for (0..@bitSizeOf(@TypeOf(indices)) / 8) |i| {
                const index = indices[i];
                result[i] = if (index >= 0x80) 0 else table[index % (@bitSizeOf(@TypeOf(table)) / 8)];
            }

            return result;
        }

        if (@sizeOf(@TypeOf(indices)) > @sizeOf(@TypeOf(table))) {
            return vpshufb(std.simd.repeat(@sizeOf(@TypeOf(indices)), table), indices);
        }

        const HAS_ARM_NEON = switch (builtin.cpu.arch) {
            .aarch64, .aarch64_be => std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon),
            .arm, .armeb => std.Target.arm.featureSetHas(builtin.cpu.features, .neon),
            else => false,
        };

        const HAS_ARM32_FAST_16_BYTE_VECTORS = switch (builtin.cpu.arch) {
            .arm, .armeb => HAS_ARM_NEON and
                (!std.mem.startsWith(u8, builtin.cpu.model.name, "cortex_a") or
                    builtin.cpu.model.name.len != 9 or
                    std.mem.indexOfAnyPos(u8, builtin.cpu.model.name, 8, "5789") == null),
            else => false,
        };
        const IS_VECTORIZER_BROKEN = builtin.cpu.arch.isSPARC() or builtin.cpu.arch.isPowerPC();
        const SUGGESTED_VEC_SIZE: ?comptime_int = if (IS_VECTORIZER_BROKEN)
            null
        else if (HAS_ARM_NEON)
            if (HAS_ARM32_FAST_16_BYTE_VECTORS) 16 else @sizeOf(usize) * 2
        else
            std.simd.suggestVectorLengthForCpu(u8, builtin.cpu);

        if (@sizeOf(@TypeOf(table)) > SUGGESTED_VEC_SIZE.?) {
            var parts: [@sizeOf(@TypeOf(table)) / SUGGESTED_VEC_SIZE.?]@Vector(SUGGESTED_VEC_SIZE.?, u8) = undefined;
            inline for (&parts, 0..) |*slot, i| {
                slot.* = vpshufb(std.simd.extract(table, i * SUGGESTED_VEC_SIZE.?, SUGGESTED_VEC_SIZE.?), std.simd.extract(indices, i * SUGGESTED_VEC_SIZE.?, SUGGESTED_VEC_SIZE.?));
            }
            return @bitCast(parts);
        }

        const methods = struct {
            extern fn @"llvm.x86.avx512.pshuf.b.512"(@Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
            extern fn @"llvm.x86.avx2.pshuf.b"(@Vector(32, u8), @Vector(32, u8)) @Vector(32, u8);
            extern fn @"llvm.x86.ssse3.pshuf.b.128"(@Vector(16, u8), @Vector(16, u8)) @Vector(16, u8);
        };

        return switch (@TypeOf(table)) {
            @Vector(64, u8) => if (comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bw)) methods.@"llvm.x86.avx512.pshuf.b.512"(table, indices) else @compileError("CPU target lacks support for vpshufb512"),
            @Vector(32, u8) => if (comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) methods.@"llvm.x86.avx2.pshuf.b"(table, indices) else @compileError("CPU target lacks support for vpshufb256"),
            @Vector(16, u8) => if (comptime std.Target.x86.featureSetHas(builtin.cpu.features, .ssse3)) methods.@"llvm.x86.ssse3.pshuf.b.128"(table, indices) else @compileError("CPU target lacks support for vpshufb128"),
            else => @compileError(std.fmt.comptimePrint("Invalid argument type passed to vpshufb: {}\n", .{@TypeOf(table)})),
        };
    }

    fn shuffle(table: anytype, indices: anytype) @Vector(@typeInfo(@TypeOf(indices)).vector.len, std.meta.Child(@TypeOf(table))) {
        const T = @Vector(switch (@typeInfo(@TypeOf(table))) {
            .array => |info| info.len,
            .vector => |info| info.len,
            else => @compileError("Invalid type passed to shuffle function."),
        }, std.meta.Child(@TypeOf(table)));
        const child_type = std.meta.Child(@TypeOf(table));
        switch (builtin.cpu.arch) {
            .x86_64 => {
                const UNIQUE_INDICES_AT_ONCE = 16;
                const VEC_LEN = std.simd.suggestVectorLengthForCpu(child_type, builtin.cpu).?;
                const I_LEN = @typeInfo(@TypeOf(indices)).vector.len;

                if (I_LEN < VEC_LEN)
                    @compileError("Come implement this please");
                //                return vpshufb(@shuffle(child_type, table, undefined, std.simd.repeat(@sizeOf(@TypeOf(indices)), std.simd.iota(i32, @typeInfo(T).vector.len))), indices);

                // Break the indices into chunks
                var parts: [@divExact(I_LEN, VEC_LEN)]@Vector(VEC_LEN, @typeInfo(T).vector.child) = undefined;
                inline for (&parts, 0..) |*slot, i| {
                    const sub_indices = std.simd.extract(indices, i * VEC_LEN, VEC_LEN);
                    const Q = @TypeOf(slot.*);
                    var result: Q = @splat(0);

                    // Break the table into 16-byte chunks
                    inline for (0..@typeInfo(T).vector.len / UNIQUE_INDICES_AT_ONCE) |k| {
                        result |= vpshufb(std.simd.repeat(VEC_LEN, std.simd.extract(table, k * UNIQUE_INDICES_AT_ONCE, UNIQUE_INDICES_AT_ONCE)), @as(Q, @splat(0x70)) +| (sub_indices -% @as(Q, @splat(16 * k))));
                    }

                    slot.* = result;
                }
                return @bitCast(parts);
            },
            else => @compileError(std.fmt.comptimePrint("{s} {s} does not have a shuffle implementation yet!\n", .{ builtin.cpu.arch.genericName(), builtin.cpu.model.name })),
            // .arm, .armeb => {},
            // .aarch64, .aarch64_32, .aarch64_be => {},
        }
    }

    fn vperm2(table: @Vector(128, u8), indices: anytype) @TypeOf(indices) {
        if (@inComptime() or comptime !std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi))
            return shuffle(table, indices & @as(@TypeOf(indices), @splat(0b0111_1111)));

        const table_part_1, const table_part_2 = @as([2]@Vector(64, u8), @bitCast(table));
        return std.simd.extract(struct {
            extern fn @"llvm.x86.avx512.vpermi2var.qi.512"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
        }.@"llvm.x86.avx512.vpermi2var.qi.512"(table_part_1, pad_with_undefineds(@Vector(64, u8), indices), table_part_2), 0, @typeInfo(@TypeOf(indices)).vector.len);
    }

    pub fn perm(table: anytype, indices_: anytype) @TypeOf(indices_) {
        switch (@typeInfo(@TypeOf(table))) {
            .array => |a| {
                return perm(@as(@Vector(a.len, a.child), table), indices_);
            },
            .vector => {},
            else => @compileError("Invalid type passed to perm"),
        }

        switch (@typeInfo(@TypeOf(indices_))) {
            .array => |a| {
                return perm(table, @as(@Vector(a.len, a.child), indices_));
            },
            .vector => {},
            else => @compileError("Invalid type passed to perm"),
        }

        comptime assert(@typeInfo(@TypeOf(table)).vector.len > 1 and std.math.isPowerOfTwo(@typeInfo(@TypeOf(table)).vector.len));
        const indices = pad_with_undefineds(@Vector(
            std.math.ceilPowerOfTwo(u16, @typeInfo(@TypeOf(indices_)).vector.len) catch unreachable,
            @typeInfo(@TypeOf(indices_)).vector.child,
        ), indices_);

        if (@inComptime() or !TARGET_HAS_AVX_512VBMI) {
            const ILEN = @typeInfo(@TypeOf(indices)).vector.len;
            const I = @typeInfo(@TypeOf(indices)).vector.child;
            const TLEN = @typeInfo(@TypeOf(table)).vector.len;
            const T = @typeInfo(@TypeOf(table)).vector.child;
            const table_array: [TLEN]T = @bitCast(table);
            const indices_array: [ILEN]I = @bitCast(indices);
            var result: [ILEN]I = undefined;
            for (0..ILEN) |i|
                result[i] = table_array[@intCast(indices_array[i] & (comptime TLEN - 1))];
            return @bitCast(result);
        }

        if (@sizeOf(@TypeOf(indices)) > 64) {
            const I_LEN = @typeInfo(@TypeOf(indices)).vector.len;
            const VEC_LEN = @divExact(512, @bitSizeOf(@typeInfo(@TypeOf(indices)).vector.child));
            var parts: [@divExact(I_LEN, VEC_LEN)]@Vector(VEC_LEN, @typeInfo(@TypeOf(indices)).vector.child) = undefined;
            inline for (&parts, 0..) |*slot, i| {
                const sub_indices = std.simd.extract(indices, i * VEC_LEN, VEC_LEN);
                slot.* = perm(table, sub_indices);
            }
            return @bitCast(parts);
        }

        if (comptime @bitSizeOf(@TypeOf(table)) <= 64)
            return perm(pad_with_undefineds(@Vector(@typeInfo(@TypeOf(table)).vector.len * 2, @typeInfo(@TypeOf(table)).vector.child), table), indices & @as(@TypeOf(indices), @splat(@typeInfo(@TypeOf(table)).vector.len - 1)));

        if (comptime @bitSizeOf(@TypeOf(table)) == 128 and switch (@bitSizeOf(@typeInfo(@TypeOf(table)).vector.child)) {
            32, 64 => true,
            else => false,
        }) return perm(pad_with_undefineds(@Vector(@typeInfo(@TypeOf(table)).vector.len * 2, @typeInfo(@TypeOf(table)).vector.child), table), indices & @as(@TypeOf(indices), @splat(@typeInfo(@TypeOf(table)).vector.len - 1)));

        comptime var field_name: []const u8 = "avx512.permvar." ++ switch (@bitSizeOf(@typeInfo(@TypeOf(table)).vector.child)) {
            8 => "q",
            16 => "h",
            32 => "s",
            64 => "d",
            else => @compileError("Invalid child size in perm"),
        } ++ "i." ++ switch (@bitSizeOf(@TypeOf(table))) {
            128 => "128",
            256 => "256",
            512 => "512",
            else => @compileError("Invalid vector size in perm"),
        };

        field_name = if (comptime std.mem.eql(u8, field_name, "avx512.permvar.si.256"))
            "avx2.permd"
        else
            field_name;

        return std.simd.extract(@field(struct {
            extern fn @"llvm.x86.avx512.permvar.qi.512"(@Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
            extern fn @"llvm.x86.avx512.permvar.hi.512"(@Vector(32, u16), @Vector(32, u16)) @Vector(32, u16);
            extern fn @"llvm.x86.avx512.permvar.si.512"(@Vector(16, u32), @Vector(16, u32)) @Vector(16, u32);
            extern fn @"llvm.x86.avx512.permvar.di.512"(@Vector(8, u64), @Vector(8, u64)) @Vector(8, u64);

            extern fn @"llvm.x86.avx512.permvar.qi.256"(@Vector(32, u8), @Vector(32, u8)) @Vector(32, u8);
            extern fn @"llvm.x86.avx512.permvar.hi.256"(@Vector(16, u16), @Vector(16, u16)) @Vector(16, u16);
            extern fn @"llvm.x86.avx2.permd"(@Vector(8, u32), @Vector(8, u32)) @Vector(8, u32);
            extern fn @"llvm.x86.avx512.permvar.di.256"(@Vector(4, u64), @Vector(4, u64)) @Vector(4, u64);

            extern fn @"llvm.x86.avx512.permvar.qi.128"(@Vector(16, u8), @Vector(16, u8)) @Vector(16, u8);
            extern fn @"llvm.x86.avx512.permvar.hi.128"(@Vector(8, u16), @Vector(8, u16)) @Vector(8, u16);
            // extern fn @"llvm.x86.avx512.permvar.si.128"(@Vector(4, u32), @Vector(4, u32)) @Vector(4, u32);
            // extern fn @"llvm.x86.avx512.permvar.di.128"(@Vector(2, u64), @Vector(2, u64)) @Vector(2, u64);

            //extern fn @"llvm.x86."(@Vector(2, u64), @Vector(2, u64)) @Vector(2, u64);

            // table_1, indicies, table_2
            // extern fn @"llvm.x86.avx512.vpermi2var.qi.512"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
            // extern fn @"llvm.x86.avx512.vpermi2var.hi.512"(@Vector(32, u16), @Vector(32, u16), @Vector(32, u16)) @Vector(32, u16);
            // extern fn @"llvm.x86.avx512.vpermi2var.si.512"(@Vector(16, u32), @Vector(16, u32), @Vector(16, u32)) @Vector(16, u32);
            // extern fn @"llvm.x86.avx512.vpermi2var.di.512"(@Vector(8, u64), @Vector(8, u64), @Vector(8, u64)) @Vector(8, u64);
            // extern fn @"llvm.x86.avx512.vpermi2var.qi.256"(@Vector(32, u8), @Vector(32, u8), @Vector(32, u8)) @Vector(32, u8);
            // extern fn @"llvm.x86.avx512.vpermi2var.hi.256"(@Vector(16, u16), @Vector(16, u16), @Vector(16, u16)) @Vector(16, u16);
            // extern fn @"llvm.x86.avx512.vpermi2var.si.256"(@Vector(8, u32), @Vector(8, u32), @Vector(8, u32)) @Vector(8, u32);
            // extern fn @"llvm.x86.avx512.vpermi2var.di.256"(@Vector(4, u64), @Vector(4, u64), @Vector(4, u64)) @Vector(4, u64);
            // extern fn @"llvm.x86.avx512.vpermi2var.qi.128"(@Vector(16, u8), @Vector(16, u8), @Vector(16, u8)) @Vector(16, u8);
            // extern fn @"llvm.x86.avx512.vpermi2var.hi.128"(@Vector(8, u16), @Vector(8, u16), @Vector(8, u16)) @Vector(8, u16);
            // extern fn @"llvm.x86.avx512.vpermi2var.si.128"(@Vector(4, u32), @Vector(4, u32), @Vector(4, u32)) @Vector(4, u32);
            // extern fn @"llvm.x86.avx512.vpermi2var.di.128"(@Vector(2, u64), @Vector(2, u64), @Vector(2, u64)) @Vector(2, u64);
        }, "llvm.x86." ++ field_name)(table, pad_with_undefineds(@TypeOf(table), indices)), 0, @typeInfo(@TypeOf(indices)).vector.len);
    }

    pub fn perm4b(table: [4]@Vector(64, u8), indices: @Vector(64, u8)) @Vector(64, u8) {
        return @select(
            u8,
            indices < @as(@Vector(64, u8), @splat(0x80)),
            perm2b(@as(@Vector(128, u8), @bitCast(table[0..][0..2].*)), indices),
            perm2b(@as(@Vector(128, u8), @bitCast(table[2..][0..2].*)), indices),
        );
    }

    pub fn perm2b(table: anytype, indices_: anytype) @TypeOf(indices_) {
        switch (@typeInfo(@TypeOf(table))) {
            .array => |a| {
                return perm2b(@as(@Vector(a.len, a.child), table), indices_);
            },
            .vector => {},
            else => @compileError("Invalid type passed to perm2"),
        }

        switch (@typeInfo(@TypeOf(indices_))) {
            .array => |a| {
                return perm2b(table, @as(@Vector(a.len, a.child), indices_));
            },
            .vector => {},
            else => @compileError("Invalid type passed to perm2"),
        }

        comptime assert(@typeInfo(@TypeOf(table)).vector.len > 1 and std.math.isPowerOfTwo(@typeInfo(@TypeOf(table)).vector.len));
        const indices = pad_with_undefineds(@Vector(
            std.math.ceilPowerOfTwo(u16, @typeInfo(@TypeOf(indices_)).vector.len) catch unreachable,
            @typeInfo(@TypeOf(indices_)).vector.child,
        ), indices_);

        if (@inComptime() or !TARGET_HAS_AVX_512VBMI) {
            const ILEN = @typeInfo(@TypeOf(indices)).vector.len;
            const I = @typeInfo(@TypeOf(indices)).vector.child;
            const TLEN = @typeInfo(@TypeOf(table)).vector.len;
            const T = @typeInfo(@TypeOf(table)).vector.child;
            const table_array: [TLEN]T = @bitCast(table);
            const indices_array: [ILEN]I = @bitCast(indices);
            var result: [ILEN]I = undefined;
            for (0..ILEN) |i|
                result[i] = table_array[@intCast(indices_array[i] & (comptime TLEN - 1))];
            return @bitCast(result);
        }

        if (@sizeOf(@TypeOf(indices)) > 64) {
            const I_LEN = @typeInfo(@TypeOf(indices)).vector.len;
            const VEC_LEN = @divExact(512, @bitSizeOf(@typeInfo(@TypeOf(indices)).vector.child));
            var parts: [@divExact(I_LEN, VEC_LEN)]@Vector(VEC_LEN, @typeInfo(@TypeOf(indices)).vector.child) = undefined;
            inline for (&parts, 0..) |*slot, i| {
                const sub_indices = std.simd.extract(indices, i * VEC_LEN, VEC_LEN);
                slot.* = perm2b(table, sub_indices);
            }
            return @bitCast(parts);
        }

        const H = @Vector(@typeInfo(@TypeOf(table)).vector.len / 2, @typeInfo(@TypeOf(table)).vector.child);

        const field_name: []const u8 = "avx512.vpermi2var." ++ switch (@bitSizeOf(@typeInfo(@TypeOf(table)).vector.child)) {
            8 => "q",
            16 => "h",
            32 => "s",
            64 => "d",
            else => @compileError("Invalid child size in perm2"),
        } ++ "i." ++ switch (@bitSizeOf(H)) {
            128 => "128",
            256 => "256",
            512 => "512",
            else => @compileError("Invalid vector size in perm2"),
        };

        const table_1, const table_2 = @as([2]H, @bitCast(table));

        return std.simd.extract(@field(struct {
            // extern fn @"llvm.x86.avx512.permvar.qi.512"(@Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
            // extern fn @"llvm.x86.avx512.permvar.hi.512"(@Vector(32, u16), @Vector(32, u16)) @Vector(32, u16);
            // extern fn @"llvm.x86.avx512.permvar.si.512"(@Vector(16, u32), @Vector(16, u32)) @Vector(16, u32);
            // extern fn @"llvm.x86.avx512.permvar.di.512"(@Vector(8, u64), @Vector(8, u64)) @Vector(8, u64);

            // extern fn @"llvm.x86.avx512.permvar.qi.256"(@Vector(32, u8), @Vector(32, u8)) @Vector(32, u8);
            // extern fn @"llvm.x86.avx512.permvar.hi.256"(@Vector(16, u16), @Vector(16, u16)) @Vector(16, u16);
            // extern fn @"llvm.x86.avx2.permd"(@Vector(8, u32), @Vector(8, u32)) @Vector(8, u32);
            // extern fn @"llvm.x86.avx512.permvar.di.256"(@Vector(4, u64), @Vector(4, u64)) @Vector(4, u64);

            // extern fn @"llvm.x86.avx512.permvar.qi.128"(@Vector(16, u8), @Vector(16, u8)) @Vector(16, u8);
            // extern fn @"llvm.x86.avx512.permvar.hi.128"(@Vector(8, u16), @Vector(8, u16)) @Vector(8, u16);
            // extern fn @"llvm.x86.avx512.permvar.si.128"(@Vector(4, u32), @Vector(4, u32)) @Vector(4, u32);
            // extern fn @"llvm.x86.avx512.permvar.di.128"(@Vector(2, u64), @Vector(2, u64)) @Vector(2, u64);

            //extern fn @"llvm.x86."(@Vector(2, u64), @Vector(2, u64)) @Vector(2, u64);

            // table_1, indicies, table_2
            extern fn @"llvm.x86.avx512.vpermi2var.qi.512"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
            extern fn @"llvm.x86.avx512.vpermi2var.hi.512"(@Vector(32, u16), @Vector(32, u16), @Vector(32, u16)) @Vector(32, u16);
            extern fn @"llvm.x86.avx512.vpermi2var.si.512"(@Vector(16, u32), @Vector(16, u32), @Vector(16, u32)) @Vector(16, u32);
            extern fn @"llvm.x86.avx512.vpermi2var.di.512"(@Vector(8, u64), @Vector(8, u64), @Vector(8, u64)) @Vector(8, u64);
            extern fn @"llvm.x86.avx512.vpermi2var.qi.256"(@Vector(32, u8), @Vector(32, u8), @Vector(32, u8)) @Vector(32, u8);
            extern fn @"llvm.x86.avx512.vpermi2var.hi.256"(@Vector(16, u16), @Vector(16, u16), @Vector(16, u16)) @Vector(16, u16);
            extern fn @"llvm.x86.avx512.vpermi2var.si.256"(@Vector(8, u32), @Vector(8, u32), @Vector(8, u32)) @Vector(8, u32);
            extern fn @"llvm.x86.avx512.vpermi2var.di.256"(@Vector(4, u64), @Vector(4, u64), @Vector(4, u64)) @Vector(4, u64);
            extern fn @"llvm.x86.avx512.vpermi2var.qi.128"(@Vector(16, u8), @Vector(16, u8), @Vector(16, u8)) @Vector(16, u8);
            extern fn @"llvm.x86.avx512.vpermi2var.hi.128"(@Vector(8, u16), @Vector(8, u16), @Vector(8, u16)) @Vector(8, u16);
            extern fn @"llvm.x86.avx512.vpermi2var.si.128"(@Vector(4, u32), @Vector(4, u32), @Vector(4, u32)) @Vector(4, u32);
            extern fn @"llvm.x86.avx512.vpermi2var.di.128"(@Vector(2, u64), @Vector(2, u64), @Vector(2, u64)) @Vector(2, u64);
        }, "llvm.x86." ++ field_name)(table_1, pad_with_undefineds(H, indices), table_2), 0, @typeInfo(@TypeOf(indices)).vector.len);
    }

    // zig fmt: off
pub const TARGET_HAS_AVX_512F          = builtin.target.cpu.arch == .x86_64 and builtin.zig_backend != .stage2_x86_64 and std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx512f);
pub const TARGET_HAS_AVX_512VL         = TARGET_HAS_AVX_512F and std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx512vl);
pub const TARGET_HAS_AVX_512VBMI       = TARGET_HAS_AVX_512F and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi);
pub const TARGET_HAS_AVX_512VBMI2      = TARGET_HAS_AVX_512F and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi2);
pub const TARGET_HAS_AVX_512BW         = TARGET_HAS_AVX_512F and std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx512bw);
pub const TARGET_HAS_AVX_512IFMA       = TARGET_HAS_AVX_512F and std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx512ifma);
pub const TARGET_HAS_AVX_512BITALG     = TARGET_HAS_AVX_512F and std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx512bitalg);
pub const TARGET_HAS_AVX_512VPOPCNTDQ  = TARGET_HAS_AVX_512F and std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx512vpopcntdq);
pub const TARGET_HAS_AVX_512VPCLMULQDQ = TARGET_HAS_AVX_512F and std.Target.x86.featureSetHas(builtin.target.cpu.features, .vpclmulqdq);
pub const TARGET_HAS_AVX_512DQ         = TARGET_HAS_AVX_512F and std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx512dq);
pub const TARGET_HAS_AVX_512VNNI       = builtin.target.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx512vnni);
pub const TARGET_HAS_AVX_VNNI          = builtin.target.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.target.cpu.features, .avxvnni);
pub const TARGET_HAS_SSE3              = builtin.target.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.target.cpu.features, .sse3);
pub const TARGET_HAS_AVX               = builtin.target.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx);
pub const TARGET_HAS_AVX2              = builtin.target.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx2);
pub const TARGET_HAS_BMI2              = builtin.target.cpu.arch == .x86_64 and builtin.zig_backend != .stage2_x86_64 and std.Target.x86.featureSetHas(builtin.target.cpu.features, .bmi2);
pub const TARGET_IS_AMD_ZEN            = std.mem.startsWith(u8, builtin.cpu.model.name, "znver");

const HAS_FAST_PDEP_AND_PEXT = blk: {
    const cpu_name = builtin.cpu.model.llvm_name orelse builtin.cpu.model.name;
    break :blk (builtin.cpu.arch.isPowerPC() and std.Target.powerpc.featureSetHas(builtin.cpu.features, .isa_v31_instructions)) or
        builtin.cpu.arch.isX86() and
            std.Target.x86.featureSetHas(builtin.cpu.features, .bmi2) and
            // pdep is microcoded (slow) on AMD architectures before Zen 3.
            !std.mem.startsWith(u8, cpu_name, "bdver") and
            (!std.mem.startsWith(u8, cpu_name, "znver") or !(cpu_name["znver".len] < '3' and cpu_name["znver".len..].len == 1));
};
// zig fmt: on

    pub fn child(comptime T: type) type {
        switch (@typeInfo(T)) {
            .pointer => |p| {
                switch (@typeInfo(p.child)) {
                    .int => return p.child,
                    .array => |a| return a.child,
                    .vector => |a| return a.child,
                    .@"enum" => |e| return e.tag_type,
                    else => |a| @compileError(std.fmt.comptimePrint("Not yet implemented for {}", .{a})),
                }
            },
            .array => |a| return a.child,
            .vector => |a| return a.child,
            else => |a| @compileError(std.fmt.comptimePrint("Not yet implemented for {}", .{a})),
        }
    }

    pub fn length(comptime T: type) comptime_int {
        switch (@typeInfo(T)) {
            .pointer => |p| {
                switch (@typeInfo(p.child)) {
                    .array => |a| return a.len,
                    .vector => |a| return a.len,
                    else => |a| @compileError(std.fmt.comptimePrint("Not yet implemented for {}", .{a})),
                }
            },
            .array => |a| return a.len,
            .vector => |a| return a.len,
            else => |a| @compileError(std.fmt.comptimePrint("Not yet implemented for {}", .{a})),
        }
    }

    pub fn masked_load(slice_or_ptr: anytype, vec_mask: anytype, padding: child(@TypeOf(slice_or_ptr))) @Vector(switch (@typeInfo(@TypeOf(vec_mask))) {
        .vector => |v| v.len,
        .int => |i| i.bits,
        else => @compileError("Invalid argument to masked_load"),
    }, child(@TypeOf(slice_or_ptr))) {
        const N = switch (@typeInfo(@TypeOf(vec_mask))) {
            .vector => |v| v.len,
            .int => |i| i.bits,
            else => @compileError("Invalid argument to masked_load"),
        };
        const E = child(@TypeOf(slice_or_ptr));
        const ptr = if (@typeInfo(@TypeOf(slice_or_ptr)).pointer.size == .slice)
            slice_or_ptr.ptr
        else
            slice_or_ptr;

        const V = @Vector(N, E);
        _ = V; // autofix

        if (@inComptime() or !TARGET_HAS_AVX_512F) {
            var buf: [N]E = @splat(padding);
            var mask: std.meta.Int(.unsigned, N) = @bitCast(vec_mask);
            while (mask != 0) : (mask &= mask -% 1) {
                const v = ptr[@ctz(mask)];
                switch (@typeInfo(@TypeOf(v))) {
                    .@"enum" => buf[@ctz(mask)] = @intFromEnum(v),
                    else => buf[@ctz(mask)] = v,
                }
            }
            return buf;
        }

        const funcs = struct {
            extern fn @"llvm.masked.load.8i8.p0"(@TypeOf(ptr), i32, @Vector(8, bool), @Vector(8, E)) @Vector(8, E);
            extern fn @"llvm.masked.load.v8i16.p0"(@TypeOf(ptr), i32, @Vector(8, bool), @Vector(8, E)) @Vector(8, E);
            extern fn @"llvm.masked.load.v8i32.p0"(@TypeOf(ptr), i32, @Vector(8, bool), @Vector(8, E)) @Vector(8, E);
            extern fn @"llvm.masked.load.v8i64.p0"(@TypeOf(ptr), i32, @Vector(8, bool), @Vector(8, E)) @Vector(8, E);
            extern fn @"llvm.masked.load.v16i8.p0"(@TypeOf(ptr), i32, @Vector(16, bool), @Vector(16, E)) @Vector(16, E);
            extern fn @"llvm.masked.load.v16i16.p0"(@TypeOf(ptr), i32, @Vector(16, bool), @Vector(16, E)) @Vector(16, E);
            extern fn @"llvm.masked.load.v16i32.p0"(@TypeOf(ptr), i32, @Vector(16, bool), @Vector(16, E)) @Vector(16, E);
            extern fn @"llvm.masked.load.v32i8.p0"(@TypeOf(ptr), i32, @Vector(32, bool), @Vector(32, E)) @Vector(32, E);
            extern fn @"llvm.masked.load.v32i16.p0"(@TypeOf(ptr), i32, @Vector(32, bool), @Vector(32, E)) @Vector(32, E);
            extern fn @"llvm.masked.load.v64i8.p0"(@TypeOf(ptr), i32, @Vector(64, bool), @Vector(64, E)) @Vector(64, E);
        };

        const bool_mask: @Vector(N, bool) = @bitCast(vec_mask);

        switch (N) {
            1...7 => switch (@bitSizeOf(E)) {
                8 => return std.simd.extract(
                    funcs.@"llvm.masked.load.8i8.p0"(ptr, @alignOf(u8), padZeros(@Vector(8, bool), bool_mask), @splat(padding)),
                    0,
                    N,
                ),
                //16 => return funcs.@"llvm.masked.load.v8i16.p0"(ptr, @alignOf(u16), bool_mask, @splat(padding)),
                //32 => return funcs.@"llvm.masked.load.v8i32.p0"(ptr, @alignOf(u32), bool_mask, @splat(padding)),
                //64 => return funcs.@"llvm.masked.load.v8i64.p0"(ptr, @alignOf(u64), bool_mask, @splat(padding)),
                else => {},
            },
            8 => switch (@bitSizeOf(E)) {
                8 => return funcs.@"llvm.masked.load.8i8.p0"(ptr, @alignOf(u8), bool_mask, @splat(padding)),
                16 => return funcs.@"llvm.masked.load.v8i16.p0"(ptr, @alignOf(u16), bool_mask, @splat(padding)),
                32 => return funcs.@"llvm.masked.load.v8i32.p0"(ptr, @alignOf(u32), bool_mask, @splat(padding)),
                64 => return funcs.@"llvm.masked.load.v8i64.p0"(ptr, @alignOf(u64), bool_mask, @splat(padding)),
                else => {},
            },
            16 => switch (@bitSizeOf(E)) {
                8 => return funcs.@"llvm.masked.load.v16i8.p0"(ptr, @alignOf(u8), bool_mask, @splat(padding)),
                16 => return funcs.@"llvm.masked.load.v16i16.p0"(ptr, @alignOf(u16), bool_mask, @splat(padding)),
                32 => return funcs.@"llvm.masked.load.v16i32.p0"(ptr, @alignOf(u32), bool_mask, @splat(padding)),
                else => {},
            },
            32 => switch (@bitSizeOf(E)) {
                8 => return funcs.@"llvm.masked.load.v32i8.p0"(ptr, @alignOf(u8), bool_mask, @splat(padding)),
                16 => return funcs.@"llvm.masked.load.v32i16.p0"(ptr, @alignOf(u16), bool_mask, @splat(padding)),
                else => {},
            },
            64 => switch (@bitSizeOf(E)) {
                8 => return funcs.@"llvm.masked.load.v64i8.p0"(ptr, @alignOf(u8), bool_mask, @splat(padding)),
                else => {},
            },
            else => {},
        }

        const H = @divExact(N, 2);

        return std.simd.join(
            masked_load(slice_or_ptr[0..], std.simd.extract(bool_mask, 0, H), padding),
            masked_load(slice_or_ptr[H..], std.simd.extract(bool_mask, H, H), padding),
        );
    }

    pub fn pad_with_undefineds(T: type, value: anytype) if (@sizeOf(@TypeOf(value)) > @sizeOf(T)) @TypeOf(value) else T {
        const padding_len = @typeInfo(T).vector.len - switch (@typeInfo(@TypeOf(value))) {
            .vector => |v| v.len,
            .array => |a| a.len,
            else => @compileError("Invalid type passed to pad_with_undefineds"),
        };
        return if (padding_len <= 0) value else std.simd.join(value, @as(@Vector(padding_len, @typeInfo(T).vector.child), @splat(undefined)));
    }

    pub fn padZeros(T: type, value: anytype) if (@sizeOf(@TypeOf(value)) > @sizeOf(T)) @TypeOf(value) else T {
        const padding_len = @typeInfo(T).vector.len - switch (@typeInfo(@TypeOf(value))) {
            .vector => |v| v.len,
            .array => |a| a.len,
            else => @compileError("Invalid type passed to pad_with_zeroes"),
        };
        return if (padding_len <= 0) value else std.simd.join(value, @as(@Vector(padding_len, @typeInfo(T).vector.child), @splat(std.mem.zeroes(@typeInfo(T).vector.child))));
    }

    /// Basically equivalent to:
    ///     `@select(u8, std.simd.iota(u64, 64) < @splat(slice.len), slice.ptr[0..64].*, @as(@Vector(64, u8), @truncate(lens)))`
    /// except this is guaranteed by the compiler to use a safe masked load on AVX-512 machines (even in debug modes)
    /// On other arches, we default to memcpy'ing the data to a buffer of 64 bytes and then returning that.
    pub fn safe_load(comptime N_: u16, slice: anytype, padding: @typeInfo(@TypeOf(slice)).pointer.child) @Vector(std.math.ceilPowerOfTwo(u16, N_) catch unreachable, @typeInfo(@TypeOf(slice)).pointer.child) {
        const N = comptime std.math.ceilPowerOfTwo(u16, N_) catch unreachable;
        const E = @typeInfo(@TypeOf(slice)).pointer.child;
        const ptr = slice.ptr;

        if (TARGET_HAS_AVX_512F) {
            const vec_mask = generate_safe_load_mask(N_, slice.len);
            return masked_load(slice, vec_mask, padding);
        } else {
            var buf: [N]E = @splat(padding);
            @memcpy(buf[0..@min(slice.len, N_)], ptr);
            return buf;
        }
    }

    pub fn bzhi(msk: anytype, cnt: @TypeOf(msk)) @TypeOf(msk) {
        comptime std.debug.assert(@TypeOf(msk) == u64 or @TypeOf(msk) == u32);
        if (!@inComptime() and TARGET_HAS_BMI2) {
            const intrinsics = struct {
                extern fn @"llvm.x86.bmi.bzhi.64"(u64, u64) u64;
                extern fn @"llvm.x86.bmi.bzhi.32"(u32, u32) u32;
            };

            return switch (@TypeOf(msk)) {
                u64 => intrinsics.@"llvm.x86.bmi.bzhi.64"(msk, cnt),
                u32 => intrinsics.@"llvm.x86.bmi.bzhi.32"(msk, cnt),
                else => @compileError("Invalid type for msk in bzhi"),
            };
        } else {
            const m: u8 = @truncate(cnt);
            return msk & ~if (m >= @bitSizeOf(@TypeOf(msk))) @as(@TypeOf(msk), 0) else (~@as(@TypeOf(msk), 0) << @intCast(m));
        }
    }

    pub fn generate_safe_load_int_mask(comptime N_: u16, slice_len: usize) if (N_ > 32) u64 else u32 {
        comptime assert(N_ <= 64);
        const uint_ = if (N_ > 32) u64 else u32;
        const full_mask = comptime bzhi(~@as(uint_, 0), N_);
        const unsafe_mask = bzhi(full_mask, @truncate(slice_len));
        const safe_mask = if (slice_len >= N_) full_mask else unsafe_mask;
        return safe_mask;
    }

    pub fn generate_safe_load_mask(comptime N_: u16, slice_len: usize) @Vector(std.math.ceilPowerOfTwo(u16, N_) catch unreachable, bool) {
        comptime assert(N_ <= 64);
        const N = comptime std.math.ceilPowerOfTwo(u16, N_) catch unreachable;
        const bit_width = if (N_ > 32) 64 else 32;
        const full_mask = comptime bzhi(~@as(std.meta.Int(.unsigned, bit_width), 0), N_);
        const unsafe_mask = bzhi(full_mask, @truncate(slice_len));
        const safe_mask = if (slice_len >= N_) full_mask else unsafe_mask;
        return std.simd.extract(@as(@Vector(bit_width, bool), @bitCast(safe_mask)), 0, N);
    }

    pub fn masked_store(slice: anytype, vec_mask: anytype, vec: anytype) void {
        const VLEN = length(@TypeOf(vec));
        const E = child(@TypeOf(slice));

        const mask = std.simd.extract(@as(@Vector(switch (@typeInfo(@TypeOf(vec_mask))) {
            .vector => |v| v.len,
            .int => |i| i.bits,
            else => @compileError("Invalid argument to masked_load"),
        }, bool), @bitCast(vec_mask)), 0, VLEN);

        if (TARGET_HAS_AVX_512F) {
            struct {
                extern fn @"llvm.masked.store"(@TypeOf(vec), @TypeOf(slice.ptr), i32, @Vector(VLEN, bool)) void;
            }.@"llvm.masked.store"(
                vec,
                slice.ptr,
                @alignOf(E),
                mask,
            );
        } else {
            for (slice, 0..) |*slot, i| {
                if (mask[i]) slot.* = vec[i];
            }
        }
    }

    /// Wrapper that allows us to automatically delegate to highly regular intrinsic calls like `x86.avx512.mask.compress.b`
    inline fn Delegator(comptime intrin_name: []const u8, comptime T: type) switch (@typeInfo(T)) {
        .vector => |v| @TypeOf(@field(intrinsic_calls, std.fmt.comptimePrint("llvm.{s}.v{}u{}", .{
            intrin_name,
            v.len,
            @bitSizeOf(v.child),
        }))),
        else => @compileError("Can't get return type yet"),
    } {
        return switch (@typeInfo(T)) {
            .vector => |v| {
                return @field(intrinsic_calls, std.fmt.comptimePrint("llvm.{s}.v{}u{}", .{
                    intrin_name,
                    v.len,
                    @bitSizeOf(v.child),
                }));
            },

            else => @compileError(std.fmt.comptimePrint("Unknown type: {}", .{T})),
        };
    }

    const intrinsic_calls = struct {
        // -------------------------------------------------------------------------------------------------
        extern fn @"llvm.fshl.u8"(u8, u8, u8) u8;
        extern fn @"llvm.fshl.u16"(u16, u16, u16) u16;
        extern fn @"llvm.fshl.u32"(u32, u32, u32) u32;
        extern fn @"llvm.fshl.u64"(u64, u64, u64) u64;
        extern fn @"llvm.fshl.u128"(u128, u128, u128) u128;

        extern fn @"llvm.fshl.v1u8"(@Vector(1, u8), @Vector(1, u8), @Vector(1, u8)) @Vector(1, u8);
        extern fn @"llvm.fshl.v2u8"(@Vector(2, u8), @Vector(2, u8), @Vector(2, u8)) @Vector(2, u8);
        extern fn @"llvm.fshl.v4u8"(@Vector(4, u8), @Vector(4, u8), @Vector(4, u8)) @Vector(4, u8);
        extern fn @"llvm.fshl.v8u8"(@Vector(8, u8), @Vector(8, u8), @Vector(8, u8)) @Vector(8, u8);
        extern fn @"llvm.fshl.v16u8"(@Vector(16, u8), @Vector(16, u8), @Vector(16, u8)) @Vector(16, u8);
        extern fn @"llvm.fshl.v32u8"(@Vector(32, u8), @Vector(32, u8), @Vector(32, u8)) @Vector(32, u8);
        extern fn @"llvm.fshl.v64u8"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);

        extern fn @"llvm.fshl.v1u16"(@Vector(1, u16), @Vector(1, u16), @Vector(1, u16)) @Vector(1, u16);
        extern fn @"llvm.fshl.v2u16"(@Vector(2, u16), @Vector(2, u16), @Vector(2, u16)) @Vector(2, u16);
        extern fn @"llvm.fshl.v4u16"(@Vector(4, u16), @Vector(4, u16), @Vector(4, u16)) @Vector(4, u16);
        extern fn @"llvm.fshl.v8u16"(@Vector(8, u16), @Vector(8, u16), @Vector(8, u16)) @Vector(8, u16);
        extern fn @"llvm.fshl.v16u16"(@Vector(16, u16), @Vector(16, u16), @Vector(16, u16)) @Vector(16, u16);
        extern fn @"llvm.fshl.v32u16"(@Vector(32, u16), @Vector(32, u16), @Vector(32, u16)) @Vector(32, u16);

        extern fn @"llvm.fshl.v1u32"(@Vector(1, u32), @Vector(1, u32), @Vector(1, u32)) @Vector(1, u32);
        extern fn @"llvm.fshl.v2u32"(@Vector(2, u32), @Vector(2, u32), @Vector(2, u32)) @Vector(2, u32);
        extern fn @"llvm.fshl.v4u32"(@Vector(4, u32), @Vector(4, u32), @Vector(4, u32)) @Vector(4, u32);
        extern fn @"llvm.fshl.v8u32"(@Vector(8, u32), @Vector(8, u32), @Vector(8, u32)) @Vector(8, u32);
        extern fn @"llvm.fshl.v16u32"(@Vector(16, u32), @Vector(16, u32), @Vector(16, u32)) @Vector(16, u32);

        extern fn @"llvm.fshl.v1u64"(@Vector(1, u64), @Vector(1, u64), @Vector(1, u64)) @Vector(1, u64);
        extern fn @"llvm.fshl.v2u64"(@Vector(2, u64), @Vector(2, u64), @Vector(2, u64)) @Vector(2, u64);
        extern fn @"llvm.fshl.v4u64"(@Vector(4, u64), @Vector(4, u64), @Vector(4, u64)) @Vector(4, u64);
        extern fn @"llvm.fshl.v8u64"(@Vector(8, u64), @Vector(8, u64), @Vector(8, u64)) @Vector(8, u64);

        // -------------------------------------------------------------------------------------------------

        extern fn @"llvm.fshr.u8"(u8, u8, u8) u8;
        extern fn @"llvm.fshr.u16"(u16, u16, u16) u16;
        extern fn @"llvm.fshr.u32"(u32, u32, u32) u32;
        extern fn @"llvm.fshr.u64"(u64, u64, u64) u64;
        extern fn @"llvm.fshr.u128"(u128, u128, u128) u128;

        extern fn @"llvm.fshr.v1u8"(@Vector(1, u8), @Vector(1, u8), @Vector(1, u8)) @Vector(1, u8);
        extern fn @"llvm.fshr.v2u8"(@Vector(2, u8), @Vector(2, u8), @Vector(2, u8)) @Vector(2, u8);
        extern fn @"llvm.fshr.v4u8"(@Vector(4, u8), @Vector(4, u8), @Vector(4, u8)) @Vector(4, u8);
        extern fn @"llvm.fshr.v8u8"(@Vector(8, u8), @Vector(8, u8), @Vector(8, u8)) @Vector(8, u8);
        extern fn @"llvm.fshr.v16u8"(@Vector(16, u8), @Vector(16, u8), @Vector(16, u8)) @Vector(16, u8);
        extern fn @"llvm.fshr.v32u8"(@Vector(32, u8), @Vector(32, u8), @Vector(32, u8)) @Vector(32, u8);
        extern fn @"llvm.fshr.v64u8"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);

        extern fn @"llvm.fshr.v1u16"(@Vector(1, u16), @Vector(1, u16), @Vector(1, u16)) @Vector(1, u16);
        extern fn @"llvm.fshr.v2u16"(@Vector(2, u16), @Vector(2, u16), @Vector(2, u16)) @Vector(2, u16);
        extern fn @"llvm.fshr.v4u16"(@Vector(4, u16), @Vector(4, u16), @Vector(4, u16)) @Vector(4, u16);
        extern fn @"llvm.fshr.v8u16"(@Vector(8, u16), @Vector(8, u16), @Vector(8, u16)) @Vector(8, u16);
        extern fn @"llvm.fshr.v16u16"(@Vector(16, u16), @Vector(16, u16), @Vector(16, u16)) @Vector(16, u16);
        extern fn @"llvm.fshr.v32u16"(@Vector(32, u16), @Vector(32, u16), @Vector(32, u16)) @Vector(32, u16);

        extern fn @"llvm.fshr.v1u32"(@Vector(1, u32), @Vector(1, u32), @Vector(1, u32)) @Vector(1, u32);
        extern fn @"llvm.fshr.v2u32"(@Vector(2, u32), @Vector(2, u32), @Vector(2, u32)) @Vector(2, u32);
        extern fn @"llvm.fshr.v4u32"(@Vector(4, u32), @Vector(4, u32), @Vector(4, u32)) @Vector(4, u32);
        extern fn @"llvm.fshr.v8u32"(@Vector(8, u32), @Vector(8, u32), @Vector(8, u32)) @Vector(8, u32);
        extern fn @"llvm.fshr.v16u32"(@Vector(16, u32), @Vector(16, u32), @Vector(16, u32)) @Vector(16, u32);

        extern fn @"llvm.fshr.v1u64"(@Vector(1, u64), @Vector(1, u64), @Vector(1, u64)) @Vector(1, u64);
        extern fn @"llvm.fshr.v2u64"(@Vector(2, u64), @Vector(2, u64), @Vector(2, u64)) @Vector(2, u64);
        extern fn @"llvm.fshr.v4u64"(@Vector(4, u64), @Vector(4, u64), @Vector(4, u64)) @Vector(4, u64);
        extern fn @"llvm.fshr.v8u64"(@Vector(8, u64), @Vector(8, u64), @Vector(8, u64)) @Vector(8, u64);

        // -------------------------------------------------------------------------------------------------

        extern fn @"llvm.masked.expandload.v1u8"([*]const u8, @Vector(1, bool), @Vector(1, u8)) @Vector(1, u8);
        extern fn @"llvm.masked.expandload.v2u8"([*]const u8, @Vector(2, bool), @Vector(2, u8)) @Vector(2, u8);
        extern fn @"llvm.masked.expandload.v4u8"([*]const u8, @Vector(4, bool), @Vector(4, u8)) @Vector(4, u8);
        extern fn @"llvm.masked.expandload.v8u8"([*]const u8, @Vector(8, bool), @Vector(8, u8)) @Vector(8, u8);
        extern fn @"llvm.masked.expandload.v16u8"([*]const u8, @Vector(16, bool), @Vector(16, u8)) @Vector(16, u8);
        extern fn @"llvm.masked.expandload.v32u8"([*]const u8, @Vector(32, bool), @Vector(32, u8)) @Vector(32, u8);
        extern fn @"llvm.masked.expandload.v64u8"([*]const u8, @Vector(64, bool), @Vector(64, u8)) @Vector(64, u8);
        extern fn @"llvm.masked.expandload.v1u16"([*]const u16, @Vector(1, bool), @Vector(1, u16)) @Vector(1, u16);
        extern fn @"llvm.masked.expandload.v2u16"([*]const u16, @Vector(2, bool), @Vector(2, u16)) @Vector(2, u16);
        extern fn @"llvm.masked.expandload.v4u16"([*]const u16, @Vector(4, bool), @Vector(4, u16)) @Vector(4, u16);
        extern fn @"llvm.masked.expandload.v8u16"([*]const u16, @Vector(8, bool), @Vector(8, u16)) @Vector(8, u16);
        extern fn @"llvm.masked.expandload.v16u16"([*]const u16, @Vector(16, bool), @Vector(16, u16)) @Vector(16, u16);
        extern fn @"llvm.masked.expandload.v32u16"([*]const u16, @Vector(32, bool), @Vector(32, u16)) @Vector(32, u16);
        extern fn @"llvm.masked.expandload.v1u32"([*]const u32, @Vector(1, bool), @Vector(1, u32)) @Vector(1, u32);
        extern fn @"llvm.masked.expandload.v2u32"([*]const u32, @Vector(2, bool), @Vector(2, u32)) @Vector(2, u32);
        extern fn @"llvm.masked.expandload.v4u32"([*]const u32, @Vector(4, bool), @Vector(4, u32)) @Vector(4, u32);
        extern fn @"llvm.masked.expandload.v8u32"([*]const u32, @Vector(8, bool), @Vector(8, u32)) @Vector(8, u32);
        extern fn @"llvm.masked.expandload.v16u32"([*]const u32, @Vector(16, bool), @Vector(16, u32)) @Vector(16, u32);
        extern fn @"llvm.masked.expandload.v1u64"([*]const u64, @Vector(1, bool), @Vector(1, u64)) @Vector(1, u64);
        extern fn @"llvm.masked.expandload.v2u64"([*]const u64, @Vector(2, bool), @Vector(2, u64)) @Vector(2, u64);
        extern fn @"llvm.masked.expandload.v4u64"([*]const u64, @Vector(4, bool), @Vector(4, u64)) @Vector(4, u64);
        extern fn @"llvm.masked.expandload.v8u64"([*]const u64, @Vector(8, bool), @Vector(8, u64)) @Vector(8, u64);

        // -------------------------------------------------------------------------------------------------

        extern fn @"llvm.masked.compressstore.v1u8"(@Vector(1, u8), [*]u8, @Vector(1, bool)) void;
        extern fn @"llvm.masked.compressstore.v2u8"(@Vector(2, u8), [*]u8, @Vector(2, bool)) void;
        extern fn @"llvm.masked.compressstore.v4u8"(@Vector(4, u8), [*]u8, @Vector(4, bool)) void;
        extern fn @"llvm.masked.compressstore.v8u8"(@Vector(8, u8), [*]u8, @Vector(8, bool)) void;
        extern fn @"llvm.masked.compressstore.v16u8"(@Vector(16, u8), [*]u8, @Vector(16, bool)) void;
        extern fn @"llvm.masked.compressstore.v32u8"(@Vector(32, u8), [*]u8, @Vector(32, bool)) void;
        extern fn @"llvm.masked.compressstore.v64u8"(@Vector(64, u8), [*]u8, @Vector(64, bool)) void;

        extern fn @"llvm.masked.compressstore.v1u16"(@Vector(1, u16), [*]u16, @Vector(1, bool)) void;
        extern fn @"llvm.masked.compressstore.v2u16"(@Vector(2, u16), [*]u16, @Vector(2, bool)) void;
        extern fn @"llvm.masked.compressstore.v4u16"(@Vector(4, u16), [*]u16, @Vector(4, bool)) void;
        extern fn @"llvm.masked.compressstore.v8u16"(@Vector(8, u16), [*]u16, @Vector(8, bool)) void;
        extern fn @"llvm.masked.compressstore.v16u16"(@Vector(16, u16), [*]u16, @Vector(16, bool)) void;
        extern fn @"llvm.masked.compressstore.v32u16"(@Vector(32, u16), [*]u16, @Vector(32, bool)) void;

        extern fn @"llvm.masked.compressstore.v1u32"(@Vector(1, u32), [*]u32, @Vector(1, bool)) void;
        extern fn @"llvm.masked.compressstore.v2u32"(@Vector(2, u32), [*]u32, @Vector(2, bool)) void;
        extern fn @"llvm.masked.compressstore.v4u32"(@Vector(4, u32), [*]u32, @Vector(4, bool)) void;
        extern fn @"llvm.masked.compressstore.v8u32"(@Vector(8, u32), [*]u32, @Vector(8, bool)) void;
        extern fn @"llvm.masked.compressstore.v16u32"(@Vector(16, u32), [*]u32, @Vector(16, bool)) void;

        extern fn @"llvm.masked.compressstore.v1u64"(@Vector(1, u64), [*]u64, @Vector(1, bool)) void;
        extern fn @"llvm.masked.compressstore.v2u64"(@Vector(2, u64), [*]u64, @Vector(2, bool)) void;
        extern fn @"llvm.masked.compressstore.v4u64"(@Vector(4, u64), [*]u64, @Vector(4, bool)) void;
        extern fn @"llvm.masked.compressstore.v8u64"(@Vector(8, u64), [*]u64, @Vector(8, bool)) void;

        // -------------------------------------------------------------------------------------------------
        // Absolute God-tier engineering. You want callconv(.auto) only on the ones with integers that are not valid
        // according to the C calling convention. Otherwise things explode.

        extern fn @"llvm.x86.avx512.mask.compress.b.v16u8"(@Vector(16, u8), @Vector(16, u8), u16) @Vector(16, u8);
        extern fn @"llvm.x86.avx512.mask.compress.b.v32u8"(@Vector(32, u8), @Vector(32, u8), u32) @Vector(32, u8);
        extern fn @"llvm.x86.avx512.mask.compress.b.v64u8"(@Vector(64, u8), @Vector(64, u8), u64) @Vector(64, u8);
        extern fn @"llvm.x86.avx512.mask.compress.w.v8u16"(@Vector(8, u16), @Vector(8, u16), u8) @Vector(8, u16);
        extern fn @"llvm.x86.avx512.mask.compress.w.v16u16"(@Vector(16, u16), @Vector(16, u16), u16) @Vector(16, u16);
        extern fn @"llvm.x86.avx512.mask.compress.w.v32u16"(@Vector(32, u16), @Vector(32, u16), u32) @Vector(32, u16);
        extern fn @"llvm.x86.avx512.mask.compress.d.v4u32"(@Vector(4, u32), @Vector(4, u32), u4) callconv(.auto) @Vector(4, u32);
        extern fn @"llvm.x86.avx512.mask.compress.d.v8u32"(@Vector(8, u32), @Vector(8, u32), u8) @Vector(8, u32);
        extern fn @"llvm.x86.avx512.mask.compress.d.v16u32"(@Vector(16, u32), @Vector(16, u32), u16) @Vector(16, u32);
        extern fn @"llvm.x86.avx512.mask.compress.q.v2u64"(@Vector(2, u64), @Vector(2, u64), u2) callconv(.auto) @Vector(2, u64);
        extern fn @"llvm.x86.avx512.mask.compress.q.v4u64"(@Vector(4, u64), @Vector(4, u64), u4) callconv(.auto) @Vector(4, u64);
        extern fn @"llvm.x86.avx512.mask.compress.q.v8u64"(@Vector(8, u64), @Vector(8, u64), u8) @Vector(8, u64);

        // -------------------------------------------------------------------------------------------------

        extern fn @"llvm.x86.avx512.mask.expand.b.v16u8"(@Vector(16, u8), @Vector(16, u8), u16) @Vector(16, u8);
        extern fn @"llvm.x86.avx512.mask.expand.b.v32u8"(@Vector(32, u8), @Vector(32, u8), u32) @Vector(32, u8);
        extern fn @"llvm.x86.avx512.mask.expand.b.v64u8"(@Vector(64, u8), @Vector(64, u8), u64) @Vector(64, u8);
        extern fn @"llvm.x86.avx512.mask.expand.w.v8u16"(@Vector(8, u16), @Vector(8, u16), u8) @Vector(8, u16);
        extern fn @"llvm.x86.avx512.mask.expand.w.v16u16"(@Vector(16, u16), @Vector(16, u16), u16) @Vector(16, u16);
        extern fn @"llvm.x86.avx512.mask.expand.w.v32u16"(@Vector(32, u16), @Vector(32, u16), u32) @Vector(32, u16);
        extern fn @"llvm.x86.avx512.mask.expand.d.v4u32"(@Vector(4, u32), @Vector(4, u32), u4) callconv(.auto) @Vector(4, u32);
        extern fn @"llvm.x86.avx512.mask.expand.d.v8u32"(@Vector(8, u32), @Vector(8, u32), u8) @Vector(8, u32);
        extern fn @"llvm.x86.avx512.mask.expand.d.v16u32"(@Vector(16, u32), @Vector(16, u32), u16) @Vector(16, u32);
        extern fn @"llvm.x86.avx512.mask.expand.q.v2u64"(@Vector(2, u64), @Vector(2, u64), u2) callconv(.auto) @Vector(2, u64);
        extern fn @"llvm.x86.avx512.mask.expand.q.v4u64"(@Vector(4, u64), @Vector(4, u64), u4) callconv(.auto) @Vector(4, u64);
        extern fn @"llvm.x86.avx512.mask.expand.q.v8u64"(@Vector(8, u64), @Vector(8, u64), u8) @Vector(8, u64);

        // -------------------------------------------------------------------------------------------------

        extern fn @"llvm.masked.load.v1u8"([*]align(1) const u8, i32, @Vector(1, bool), @Vector(1, u8)) @Vector(1, u8);
        extern fn @"llvm.masked.load.v2u8"([*]align(1) const u8, i32, @Vector(2, bool), @Vector(2, u8)) @Vector(2, u8);
        extern fn @"llvm.masked.load.v4u8"([*]align(1) const u8, i32, @Vector(4, bool), @Vector(4, u8)) @Vector(4, u8);
        extern fn @"llvm.masked.load.v8u8"([*]align(1) const u8, i32, @Vector(8, bool), @Vector(8, u8)) @Vector(8, u8);
        extern fn @"llvm.masked.load.v16u8"([*]align(1) const u8, i32, @Vector(16, bool), @Vector(16, u8)) @Vector(16, u8);
        extern fn @"llvm.masked.load.v32u8"([*]align(1) const u8, i32, @Vector(32, bool), @Vector(32, u8)) @Vector(32, u8);
        extern fn @"llvm.masked.load.v64u8"([*]align(1) const u8, i32, @Vector(64, bool), @Vector(64, u8)) @Vector(64, u8);
        extern fn @"llvm.masked.load.v1u16"([*]align(1) const u16, i32, @Vector(1, bool), @Vector(1, u16)) @Vector(1, u16);
        extern fn @"llvm.masked.load.v2u16"([*]align(1) const u16, i32, @Vector(2, bool), @Vector(2, u16)) @Vector(2, u16);
        extern fn @"llvm.masked.load.v4u16"([*]align(1) const u16, i32, @Vector(4, bool), @Vector(4, u16)) @Vector(4, u16);
        extern fn @"llvm.masked.load.v8u16"([*]align(1) const u16, i32, @Vector(8, bool), @Vector(8, u16)) @Vector(8, u16);
        extern fn @"llvm.masked.load.v16u16"([*]align(1) const u16, i32, @Vector(16, bool), @Vector(16, u16)) @Vector(16, u16);
        extern fn @"llvm.masked.load.v32u16"([*]align(1) const u16, i32, @Vector(32, bool), @Vector(32, u16)) @Vector(32, u16);
        extern fn @"llvm.masked.load.v1u32"([*]align(1) const u32, i32, @Vector(1, bool), @Vector(1, u32)) @Vector(1, u32);
        extern fn @"llvm.masked.load.v2u32"([*]align(1) const u32, i32, @Vector(2, bool), @Vector(2, u32)) @Vector(2, u32);
        extern fn @"llvm.masked.load.v4u32"([*]align(1) const u32, i32, @Vector(4, bool), @Vector(4, u32)) @Vector(4, u32);
        extern fn @"llvm.masked.load.v8u32"([*]align(1) const u32, i32, @Vector(8, bool), @Vector(8, u32)) @Vector(8, u32);
        extern fn @"llvm.masked.load.v16u32"([*]align(1) const u32, i32, @Vector(16, bool), @Vector(16, u32)) @Vector(16, u32);
        extern fn @"llvm.masked.load.v1u64"([*]align(1) const u64, i32, @Vector(1, bool), @Vector(1, u64)) @Vector(1, u64);
        extern fn @"llvm.masked.load.v2u64"([*]align(1) const u64, i32, @Vector(2, bool), @Vector(2, u64)) @Vector(2, u64);
        extern fn @"llvm.masked.load.v4u64"([*]align(1) const u64, i32, @Vector(4, bool), @Vector(4, u64)) @Vector(4, u64);
        extern fn @"llvm.masked.load.v8u64"([*]align(1) const u64, i32, @Vector(8, bool), @Vector(8, u64)) @Vector(8, u64);

        // -------------------------------------------------------------------------------------------------

        extern fn @"llvm.masked.store.v1u8"(@Vector(1, u8), [*]align(1) u8, i32, @Vector(1, bool)) void;
        extern fn @"llvm.masked.store.v2u8"(@Vector(2, u8), [*]align(1) u8, i32, @Vector(2, bool)) void;
        extern fn @"llvm.masked.store.v4u8"(@Vector(4, u8), [*]align(1) u8, i32, @Vector(4, bool)) void;
        extern fn @"llvm.masked.store.v8u8"(@Vector(8, u8), [*]align(1) u8, i32, @Vector(8, bool)) void;
        extern fn @"llvm.masked.store.v16u8"(@Vector(16, u8), [*]align(1) u8, i32, @Vector(16, bool)) void;
        extern fn @"llvm.masked.store.v32u8"(@Vector(32, u8), [*]align(1) u8, i32, @Vector(32, bool)) void;
        extern fn @"llvm.masked.store.v64u8"(@Vector(64, u8), [*]align(1) u8, i32, @Vector(64, bool)) void;
        extern fn @"llvm.masked.store.v1u16"(@Vector(1, u16), [*]align(1) u16, i32, @Vector(1, bool)) void;
        extern fn @"llvm.masked.store.v2u16"(@Vector(2, u16), [*]align(1) u16, i32, @Vector(2, bool)) void;
        extern fn @"llvm.masked.store.v4u16"(@Vector(4, u16), [*]align(1) u16, i32, @Vector(4, bool)) void;
        extern fn @"llvm.masked.store.v8u16"(@Vector(8, u16), [*]align(1) u16, i32, @Vector(8, bool)) void;
        extern fn @"llvm.masked.store.v16u16"(@Vector(16, u16), [*]align(1) u16, i32, @Vector(16, bool)) void;
        extern fn @"llvm.masked.store.v32u16"(@Vector(32, u16), [*]align(1) u16, i32, @Vector(32, bool)) void;
        extern fn @"llvm.masked.store.v1u32"(@Vector(1, u32), [*]align(1) u32, i32, @Vector(1, bool)) void;
        extern fn @"llvm.masked.store.v2u32"(@Vector(2, u32), [*]align(1) u32, i32, @Vector(2, bool)) void;
        extern fn @"llvm.masked.store.v4u32"(@Vector(4, u32), [*]align(1) u32, i32, @Vector(4, bool)) void;
        extern fn @"llvm.masked.store.v8u32"(@Vector(8, u32), [*]align(1) u32, i32, @Vector(8, bool)) void;
        extern fn @"llvm.masked.store.v16u32"(@Vector(16, u32), [*]align(1) u32, i32, @Vector(16, bool)) void;
        extern fn @"llvm.masked.store.v1u64"(@Vector(1, u64), [*]align(1) u64, i32, @Vector(1, bool)) void;
        extern fn @"llvm.masked.store.v2u64"(@Vector(2, u64), [*]align(1) u64, i32, @Vector(2, bool)) void;
        extern fn @"llvm.masked.store.v4u64"(@Vector(4, u64), [*]align(1) u64, i32, @Vector(4, bool)) void;
        extern fn @"llvm.masked.store.v8u64"(@Vector(8, u64), [*]align(1) u64, i32, @Vector(8, bool)) void;
    };
};
