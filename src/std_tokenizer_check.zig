//! Checks a tokenizer's output against `std.zig.Tokenizer`, token by token.
//!
//! The compacting tokenizers differ from `std.zig.Tokenizer` by design in a few ways, which are counted separately
//! rather than reported as mismatches:
//!  - They emit `//` tokens for comments. Plain comments are skipped here; doc comments are compared with std's
//!    `doc_comment`/`container_doc_comment` tokens.
//!  - They don't join float literals: `1.5` is `1` `.` `5`, and `1e-5` is `1e` `-` `5`.
//!  - `.?` and `.**` are single tokens (std has `.` `?`, and `.*` `*` with the first tagged `invalid_periodasterisks`).
//!  - A comment, string or character literal that contains a control character (including a tab) is still that kind of
//!    token; validating its contents is left for later. std makes the rest of the line one `invalid` token instead.
//!  - The 512-byte tokenizer targets Zig 0.15+, where `async`, `await` and `usingnamespace` are identifiers.
const std = @import("std");

/// A token, with its kind given by the tokenizer's tag name. Spans are `[start, end)`.
pub const Token = struct {
    kind: []const u8,
    start: u32,
    end: u32,
};

pub const Report = struct {
    /// Tokens produced by `std.zig.Tokenizer` (excluding `eof`).
    std_tokens: usize = 0,
    mismatches: usize = 0,
    float_literals_split: usize = 0,
    merged_symbols: usize = 0,
    control_characters: usize = 0,
    removed_keywords: usize = 0,
    files_with_mismatches: usize = 0,
    printed: usize = 0,

    pub fn add(self: *Report, other: Report) void {
        self.std_tokens += other.std_tokens;
        self.mismatches += other.mismatches;
        self.float_literals_split += other.float_literals_split;
        self.merged_symbols += other.merged_symbols;
        self.control_characters += other.control_characters;
        self.removed_keywords += other.removed_keywords;
        self.files_with_mismatches += other.files_with_mismatches;
    }

    pub fn print(self: Report, name: []const u8) void {
        std.debug.print(
            "{s}: {} mismatches with std.zig.Tokenizer over {} tokens ({} files). Known differences: {} float literals split, {} `.?`/`.**`, {} control characters in comments/strings, {} removed keywords.\n",
            .{ name, self.mismatches, self.std_tokens, self.files_with_mismatches, self.float_literals_split, self.merged_symbols, self.control_characters, self.removed_keywords },
        );
    }
};

/// Compares `tokens` (in source order) against `std.zig.Tokenizer` and adds the results to `report`.
/// Prints up to `max_printed` mismatches across all calls that share `report`.
pub fn check(source: [:0]const u8, tokens: []const Token, path: []const u8, report: *Report, max_printed: usize) void {
    var file_report: Report = .{};
    defer {
        if (file_report.mismatches != 0) file_report.files_with_mismatches = 1;
        report.add(file_report);
    }

    var i: usize = 0;
    var std_tokenizer: std.zig.Tokenizer = .init(source);

    while (true) {
        const t = std_tokenizer.next();
        if (t.tag == .eof) break;
        file_report.std_tokens += 1;

        const before_comments = i;
        i = skipPlainComments(source, tokens, i);
        if (i < tokens.len and tokens[i].start == t.loc.start and tokens[i].end == t.loc.end and
            kindsMatch(t.tag, tokens[i].kind, source[t.loc.start..t.loc.end]))
        {
            i += 1;
            continue;
        }

        if (t.tag == .number_literal and i < tokens.len and tokens[i].start == t.loc.start) {
            if (floatLiteralSplit(tokens, i, t.loc.end)) |next| {
                file_report.float_literals_split += 1;
                i = next;
                continue;
            }
        }

        if ((t.tag == .period or t.tag == .invalid_periodasterisks) and i < tokens.len and tokens[i].start == t.loc.start and
            (std.mem.eql(u8, tokens[i].kind, ".?") or std.mem.eql(u8, tokens[i].kind, ".**")))
        {
            var lookahead = std_tokenizer;
            const next = lookahead.next();
            if (next.loc.start == t.loc.end and next.loc.end == tokens[i].end and
                (next.tag == .question_mark or next.tag == .asterisk))
            {
                file_report.merged_symbols += 1;
                file_report.std_tokens += 1;
                std_tokenizer = lookahead;
                i += 1;
                continue;
            }
        }

        if (t.tag == .invalid and startsTokenWithControlCharacter(source, tokens[before_comments..], t.loc.start)) {
            file_report.control_characters += 1;
            while (i < tokens.len and tokens[i].start < t.loc.end) i += 1;
            continue;
        }

        switch (t.tag) {
            .keyword_async, .keyword_await, .keyword_usingnamespace => if (i < tokens.len and
                tokens[i].start == t.loc.start and tokens[i].end == t.loc.end and
                std.mem.eql(u8, tokens[i].kind, "identifier"))
            {
                file_report.removed_keywords += 1;
                i += 1;
                continue;
            },
            else => {},
        }

        file_report.mismatches += 1;
        if (report.printed < max_printed) {
            report.printed += 1;
            const pos = lineAndColumn(source, t.loc.start);
            std.debug.print("{s}:{}:{}: std has {s} \"{s}\"", .{
                path,
                pos.line,
                pos.column,
                @tagName(t.tag),
                excerpt(source[t.loc.start..t.loc.end]),
            });
            if (i < tokens.len) {
                std.debug.print(", got {s} \"{s}\" at offset {} (std: {})\n", .{
                    tokens[i].kind,
                    excerpt(source[tokens[i].start..@min(tokens[i].end, source.len)]),
                    tokens[i].start,
                    t.loc.start,
                });
            } else std.debug.print(", got nothing\n", .{});
        }

        // Resynchronize after the mismatched std token.
        while (i < tokens.len and tokens[i].start < t.loc.end) i += 1;
    }

    i = skipPlainComments(source, tokens, i);
    while (i < tokens.len) : (i = skipPlainComments(source, tokens, i + 1)) {
        file_report.mismatches += 1;
        if (report.printed < max_printed) {
            report.printed += 1;
            const pos = lineAndColumn(source, tokens[i].start);
            std.debug.print("{s}:{}:{}: std has nothing, got {s} \"{s}\"\n", .{
                path,
                pos.line,
                pos.column,
                tokens[i].kind,
                excerpt(source[tokens[i].start..@min(tokens[i].end, source.len)]),
            });
        }
    }
}

/// Whether one of `tokens` starts at `start` and is a comment, string, character literal or line of a multiline
/// string with a control character in it.
fn startsTokenWithControlCharacter(source: []const u8, tokens: []const Token, start: usize) bool {
    for (tokens) |token| {
        if (token.start > start) return false;
        if (token.start < start) continue;
        const kinds = [_][]const u8{ "//", "string", "char_literal", "double_backslash", "\\\\" };
        for (kinds) |kind| {
            if (!std.mem.eql(u8, token.kind, kind)) continue;
            for (source[token.start..token.end]) |c| switch (c) {
                0x01...0x09, 0x0b...0x0c, 0x0e...0x1f, 0x7f => return true,
                else => {},
            };
        }
        return false;
    }
    return false;
}

fn isPlainComment(source: []const u8, token: Token) bool {
    if (!std.mem.eql(u8, token.kind, "//")) return false;
    const text = source[token.start..token.end];
    const is_doc_comment = (std.mem.startsWith(u8, text, "///") and !std.mem.startsWith(u8, text, "////")) or
        std.mem.startsWith(u8, text, "//!");
    return !is_doc_comment;
}

fn skipPlainComments(source: []const u8, tokens: []const Token, start: usize) usize {
    var i = start;
    while (i < tokens.len and isPlainComment(source, tokens[i])) i += 1;
    return i;
}

fn kindsMatch(tag: std.zig.Token.Tag, kind: []const u8, text: []const u8) bool {
    const eql = std.mem.eql;
    if (tag.lexeme()) |lexeme| return eql(u8, kind, lexeme);
    return switch (tag) {
        .identifier => eql(u8, kind, if (std.mem.startsWith(u8, text, "@\"")) "string_identifier" else "identifier"),
        .builtin => eql(u8, kind, "builtin"),
        .number_literal => eql(u8, kind, "number"),
        .string_literal => eql(u8, kind, "string"),
        .char_literal => eql(u8, kind, "char_literal"),
        .multiline_string_literal_line => eql(u8, kind, "double_backslash") or eql(u8, kind, "\\\\"),
        .doc_comment => eql(u8, kind, "//") or eql(u8, kind, "///"),
        .container_doc_comment => eql(u8, kind, "//") or eql(u8, kind, "//!"),
        .invalid => eql(u8, kind, "invalid"),
        else => false,
    };
}

/// If the tokens starting at `tokens[i]` exactly cover a float literal that ends at `end`, returns the index after them.
fn floatLiteralSplit(tokens: []const Token, i: usize, end: usize) ?usize {
    var k = i;
    while (k < tokens.len and tokens[k].end <= end) : (k += 1) {
        const kind = tokens[k].kind;
        const allowed = std.mem.eql(u8, kind, "number") or std.mem.eql(u8, kind, "identifier") or
            std.mem.eql(u8, kind, ".") or std.mem.eql(u8, kind, "+") or std.mem.eql(u8, kind, "-");
        if (!allowed) return null;
    }
    if (k - i < 2 or tokens[k - 1].end != end) return null;
    return k;
}

fn lineAndColumn(source: []const u8, pos: usize) struct { line: usize, column: usize } {
    const before = source[0..@min(pos, source.len)];
    const line = std.mem.count(u8, before, "\n") + 1;
    const line_start = if (std.mem.lastIndexOfScalar(u8, before, '\n')) |nl| nl + 1 else 0;
    return .{ .line = line, .column = pos - line_start + 1 };
}

fn excerpt(text: []const u8) []const u8 {
    const line_end = std.mem.indexOfScalar(u8, text, '\n') orelse text.len;
    return text[0..@min(line_end, 60)];
}

fn expectMatchesStd(gpa: std.mem.Allocator, source: [:0]const u8, name: []const u8) !void {
    const CompactingTokenizer = @import("CompactingTokenizer.zig");
    var tokenizer: CompactingTokenizer = .empty;
    defer tokenizer.deinit(gpa);
    try tokenizer.tokenize(gpa, source);

    const slice = tokenizer.token_list.slice();
    const tokens = try gpa.alloc(Token, slice.len);
    defer gpa.free(tokens);
    for (tokens, slice.items(.kind), slice.items(.start), slice.items(.end)) |*token, kind, start, end|
        token.* = .{ .kind = @tagName(kind), .start = start, .end = end };

    var report: Report = .{};
    check(source, tokens, name, &report, 5);
    try std.testing.expectEqual(0, report.mismatches);
}

test "512-byte compression tokenizer matches std.zig.Tokenizer across chunk boundaries" {
    // Every kind of token, slid across the 64-byte and 512-byte chunk boundaries and pointer misalignments.
    // Not included: `<<|=`, which the multi-char symbol matcher can't produce (it is limited to 3 characters).
    const snippet =
        \\const x = @import("std"); // comment with "quote" and 'c' and \\ backslashes
        \\/// doc comment
        \\//! container doc
        \\//// not a doc comment
        \\pub fn foo(a: u32) u32 { return a +% 1 <<| 2; }
        \\var s = "string with \" escaped \\ and // not comment";
        \\const c = 'x'; const d = '\''; const e = '\\'; const f = '"';
        \\const m =
        \\    \\multiline "string" // not a comment
        \\    \\second line
        \\;
        \\x.* = y.?; z = a.**b; w = .{ .a = 1 }; q = a...b; r = a ++ b ** c; t = a ||| b;
        \\if (a <= b and c >= d or e != f) {} else if (a == b) {}
        \\const @"weird name" = 1; _ = @"x" + @This();
        \\_ = a +|= b; a -%= b; a >>= 3; a => b; a -> b; a *%= b; a *|= b; a +%= b; f = 1.5e-3;
        \\const keywords = .{ if, else, while, for, switch, struct, enum, union, orelse, catch, try, fn, threadlocal, linksection, unreachable, iff, fnx, x_if, _if, if_, if1 };
        \\identifierThatIsLongEnoughToCrossBoundaries_______________________________________________________ = 0x1F_FF;
        \\const n = 12345678901234567890; const builtinCall = @intFromEnum(@as(u8, 1)); export fn @"export"() void {}
        \\
    ;

    const gpa = std.testing.allocator;
    const buffer = try gpa.alignedAlloc(u8, 64, 2048);
    defer gpa.free(buffer);

    var name_buffer: [64]u8 = undefined;
    for ([_]usize{ 0, 1, 7, 33, 63 }) |misalignment| {
        for (0..700) |padding| {
            @memset(buffer, 0);
            @memset(buffer[misalignment..][0..padding], ' ');
            @memcpy(buffer[misalignment + padding ..][0..snippet.len], snippet);
            const source: [:0]const u8 = buffer[misalignment..][0 .. padding + snippet.len :0];
            try expectMatchesStd(gpa, source, try std.fmt.bufPrint(&name_buffer, "misalignment {}, padding {}", .{ misalignment, padding }));
        }
    }
}

test "512-byte compression tokenizer matches std.zig.Tokenizer on demo.zig" {
    const gpa = std.testing.allocator;
    const demo = @embedFile("files_to_parse/demo.zig");
    const buffer = try gpa.alignedAlloc(u8, 64, demo.len + 1);
    defer gpa.free(buffer);
    @memcpy(buffer[0..demo.len], demo);
    buffer[demo.len] = 0;
    try expectMatchesStd(gpa, buffer[0..demo.len :0], "demo.zig");
}
