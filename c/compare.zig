//! Checks the C port (c/compacting_tokenizer.c) against the Zig implementation token by token, then benchmarks both.
//! Run it with `zig build c-compare -Doptimize=ReleaseFast -Dcpu=... -Dc-march=... -- [corpus directory]`.
const std = @import("std");
const builtin = @import("builtin");
const CompactingTokenizer = @import("CompactingTokenizer");

extern fn azp_tokenize(source: [*]const u8, len: usize, kinds: [*]u8, starts: [*]u32, ends: [*]u32) isize;

const PASSES = 21;

/// The Zig implementation, as its own function so its code can be compared with `azp_tokenize`. Returns the number of
/// tokens (left in `zig_tokenizer`) or a negative error like the C port.
export fn zig_tokenize(source: [*]const u8, len: usize) isize {
    zig_tokenizer.tokenizeAssumeCapacity(source[0..len :0]) catch |err| return switch (err) {
        error.InvalidCarriageReturn => -1,
        error.InvalidUtf8 => -2,
    };
    return @intCast(zig_tokenizer.token_list.len);
}

/// `zig_tokenize`, with the tokens copied into the C port's layout.
fn zigTokenizeInto(source: []const u8, kinds: [*]u8, starts: [*]u32, ends: [*]u32) isize {
    const count = zig_tokenize(source.ptr, source.len);
    if (count < 0) return count;
    const tokens = zig_tokenizer.token_list.slice();
    @memcpy(kinds[0..tokens.len], @as([]const u8, @ptrCast(tokens.items(.kind))));
    @memcpy(starts[0..tokens.len], tokens.items(.start));
    @memcpy(ends[0..tokens.len], tokens.items(.end));
    return @intCast(tokens.len);
}

var zig_tokenizer: CompactingTokenizer = .empty;

const Output = struct {
    kinds: []u8,
    starts: []u32,
    ends: []u32,

    fn alloc(gpa: std.mem.Allocator, capacity: usize) !Output {
        const output: Output = .{
            .kinds = try gpa.alloc(u8, capacity),
            .starts = try gpa.alloc(u32, capacity),
            .ends = try gpa.alloc(u32, capacity),
        };
        @memset(output.kinds, 0);
        @memset(output.starts, 0);
        @memset(output.ends, 0);
        return output;
    }
};

const Implementation = enum {
    zig,
    c,

    /// Tokenizes `source`, leaving the tokens wherever the implementation writes them.
    fn tokenize(impl: Implementation, source: []const u8, output: Output) isize {
        return switch (impl) {
            .zig => zig_tokenize(source.ptr, source.len),
            .c => azp_tokenize(source.ptr, source.len, output.kinds.ptr, output.starts.ptr, output.ends.ptr),
        };
    }
};

/// Compares the two implementations' output for `source`. Returns false (after printing the difference) if they differ.
fn sameOutput(source: []const u8, zig_output: Output, c_output: Output, name: []const u8) bool {
    const zig_count = zigTokenizeInto(source, zig_output.kinds.ptr, zig_output.starts.ptr, zig_output.ends.ptr);
    const c_count = Implementation.c.tokenize(source, c_output);
    if (zig_count != c_count) {
        std.debug.print("{s}: Zig returned {}, C returned {}\n", .{ name, zig_count, c_count });
        return false;
    }
    if (zig_count < 0) return true;
    const n: usize = @intCast(zig_count);
    for (0..n) |i| {
        if (zig_output.kinds[i] != c_output.kinds[i] or zig_output.starts[i] != c_output.starts[i] or zig_output.ends[i] != c_output.ends[i]) {
            std.debug.print("{s}: token {}: Zig has {} [{}, {}), C has {} [{}, {})\n", .{
                name,
                i,
                zig_output.kinds[i],
                zig_output.starts[i],
                zig_output.ends[i],
                c_output.kinds[i],
                c_output.starts[i],
                c_output.ends[i],
            });
            return false;
        }
    }
    return true;
}

const boundary_snippet =
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
    \\x.* = y.?; z = a.**b; w = .{ .a = 1 }; q = a...b; r = a ++ b ** c; t = a ||| b; u = a <<|= b;
    \\if (a <= b and c >= d or e != f) {} else if (a == b) {}
    \\const @"weird name" = 1; _ = @"x" + @This(); // ünïcödé
    \\_ = a +|= b; a -%= b; a >>= 3; a => b; a -> b; a *%= b; a *|= b; a +%= b; f = 1.5e-3;
    \\const keywords = .{ if, else, while, for, switch, struct, enum, union, orelse, catch, try, fn, threadlocal, linksection, unreachable, iff, fnx, x_if, _if, if_, if1 };
    \\identifierThatIsLongEnoughToCrossBoundaries_______________________________________________________ = 0x1F_FF;
    \\const n = 12345678901234567890; const builtinCall = @intFromEnum(@as(u8, 1)); export fn @"export"() void {} # $ `
    \\
;

const PerfCounters = struct {
    cycles_fd: ?std.posix.fd_t,
    instructions_fd: ?std.posix.fd_t,

    const Sample = struct { ns: u64, cycles: u64, instructions: u64 };

    fn open() PerfCounters {
        return .{ .cycles_fd = openCounter(.CPU_CYCLES), .instructions_fd = openCounter(.INSTRUCTIONS) };
    }

    fn openCounter(counter: std.os.linux.PERF.COUNT.HW) ?std.posix.fd_t {
        if (builtin.os.tag != .linux) return null;
        var attr: std.os.linux.perf_event_attr = .{ .type = .HARDWARE, .config = @intFromEnum(counter) };
        attr.flags.exclude_kernel = true;
        attr.flags.exclude_hv = true;
        const rc = std.os.linux.perf_event_open(&attr, 0, -1, -1, std.os.linux.PERF.FLAG.FD_CLOEXEC);
        if (std.os.linux.E.init(rc) != .SUCCESS) return null;
        return @intCast(rc);
    }

    fn read(fd: ?std.posix.fd_t) u64 {
        var value: u64 = 0;
        if (fd) |f| _ = std.posix.read(f, std.mem.asBytes(&value)) catch {};
        return value;
    }

    fn now(self: PerfCounters) Sample {
        return .{ .cycles = read(self.cycles_fd), .instructions = read(self.instructions_fd), .ns = @intCast(std.time.nanoTimestamp()) };
    }
};

pub fn main() !void {
    const gpa = std.heap.page_allocator;
    const args = try std.process.argsAlloc(gpa);
    const corpus_path = if (args.len > 1) args[1] else "src/files_to_parse";

    // Read the corpus into 64-byte-aligned, zero-padded buffers, like main.zig does.
    var sources: std.ArrayListUnmanaged([:0]const u8) = .empty;
    var total_bytes: usize = 0;
    var max_len: usize = boundary_snippet.len + 1024;
    {
        var dir = try std.fs.cwd().openDir(corpus_path, .{ .iterate = true });
        defer dir.close();
        var walker = try dir.walk(gpa);
        defer walker.deinit();
        while (try walker.next()) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.basename, ".zig")) continue;
            // The same outliers that main.zig skips.
            if (std.mem.eql(u8, entry.basename, "udivmodti4_test.zig") or std.mem.eql(u8, entry.basename, "udivmoddi4_test.zig")) continue;
            const file = try dir.openFile(entry.path, .{});
            defer file.close();
            const len: usize = @intCast(try file.getEndPos());
            const buffer = try gpa.alignedAlloc(u8, 64, std.mem.alignForward(usize, len + 1, 64));
            @memset(buffer, 0);
            _ = try file.readAll(buffer[0..len]);
            try sources.append(gpa, buffer[0..len :0]);
            total_bytes += len;
            max_len = @max(max_len, len);
        }
    }
    std.debug.print("{} files, {} bytes\n", .{ sources.items.len, total_bytes });

    try zig_tokenizer.ensureTokenCapacity(gpa, max_len);
    const zig_output = try Output.alloc(gpa, max_len + 64);
    const c_output = try Output.alloc(gpa, max_len + 64);

    // 1. The same tokens for every file, and for a snippet slid across the 64- and 512-byte chunk boundaries.
    var differences: usize = 0;
    for (sources.items, 0..) |source, i| {
        if (!sameOutput(source, zig_output, c_output, try std.fmt.allocPrint(gpa, "file {}", .{i}))) differences += 1;
    }
    {
        const buffer = try gpa.alignedAlloc(u8, 64, 4096);
        var name_buffer: [64]u8 = undefined;
        for ([_]usize{ 0, 1, 7, 33, 63 }) |misalignment| {
            for (0..1100) |padding| {
                @memset(buffer, 0);
                @memset(buffer[misalignment..][0..padding], ' ');
                @memcpy(buffer[misalignment + padding ..][0..boundary_snippet.len], boundary_snippet);
                const source = buffer[misalignment..][0 .. padding + boundary_snippet.len];
                const name = try std.fmt.bufPrint(&name_buffer, "snippet, misalignment {}, padding {}", .{ misalignment, padding });
                if (!sameOutput(source, zig_output, c_output, name)) differences += 1;
            }
        }
    }
    std.debug.print("C port vs Zig: {} inputs with different tokens ({} files, 5500 snippet placements)\n", .{ differences, sources.items.len });

    // 2. Throughput, with the two implementations taking turns. The first pass is a warm-up and isn't recorded.
    const counters: PerfCounters = .open();
    var samples: [2][PASSES]PerfCounters.Sample = undefined;
    for (0..PASSES + 1) |pass| {
        inline for (.{ Implementation.zig, Implementation.c }, 0..) |impl, which| {
            const output = if (impl == .zig) zig_output else c_output;
            const before = counters.now();
            for (sources.items) |source| std.mem.doNotOptimizeAway(impl.tokenize(source, output));
            const after = counters.now();
            if (pass > 0) samples[which][pass - 1] = .{
                .ns = after.ns - before.ns,
                .cycles = after.cycles - before.cycles,
                .instructions = after.instructions - before.instructions,
            };
        }
    }

    std.debug.print("{} interleaved passes, medians (cycles/B also min-max):\n", .{PASSES});
    for ([_][]const u8{ "Zig", "C" }, &samples) |name, *impl_samples| {
        var ns: [PASSES]u64 = undefined;
        var cycles: [PASSES]u64 = undefined;
        var instructions: [PASSES]u64 = undefined;
        for (impl_samples, &ns, &cycles, &instructions) |sample, *n, *c, *i| {
            n.* = sample.ns;
            c.* = sample.cycles;
            i.* = sample.instructions;
        }
        inline for (.{ &ns, &cycles, &instructions }) |values| std.mem.sort(u64, values, {}, std.sort.asc(u64));
        const b: f64 = @floatFromInt(total_bytes);
        const median_cycles: f64 = @floatFromInt(cycles[PASSES / 2]);
        std.debug.print("  {s: <4} {d: >7.3} ms  {d:.2} GB/s  {d:.3} cycles/B [{d:.3}-{d:.3}]  {d:.3} instructions/B  IPC {d:.2}\n", .{
            name,
            @as(f64, @floatFromInt(ns[PASSES / 2])) / 1e6,
            b / @as(f64, @floatFromInt(ns[PASSES / 2])),
            median_cycles / b,
            @as(f64, @floatFromInt(cycles[0])) / b,
            @as(f64, @floatFromInt(cycles[PASSES - 1])) / b,
            @as(f64, @floatFromInt(instructions[PASSES / 2])) / b,
            @as(f64, @floatFromInt(instructions[PASSES / 2])) / median_cycles,
        });
    }
}
