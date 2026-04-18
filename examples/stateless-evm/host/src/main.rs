use clap::{Parser, ValueEnum};
use eyre::{bail, Context, Result};
use guest::{CryptoTraceStats, MemopsTraceStats, PreparedStatelessInput, ValidationOutput};
use jolt_sdk::{host::Program, F};
use object::{Object, ObjectSymbol, SymbolKind};
use stateless::{StatelessInput, UncompressedPublicKey};
use std::{
    collections::{BTreeMap, HashMap},
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tracer::{
    instruction::{Cycle, Instruction, RAMAccess},
    list_registered_inlines,
};

struct GuestBinary {
    raw_bytecode: Vec<Instruction>,
    symbols: Vec<FunctionSymbol>,
}

struct FunctionSymbol {
    start: u64,
    end: u64,
    name: String,
}

fn load_guest_binary(program: &Program) -> GuestBinary {
    let elf_contents = program
        .get_elf_contents()
        .expect("ELF should be built before analyze");
    let (raw_bytecode, _raw_bytes, _program_end, _entry, _xlen) = tracer::decode(&elf_contents);
    let symbols = load_function_symbols(&elf_contents);
    GuestBinary {
        raw_bytecode,
        symbols,
    }
}

fn load_function_symbols(elf: &[u8]) -> Vec<FunctionSymbol> {
    let obj = match object::File::parse(elf) {
        Ok(obj) => obj,
        Err(err) => {
            info!("failed to parse ELF for symbol attribution: {err}");
            return Vec::new();
        }
    };
    let mut symbols: Vec<FunctionSymbol> = obj
        .symbols()
        .filter(|sym| sym.kind() == SymbolKind::Text && sym.size() > 0)
        .filter_map(|sym| {
            let name = sym.name().ok()?;
            let demangled = rustc_demangle::try_demangle(name)
                .map(|d| format!("{d:#}"))
                .unwrap_or_else(|_| name.to_owned());
            Some(FunctionSymbol {
                start: sym.address(),
                end: sym.address() + sym.size(),
                name: demangled,
            })
        })
        .collect();
    symbols.sort_by_key(|sym| sym.start);
    symbols
}

fn lookup_function_id(symbols: &[FunctionSymbol], addr: u64) -> Option<usize> {
    let idx = symbols.partition_point(|sym| sym.start <= addr);
    if idx == 0 {
        return None;
    }
    let candidate = &symbols[idx - 1];
    if addr < candidate.end {
        Some(idx - 1)
    } else {
        None
    }
}
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(about = "Run the canonical stateless Ethereum validator with Jolt")]
struct Args {
    /// Path to an upstream-style JSON fixture, raw StatelessInput JSON, or
    /// prepared postcard input. Directories are treated as fixture sets for
    /// tracing mode.
    input: PathBuf,

    /// Whether to trace or run full prove+verify.
    #[arg(long, value_enum, default_value_t = Mode::Analyze)]
    mode: Mode,

    /// Directory used by Jolt to cache compiled guest artifacts.
    #[arg(long, default_value = "/tmp/jolt-guest-targets")]
    target_dir: PathBuf,

    /// Keccak backend to use when compiling the guest.
    #[arg(long, value_enum, default_value_t = KeccakBackend::Inline)]
    keccak_backend: KeccakBackend,

    /// Optional output path for the prepared postcard input. Only valid for a
    /// single input file.
    #[arg(long)]
    write_prepared: Option<PathBuf>,

    /// Limit the number of fixtures traced when the input is a directory.
    #[arg(long)]
    limit: Option<usize>,

    /// Only trace fixtures whose file name contains one of these substrings.
    #[arg(long)]
    filter: Vec<String>,

    /// Continue tracing remaining fixtures after a per-file failure.
    #[arg(long, default_value_t = false)]
    keep_going: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    Analyze,
    Prove,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum KeccakBackend {
    Inline,
    Software,
}

impl KeccakBackend {
    fn extra_guest_features(self) -> &'static [&'static str] {
        match self {
            Self::Inline => &[],
            Self::Software => &["software-keccak"],
        }
    }

    fn target_dir_component(self) -> &'static str {
        match self {
            Self::Inline => "keccak-inline",
            Self::Software => "keccak-software",
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct StatelessValidatorFixture {
    name: String,
    stateless_input: StatelessInput,
    success: bool,
}

#[derive(Debug)]
struct LoadedInput {
    name: String,
    expected_success: Option<bool>,
    prepared: PreparedStatelessInput,
}

#[derive(Debug, Clone)]
struct AnalysisResult {
    name: String,
    cycles: usize,
    padded_cycles: usize,
    success: bool,
}

const TOP_INSTRUCTION_LIMIT: usize = 12;

fn count_entry(counts: &mut BTreeMap<&'static str, usize>, key: &'static str) {
    *counts.entry(key).or_insert(0) += 1;
}

fn sorted_counts(counts: BTreeMap<&'static str, usize>) -> Vec<(&'static str, usize)> {
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1).then_with(|| lhs.0.cmp(rhs.0)));
    counts
}

fn format_counts(counts: &[(&'static str, usize)], total: usize) -> String {
    if total == 0 || counts.is_empty() {
        return "none".to_owned();
    }

    counts
        .iter()
        .map(|(name, count)| {
            format!(
                "{name}={count} ({:.2}%)",
                (*count as f64 / total as f64) * 100.0
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn is_control_flow_instruction(name: &str) -> bool {
    matches!(
        name,
        "BEQ"
            | "BGE"
            | "BGEU"
            | "BLT"
            | "BLTU"
            | "BNE"
            | "JAL"
            | "JALR"
            | "MRET"
            | "WFI"
            | "ECALL"
            | "EBREAK"
            | "FENCE"
            | "FENCEI"
    )
}

fn classify_real_cycle(cycle: &Cycle) -> &'static str {
    let instruction = cycle.instruction();
    if instruction.normalize().virtual_sequence_remaining.is_some() {
        return "inline_anchor";
    }

    match cycle.ram_access() {
        RAMAccess::Read(_) | RAMAccess::Write(_) => "load_store",
        RAMAccess::NoOp => {
            let instruction_name: &'static str = cycle.into();
            if is_control_flow_instruction(instruction_name) {
                "control_flow"
            } else {
                "alu_misc"
            }
        }
    }
}

fn classify_expanded_cycle(cycle: &Cycle) -> &'static str {
    let instruction = cycle.instruction();
    if instruction.normalize().virtual_sequence_remaining.is_some() {
        return if instruction.is_real() {
            "inline_anchor"
        } else {
            "inline_helper"
        };
    }

    match cycle.ram_access() {
        RAMAccess::Read(_) | RAMAccess::Write(_) => "load_store",
        RAMAccess::NoOp => {
            let instruction_name: &'static str = cycle.into();
            if is_control_flow_instruction(instruction_name) {
                "control_flow"
            } else {
                "alu_misc"
            }
        }
    }
}

#[derive(Default, Clone, Copy)]
struct OpcodeSplit {
    helper: usize,
    anchor: usize,
    ordinary: usize,
}

#[derive(Default, Clone, Copy)]
struct SourceCounts {
    anchor_cycles: usize,
    helper_cycles: usize,
}

fn build_bytecode_source_labels(raw_bytecode: &[Instruction]) -> HashMap<u64, String> {
    let inline_names: HashMap<(u32, u32, u32), String> =
        list_registered_inlines().into_iter().collect();

    let mut labels = HashMap::with_capacity(raw_bytecode.len());
    for instruction in raw_bytecode {
        let addr = instruction.normalize().address as u64;
        let label = match instruction {
            Instruction::INLINE(inline) => {
                let key = (inline.opcode, inline.funct3, inline.funct7);
                inline_names
                    .get(&key)
                    .map(|name| format!("INLINE:{name}"))
                    .unwrap_or_else(|| format!("INLINE:<unregistered {key:?}>"))
            }
            other => {
                let name: &'static str = other.into();
                name.to_owned()
            }
        };
        labels.insert(addr, label);
    }
    labels
}

fn log_trace_stats(
    name: &str,
    guest_binary: &GuestBinary,
    expanded_bytecode: &[Instruction],
    trace: &[Cycle],
) {
    let raw_bytecode = guest_binary.raw_bytecode.as_slice();
    let static_raw_bytecode_instructions = raw_bytecode.len();
    let static_bytecode_instructions = expanded_bytecode.len();
    let static_compressed_bytecode = expanded_bytecode
        .iter()
        .filter(|instruction| instruction.normalize().is_compressed)
        .count();
    let raw_riscv_cycles = trace
        .iter()
        .filter(|cycle| cycle.instruction().is_real())
        .count();
    let expanded_cycles = trace.len();
    let inline_helper_cycles = expanded_cycles.saturating_sub(raw_riscv_cycles);

    let bytecode_labels = build_bytecode_source_labels(raw_bytecode);

    let mut raw_categories = BTreeMap::new();
    let mut expanded_categories = BTreeMap::new();
    let mut expanded_instruction_counts = BTreeMap::new();
    let mut opcode_split: BTreeMap<&'static str, OpcodeSplit> = BTreeMap::new();
    // Per high-level source instruction (looked up by PC in the ELF bytecode):
    // how many anchor and helper cycles did each one produce?
    let mut source_counts: HashMap<String, SourceCounts> = HashMap::new();

    for cycle in trace {
        let instruction_name: &'static str = cycle.into();
        count_entry(&mut expanded_instruction_counts, instruction_name);
        count_entry(&mut expanded_categories, classify_expanded_cycle(cycle));

        let instruction = cycle.instruction();
        let normalized = instruction.normalize();
        let entry = opcode_split.entry(instruction_name).or_default();
        match normalized.virtual_sequence_remaining {
            Some(0) => entry.anchor += 1,
            Some(_) => entry.helper += 1,
            None => entry.ordinary += 1,
        }

        if let Some(label) = bytecode_labels.get(&(normalized.address as u64)) {
            let source = source_counts.entry(label.clone()).or_default();
            match normalized.virtual_sequence_remaining {
                Some(0) => source.anchor_cycles += 1,
                Some(_) => source.helper_cycles += 1,
                None => {}
            }
        }

        if instruction.is_real() {
            count_entry(&mut raw_categories, classify_real_cycle(cycle));
        }
    }

    let raw_categories = sorted_counts(raw_categories);
    let expanded_categories = sorted_counts(expanded_categories);
    let top_expanded_instructions = sorted_counts(expanded_instruction_counts)
        .into_iter()
        .take(TOP_INSTRUCTION_LIMIT)
        .collect::<Vec<_>>();

    let expansion_factor = if raw_riscv_cycles == 0 {
        0.0
    } else {
        expanded_cycles as f64 / raw_riscv_cycles as f64
    };

    info!(
        "trace detail: name={name} static_raw_bytecode_instructions={static_raw_bytecode_instructions} static_bytecode_instructions={static_bytecode_instructions} static_compressed_bytecode={static_compressed_bytecode} raw_riscv_cycles={raw_riscv_cycles} expanded_cycles={expanded_cycles} inline_helper_cycles={inline_helper_cycles} expansion_factor={expansion_factor:.4}"
    );
    info!(
        "trace categories (raw): {}",
        format_counts(&raw_categories, raw_riscv_cycles)
    );
    info!(
        "trace categories (expanded): {}",
        format_counts(&expanded_categories, expanded_cycles)
    );
    info!(
        "trace top instructions (expanded): {}",
        format_counts(&top_expanded_instructions, expanded_cycles)
    );

    log_opcode_split(name, &opcode_split, expanded_cycles);
    log_helpers_by_source(name, &source_counts, inline_helper_cycles);
    log_memops_by_function(name, trace, raw_bytecode, &guest_binary.symbols);
}

fn log_opcode_split(
    name: &str,
    opcode_split: &BTreeMap<&'static str, OpcodeSplit>,
    expanded_cycles: usize,
) {
    let mut rows: Vec<(&'static str, OpcodeSplit, usize)> = opcode_split
        .iter()
        .map(|(opname, split)| {
            let total = split.helper + split.anchor + split.ordinary;
            (*opname, *split, total)
        })
        .collect();
    rows.sort_by(|lhs, rhs| rhs.2.cmp(&lhs.2).then_with(|| lhs.0.cmp(rhs.0)));
    let pct = |count: usize| -> f64 {
        if expanded_cycles == 0 {
            0.0
        } else {
            100.0 * count as f64 / expanded_cycles as f64
        }
    };
    let formatted = rows
        .into_iter()
        .take(TOP_INSTRUCTION_LIMIT)
        .map(|(opname, split, total)| {
            format!(
                "{opname}=(total={total} [{:.2}%] helper={} anchor={} ordinary={})",
                pct(total),
                split.helper,
                split.anchor,
                split.ordinary,
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    info!("trace opcode split (expanded, top {TOP_INSTRUCTION_LIMIT}): name={name} {formatted}");
}

fn log_helpers_by_source(
    name: &str,
    source_counts: &HashMap<String, SourceCounts>,
    total_helper_cycles: usize,
) {
    let mut rows: Vec<(&str, SourceCounts)> = source_counts
        .iter()
        .filter(|(_, counts)| counts.helper_cycles > 0 || counts.anchor_cycles > 0)
        .map(|(label, counts)| (label.as_str(), *counts))
        .collect();
    rows.sort_by(|lhs, rhs| {
        rhs.1
            .helper_cycles
            .cmp(&lhs.1.helper_cycles)
            .then_with(|| rhs.1.anchor_cycles.cmp(&lhs.1.anchor_cycles))
            .then_with(|| lhs.0.cmp(rhs.0))
    });

    let pct = |count: usize| -> f64 {
        if total_helper_cycles == 0 {
            0.0
        } else {
            100.0 * count as f64 / total_helper_cycles as f64
        }
    };

    let formatted = rows
        .into_iter()
        .take(TOP_INSTRUCTION_LIMIT)
        .map(|(label, counts)| {
            let avg = if counts.anchor_cycles == 0 {
                0.0
            } else {
                counts.helper_cycles as f64 / counts.anchor_cycles as f64
            };
            format!(
                "{label}=(helpers={} [{:.2}% of helpers], anchors={}, avg_helpers_per_anchor={avg:.2})",
                counts.helper_cycles,
                pct(counts.helper_cycles),
                counts.anchor_cycles,
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    info!(
        "trace helpers by source (top {TOP_INSTRUCTION_LIMIT} of {total_helper_cycles} total helper cycles): name={name} {formatted}"
    );
}

/// Sub-word memory operations whose tracer expansions dominate helper cycles on
/// MPT-heavy workloads. Each anchor here expands into 7-14 virtual instructions
/// because Jolt's RAM is 8-byte-aligned only.
const SUBWORD_OPS: &[&str] = &["LBU", "LB", "LHU", "LH", "LW", "LWU", "SB", "SH", "SW"];

fn log_memops_by_function(
    name: &str,
    trace: &[Cycle],
    raw_bytecode: &[Instruction],
    symbols: &[FunctionSymbol],
) {
    if symbols.is_empty() {
        info!("trace memops-by-function: name={name} no symbols available");
        return;
    }

    let subword_by_addr: HashMap<u64, &'static str> = raw_bytecode
        .iter()
        .filter_map(|instruction| {
            let normalized = instruction.normalize();
            let name: &'static str = instruction.into();
            if SUBWORD_OPS.contains(&name) {
                Some((normalized.address as u64, name))
            } else {
                None
            }
        })
        .collect();

    let fn_by_addr: HashMap<u64, usize> = subword_by_addr
        .keys()
        .filter_map(|&addr| lookup_function_id(symbols, addr).map(|id| (addr, id)))
        .collect();

    #[derive(Default, Clone, Copy)]
    struct MemopStats {
        anchors: u64,
        helpers: u64,
    }

    let mut by_function: HashMap<usize, HashMap<&'static str, MemopStats>> = HashMap::new();
    let mut unattributed: HashMap<&'static str, MemopStats> = HashMap::new();
    let mut total_anchors: u64 = 0;
    let mut total_helpers: u64 = 0;

    for cycle in trace {
        let normalized = cycle.instruction().normalize();
        let Some(&op_name) = subword_by_addr.get(&(normalized.address as u64)) else {
            continue;
        };
        let bucket = match fn_by_addr.get(&(normalized.address as u64)) {
            Some(&fn_id) => by_function.entry(fn_id).or_default(),
            None => &mut unattributed,
        };
        let stats = bucket.entry(op_name).or_default();
        match normalized.virtual_sequence_remaining {
            Some(0) => {
                stats.anchors += 1;
                total_anchors += 1;
            }
            Some(_) => {
                stats.helpers += 1;
                total_helpers += 1;
            }
            None => {}
        }
    }

    let total_cycles = total_anchors + total_helpers;
    info!(
        "trace memops-by-function summary: name={name} total_subword_anchors={total_anchors} total_subword_helpers={total_helpers} total_subword_cycles={total_cycles}"
    );

    let mut per_function: Vec<(usize, HashMap<&'static str, MemopStats>, u64)> = by_function
        .into_iter()
        .map(|(fn_id, ops)| {
            let total: u64 = ops.values().map(|s| s.anchors + s.helpers).sum();
            (fn_id, ops, total)
        })
        .collect();
    per_function.sort_by(|lhs, rhs| rhs.2.cmp(&lhs.2));

    let pct = |count: u64| -> f64 {
        if total_cycles == 0 {
            0.0
        } else {
            100.0 * count as f64 / total_cycles as f64
        }
    };

    const TOP_FUNCTIONS: usize = 20;
    for (rank, (fn_id, ops, total)) in per_function.iter().take(TOP_FUNCTIONS).enumerate() {
        let anchors: u64 = ops.values().map(|s| s.anchors).sum();
        let helpers: u64 = ops.values().map(|s| s.helpers).sum();
        let mut op_rows: Vec<(&&'static str, &MemopStats)> = ops.iter().collect();
        op_rows.sort_by(|a, b| (b.1.anchors + b.1.helpers).cmp(&(a.1.anchors + a.1.helpers)));
        let breakdown = op_rows
            .into_iter()
            .map(|(op, stats)| format!("{op}={}/{}", stats.anchors, stats.helpers))
            .collect::<Vec<_>>()
            .join(" ");
        info!(
            "trace memops-by-function: name={name} rank={} cycles={total} ({:.2}%) anchors={anchors} helpers={helpers} ops={breakdown} fn={}",
            rank + 1,
            pct(*total),
            symbols[*fn_id].name
        );
    }

    if !unattributed.is_empty() {
        let anchors: u64 = unattributed.values().map(|s| s.anchors).sum();
        let helpers: u64 = unattributed.values().map(|s| s.helpers).sum();
        let total = anchors + helpers;
        let mut op_rows: Vec<(&&'static str, &MemopStats)> = unattributed.iter().collect();
        op_rows.sort_by(|a, b| (b.1.anchors + b.1.helpers).cmp(&(a.1.anchors + a.1.helpers)));
        let breakdown = op_rows
            .into_iter()
            .map(|(op, stats)| format!("{op}={}/{}", stats.anchors, stats.helpers))
            .collect::<Vec<_>>()
            .join(" ");
        info!(
            "trace memops-by-function: name={name} rank=unattributed cycles={total} ({:.2}%) anchors={anchors} helpers={helpers} ops={breakdown}",
            pct(total)
        );
    }
}

fn log_crypto_stats(name: &str, stats: &CryptoTraceStats) {
    let avg_keccak_input_bytes = if stats.keccak_calls == 0 {
        0.0
    } else {
        stats.keccak_input_bytes as f64 / stats.keccak_calls as f64
    };
    let avg_sha256_input_bytes = if stats.precompile_sha256_calls == 0 {
        0.0
    } else {
        stats.precompile_sha256_input_bytes as f64 / stats.precompile_sha256_calls as f64
    };

    info!(
        "crypto detail: name={name} keccak_calls={} keccak_input_bytes={} avg_keccak_input_bytes={avg_keccak_input_bytes:.2} signer_recover_calls={} signer_verify_calls={} precompile_sha256_calls={} precompile_sha256_input_bytes={} avg_sha256_input_bytes={avg_sha256_input_bytes:.2} precompile_ecrecover_calls={} precompile_ecrecover_fallbacks={} precompile_p256verify_calls={} precompile_p256verify_fallbacks={}",
        stats.keccak_calls,
        stats.keccak_input_bytes,
        stats.signer_recover_calls,
        stats.signer_verify_calls,
        stats.precompile_sha256_calls,
        stats.precompile_sha256_input_bytes,
        stats.precompile_ecrecover_calls,
        stats.precompile_ecrecover_fallbacks,
        stats.precompile_p256verify_calls,
        stats.precompile_p256verify_fallbacks,
    );

    let hist = &stats.keccak_size_hist_136;
    let total_hist: u64 = hist.iter().sum();
    let pct = |count: u64| -> f64 {
        if total_hist == 0 {
            0.0
        } else {
            100.0 * count as f64 / total_hist as f64
        }
    };
    info!(
        "keccak size histogram (buckets of 136B, the Keccak rate): name={name} 0..136={} ({:.1}%) 136..272={} ({:.1}%) 272..408={} ({:.1}%) 408..544={} ({:.1}%) 544..680={} ({:.1}%) 680..816={} ({:.1}%) 816..952={} ({:.1}%) >=952={} ({:.1}%)",
        hist[0], pct(hist[0]),
        hist[1], pct(hist[1]),
        hist[2], pct(hist[2]),
        hist[3], pct(hist[3]),
        hist[4], pct(hist[4]),
        hist[5], pct(hist[5]),
        hist[6], pct(hist[6]),
        hist[7], pct(hist[7]),
    );
}

fn format_hist8(hist: &[u64; 8]) -> String {
    hist.iter()
        .enumerate()
        .map(|(i, count)| format!("{i}:{count}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn log_memops_stats(name: &str, stats: &MemopsTraceStats) {
    let avg_memcpy_bytes = if stats.memcpy_calls == 0 {
        0.0
    } else {
        stats.memcpy_bytes as f64 / stats.memcpy_calls as f64
    };
    let avg_memmove_bytes = if stats.memmove_calls == 0 {
        0.0
    } else {
        stats.memmove_bytes as f64 / stats.memmove_calls as f64
    };

    info!(
        "memops detail: name={name} memcpy_calls={} memcpy_bytes={} avg_memcpy_bytes={avg_memcpy_bytes:.2} memmove_calls={} memmove_bytes={} avg_memmove_bytes={avg_memmove_bytes:.2}",
        stats.memcpy_calls,
        stats.memcpy_bytes,
        stats.memmove_calls,
        stats.memmove_bytes,
    );
    info!(
        "memops path split: name={name} memcpy_byte_only={} memcpy_aligned_word={} memcpy_misaligned_word={} memmove_forward_byte_only={} memmove_forward_aligned_word={} memmove_forward_misaligned_word={} memmove_backward_byte_only={} memmove_backward_aligned_word={} memmove_backward_misaligned_word={}",
        stats.memcpy_byte_only_calls,
        stats.memcpy_aligned_word_calls,
        stats.memcpy_misaligned_word_calls,
        stats.memmove_forward_byte_only_calls,
        stats.memmove_forward_aligned_word_calls,
        stats.memmove_forward_misaligned_word_calls,
        stats.memmove_backward_byte_only_calls,
        stats.memmove_backward_aligned_word_calls,
        stats.memmove_backward_misaligned_word_calls,
    );

    let hist = &stats.memcpy_size_hist;
    let total_hist: u64 = hist.iter().sum();
    let pct = |count: u64| -> f64 {
        if total_hist == 0 {
            0.0
        } else {
            100.0 * count as f64 / total_hist as f64
        }
    };
    info!(
        "memcpy size histogram: name={name} 0..16={} ({:.1}%) 16..32={} ({:.1}%) 32..64={} ({:.1}%) 64..128={} ({:.1}%) 128..256={} ({:.1}%) 256..512={} ({:.1}%) 512..1024={} ({:.1}%) >=1024={} ({:.1}%)",
        hist[0], pct(hist[0]),
        hist[1], pct(hist[1]),
        hist[2], pct(hist[2]),
        hist[3], pct(hist[3]),
        hist[4], pct(hist[4]),
        hist[5], pct(hist[5]),
        hist[6], pct(hist[6]),
        hist[7], pct(hist[7]),
    );
    info!(
        "memcpy alignment histograms (mod 8): name={name} src=[{}] dst=[{}] xor=[{}]",
        format_hist8(&stats.memcpy_src_align_hist),
        format_hist8(&stats.memcpy_dst_align_hist),
        format_hist8(&stats.memcpy_align_diff_hist),
    );
}

fn make_guest_program() -> Program {
    let mut program = Program::new("stateless-evm-guest");
    program.set_func("stateless_validate");
    program.set_std(true);
    program.set_heap_size(268_435_456);
    program.set_stack_size(262_144);
    program.set_max_input_size(16_777_216);
    program.set_max_output_size(256);
    program
}

fn compile_guest(target_dir: &Path, keccak_backend: KeccakBackend) -> Program {
    let target_dir = target_dir.join(keccak_backend.target_dir_component());
    let target_dir = target_dir.to_string_lossy();
    let compile_start = Instant::now();
    let mut program = make_guest_program();
    let extra_features = keccak_backend.extra_guest_features();
    let mut compute_advice_features = Vec::with_capacity(1 + extra_features.len());
    compute_advice_features.push("compute_advice");
    compute_advice_features.extend(extra_features.iter().copied());
    program.build_with_features(&target_dir, &compute_advice_features);
    program.build_with_features(&target_dir, extra_features);
    info!(
        "guest compile time: {:?} keccak_backend={keccak_backend:?}",
        compile_start.elapsed()
    );
    program
}

fn recover_signers(stateless_input: &StatelessInput) -> Result<Vec<UncompressedPublicKey>> {
    stateless_input
        .block
        .body
        .transactions
        .iter()
        .enumerate()
        .map(|(i, tx)| {
            let key = tx
                .signature()
                .recover_from_prehash(&tx.signature_hash())
                .with_context(|| format!("failed to recover signer for tx #{i}"))?;
            let encoded = key.to_encoded_point(false);
            let public_key: [u8; 65] = encoded
                .as_bytes()
                .try_into()
                .map_err(|_| eyre::eyre!("unexpected uncompressed public key length"))?;
            Ok(UncompressedPublicKey(public_key))
        })
        .collect()
}

fn load_json_input(bytes: &[u8]) -> Result<LoadedInput> {
    if let Ok(fixture) = serde_json::from_slice::<StatelessValidatorFixture>(bytes) {
        let prepared = PreparedStatelessInput {
            public_keys: recover_signers(&fixture.stateless_input)?,
            stateless_input: fixture.stateless_input,
        };
        return Ok(LoadedInput {
            name: fixture.name,
            expected_success: Some(fixture.success),
            prepared,
        });
    }

    let stateless_input: StatelessInput = serde_json::from_slice(bytes)
        .wrap_err("failed to parse JSON as an upstream fixture or StatelessInput")?;
    let name = format!("block-{}", hex::encode(stateless_input.block.hash_slow().0));
    let prepared = PreparedStatelessInput {
        public_keys: recover_signers(&stateless_input)?,
        stateless_input,
    };
    Ok(LoadedInput {
        name,
        expected_success: None,
        prepared,
    })
}

fn load_input(path: &Path) -> Result<LoadedInput> {
    let bytes =
        fs::read(path).wrap_err_with(|| format!("failed to read input {}", path.display()))?;
    if path.extension().and_then(std::ffi::OsStr::to_str) == Some("json") {
        return load_json_input(&bytes);
    }

    let prepared: PreparedStatelessInput =
        postcard::from_bytes(&bytes).wrap_err("failed to parse postcard input")?;
    let name = format!(
        "block-{}",
        hex::encode(prepared.stateless_input.block.hash_slow().0)
    );
    Ok(LoadedInput {
        name,
        expected_success: None,
        prepared,
    })
}

fn check_expected_success(expected_success: Option<bool>, output: ValidationOutput) -> Result<()> {
    if let Some(expected) = expected_success {
        if expected != output.success {
            bail!(
                "validation success mismatch: expected {expected}, got {}",
                output.success
            );
        }
    }
    Ok(())
}

fn analyze_single(
    program: &Program,
    name: &str,
    input_bytes: &[u8],
    expected_success: Option<bool>,
) -> Result<AnalysisResult> {
    let guest_input =
        postcard::to_stdvec(&input_bytes).wrap_err("failed to serialize guest input")?;
    let start = Instant::now();
    let guest_binary = load_guest_binary(program);
    let summary = program.clone().trace_analyze::<F>(&guest_input, &[], &[]);
    let elapsed = start.elapsed();
    let total_cycles = summary.trace.len();
    let padded_cycles = if total_cycles == 0 {
        1
    } else {
        total_cycles.next_power_of_two()
    };
    let output: ValidationOutput =
        postcard::from_bytes(&summary.io_device.outputs).wrap_err("failed to decode output")?;
    check_expected_success(expected_success, output)?;
    log_trace_stats(name, &guest_binary, &summary.bytecode, &summary.trace);
    log_crypto_stats(name, &output.crypto_stats);
    log_memops_stats(name, &output.memops_stats);

    info!(
        "analysis complete: name={name} cycles={total_cycles} padded_cycles={padded_cycles} success={} block_hash=0x{} elapsed={elapsed:?}",
        output.success,
        hex::encode(output.block_hash),
    );
    Ok(AnalysisResult {
        name: name.to_owned(),
        cycles: total_cycles,
        padded_cycles,
        success: output.success,
    })
}

fn supported_input_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(OsStr::to_str),
        Some("json") | Some("bin")
    )
}

fn collect_input_paths(
    input: &Path,
    filter: &[String],
    limit: Option<usize>,
) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }

    if !input.is_dir() {
        bail!(
            "input path is neither a file nor a directory: {}",
            input.display()
        );
    }

    let mut paths: Vec<PathBuf> = fs::read_dir(input)
        .wrap_err_with(|| format!("failed to read directory {}", input.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .wrap_err_with(|| format!("failed to list directory {}", input.display()))?
        .into_iter()
        .filter(|path| path.is_file() && supported_input_file(path))
        .filter(|path| {
            if filter.is_empty() {
                return true;
            }
            let file_name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
            filter.iter().any(|needle| file_name.contains(needle))
        })
        .collect();
    paths.sort();

    if let Some(limit) = limit {
        paths.truncate(limit);
    }

    if paths.is_empty() {
        bail!(
            "no supported fixture files found in {} with the current filters",
            input.display()
        );
    }

    Ok(paths)
}

fn log_loaded_input(loaded: &LoadedInput, input_bytes: &[u8]) {
    let block_hash = loaded.prepared.stateless_input.block.hash_slow().0;
    let tx_count = loaded
        .prepared
        .stateless_input
        .block
        .body
        .transactions
        .len();
    info!(
        "loaded input: name={} block_hash=0x{} tx_count={} input_bytes={} expected_success={:?}",
        loaded.name,
        hex::encode(block_hash),
        tx_count,
        input_bytes.len(),
        loaded.expected_success,
    );
}

fn write_prepared_input(path: &Path, input_bytes: &[u8]) -> Result<()> {
    fs::write(path, input_bytes).wrap_err_with(|| format!("failed to write {}", path.display()))?;
    info!("wrote prepared input to {}", path.display());
    Ok(())
}

fn log_analysis_summary(results: &[AnalysisResult], elapsed: Duration) {
    if results.len() <= 1 {
        return;
    }

    let total_cycles = results
        .iter()
        .map(|result| result.cycles as u128)
        .sum::<u128>();
    let total_padded_cycles = results
        .iter()
        .map(|result| result.padded_cycles as u128)
        .sum::<u128>();
    let avg_cycles = total_cycles as f64 / results.len() as f64;
    let slowest = results
        .iter()
        .max_by_key(|result| result.cycles)
        .expect("results is non-empty");
    let fastest = results
        .iter()
        .min_by_key(|result| result.cycles)
        .expect("results is non-empty");
    let successes = results.iter().filter(|result| result.success).count();

    info!(
        "analysis summary: fixtures={} successes={} total_cycles={} total_padded_cycles={} avg_cycles={avg_cycles:.2} fastest={}({}) slowest={}({}) elapsed={elapsed:?}",
        results.len(),
        successes,
        total_cycles,
        total_padded_cycles,
        fastest.name,
        fastest.cycles,
        slowest.name,
        slowest.cycles,
    );
}

fn analyze_inputs(args: &Args) -> Result<()> {
    if args.input.is_dir() && args.write_prepared.is_some() {
        bail!("--write-prepared only supports a single input file");
    }

    let input_paths = collect_input_paths(&args.input, &args.filter, args.limit)?;
    let program = compile_guest(&args.target_dir, args.keccak_backend);
    let batch_start = Instant::now();
    let mut results = Vec::with_capacity(input_paths.len());
    let mut first_error = None;

    for input_path in input_paths {
        let outcome = (|| -> Result<AnalysisResult> {
            let loaded = load_input(&input_path)?;
            let input_bytes = postcard::to_stdvec(&loaded.prepared)
                .wrap_err("failed to serialize prepared input")?;
            log_loaded_input(&loaded, &input_bytes);

            if let Some(path) = &args.write_prepared {
                write_prepared_input(path, &input_bytes)?;
            }

            analyze_single(
                &program,
                &loaded.name,
                &input_bytes,
                loaded.expected_success,
            )
        })();

        match outcome {
            Ok(result) => results.push(result),
            Err(error) => {
                if !args.keep_going {
                    return Err(error);
                }
                info!("analysis failed for {}: {error}", input_path.display());
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }

    log_analysis_summary(&results, batch_start.elapsed());

    if let Some(error) = first_error {
        return Err(error);
    }

    Ok(())
}

fn prove(
    target_dir: &Path,
    keccak_backend: KeccakBackend,
    name: &str,
    input_bytes: &[u8],
    expected_success: Option<bool>,
) -> Result<()> {
    let mut program = compile_guest(target_dir, keccak_backend);
    let shared_preprocessing = guest::preprocess_shared_stateless_validate(&mut program)?;
    let prover_preprocessing =
        guest::preprocess_prover_stateless_validate(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_stateless_validate(shared_preprocessing, verifier_setup, None);

    let prove = guest::build_prover_stateless_validate(program, prover_preprocessing);
    let verify = guest::build_verifier_stateless_validate(verifier_preprocessing);

    let start = Instant::now();
    let (output, proof, program_io): (ValidationOutput, _, _) = prove(input_bytes);
    let prove_elapsed = start.elapsed();
    check_expected_success(expected_success, output)?;

    let start = Instant::now();
    let valid = verify(input_bytes, output, program_io.panic, proof);
    let verify_elapsed = start.elapsed();
    if !valid {
        bail!("proof verification failed");
    }

    info!(
        "prove+verify complete: name={name} success={} block_hash=0x{} prove_elapsed={prove_elapsed:?} verify_elapsed={verify_elapsed:?}",
        output.success,
        hex::encode(output.block_hash),
    );
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    match args.mode {
        Mode::Analyze => analyze_inputs(&args)?,
        Mode::Prove => {
            if args.input.is_dir() {
                bail!("prove mode only supports a single input file");
            }
            if args.limit.is_some() || !args.filter.is_empty() || args.keep_going {
                bail!("--limit, --filter, and --keep-going are only supported in analyze mode");
            }

            let loaded = load_input(&args.input)?;
            let input_bytes = postcard::to_stdvec(&loaded.prepared)
                .wrap_err("failed to serialize prepared input")?;
            log_loaded_input(&loaded, &input_bytes);

            if let Some(path) = &args.write_prepared {
                write_prepared_input(path, &input_bytes)?;
            }

            prove(
                &args.target_dir,
                args.keccak_backend,
                &loaded.name,
                &input_bytes,
                loaded.expected_success,
            )?;
        }
    }

    Ok(())
}
