//! DAG generator for Jolt prover sumcheck dependencies.
//!
//! This module provides utilities to generate a Mermaid diagram of the
//! dependency graph between sumchecks in Jolt's proving system.
//!
//! The dependencies are extracted via **static analysis** of the source code,
//! looking for `get_*_opening(..., SumcheckId::X)` patterns.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p jolt-core --bin generate-dag > book/src/how/architecture/jolt_dag.md
//! ```

use crate::poly::opening_proof::SumcheckId;
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

/// Component category for color-coding sumchecks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    Spartan,
    Instruction,
    Ram,
    Registers,
    Bytecode,
    Mixed,
}

impl Component {
    pub fn color(&self) -> &'static str {
        match self {
            Component::Spartan => "#555",
            Component::Instruction => "#4a9eff",
            Component::Ram => "#ff6b6b",
            Component::Registers => "#4caf50",
            Component::Bytecode => "#00bcd4",
            Component::Mixed => "#ff9800",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Component::Spartan => "Spartan",
            Component::Instruction => "Instruction",
            Component::Ram => "RAM",
            Component::Registers => "Registers",
            Component::Bytecode => "Bytecode",
            Component::Mixed => "Mixed/Opening",
        }
    }
}

/// Maps each SumcheckId to its component
pub fn sumcheck_component(id: SumcheckId) -> Component {
    match id {
        SumcheckId::SpartanOuter
        | SumcheckId::SpartanProductVirtualization
        | SumcheckId::SpartanShift
        | SumcheckId::InstructionInputVirtualization => Component::Spartan,

        SumcheckId::InstructionClaimReduction
        | SumcheckId::InstructionReadRaf
        | SumcheckId::InstructionRaVirtualization => Component::Instruction,

        SumcheckId::RamReadWriteChecking
        | SumcheckId::RamRafEvaluation
        | SumcheckId::RamOutputCheck
        | SumcheckId::RamValEvaluation
        | SumcheckId::RamValFinalEvaluation
        | SumcheckId::RamRaClaimReduction
        | SumcheckId::RamHammingBooleanity
        | SumcheckId::RamRaVirtualization => Component::Ram,

        SumcheckId::RegistersClaimReduction
        | SumcheckId::RegistersReadWriteChecking
        | SumcheckId::RegistersValEvaluation => Component::Registers,

        SumcheckId::BytecodeReadRaf => Component::Bytecode,

        SumcheckId::Booleanity
        | SumcheckId::IncClaimReduction
        | SumcheckId::HammingWeightClaimReduction => Component::Mixed,
    }
}

/// Maps each SumcheckId to its stage number
pub fn sumcheck_stage(id: SumcheckId) -> u8 {
    match id {
        SumcheckId::SpartanOuter => 1,

        SumcheckId::SpartanProductVirtualization
        | SumcheckId::RamRafEvaluation
        | SumcheckId::RamReadWriteChecking
        | SumcheckId::RamOutputCheck
        | SumcheckId::InstructionClaimReduction => 2,

        SumcheckId::SpartanShift
        | SumcheckId::InstructionInputVirtualization
        | SumcheckId::RegistersClaimReduction => 3,

        SumcheckId::RegistersReadWriteChecking
        | SumcheckId::RamValEvaluation
        | SumcheckId::RamValFinalEvaluation => 4,

        SumcheckId::RegistersValEvaluation
        | SumcheckId::RamRaClaimReduction
        | SumcheckId::InstructionReadRaf => 5,

        SumcheckId::BytecodeReadRaf
        | SumcheckId::RamHammingBooleanity
        | SumcheckId::Booleanity
        | SumcheckId::RamRaVirtualization
        | SumcheckId::InstructionRaVirtualization
        | SumcheckId::IncClaimReduction => 6,

        SumcheckId::HammingWeightClaimReduction => 7,
    }
}

/// Short name for Mermaid node IDs
pub fn sumcheck_short_name(id: SumcheckId) -> &'static str {
    match id {
        SumcheckId::SpartanOuter => "SpartanOuter",
        SumcheckId::SpartanProductVirtualization => "SpartanProductVirt",
        SumcheckId::SpartanShift => "SpartanShift",
        SumcheckId::InstructionClaimReduction => "InstrClaimRed",
        SumcheckId::InstructionInputVirtualization => "InstrInputVirt",
        SumcheckId::InstructionReadRaf => "InstrReadRaf",
        SumcheckId::InstructionRaVirtualization => "InstrRaVirt",
        SumcheckId::RamReadWriteChecking => "RamRWCheck",
        SumcheckId::RamRafEvaluation => "RamRafEval",
        SumcheckId::RamOutputCheck => "RamOutCheck",
        SumcheckId::RamValEvaluation => "RamValEval",
        SumcheckId::RamValFinalEvaluation => "RamValFinal",
        SumcheckId::RamRaClaimReduction => "RamRaClaimRed",
        SumcheckId::RamHammingBooleanity => "RamHammingBool",
        SumcheckId::RamRaVirtualization => "RamRaVirt",
        SumcheckId::RegistersClaimReduction => "RegClaimRed",
        SumcheckId::RegistersReadWriteChecking => "RegRWCheck",
        SumcheckId::RegistersValEvaluation => "RegValEval",
        SumcheckId::BytecodeReadRaf => "BytecodeReadRaf",
        SumcheckId::Booleanity => "Booleanity",
        SumcheckId::IncClaimReduction => "IncClaimRed",
        SumcheckId::HammingWeightClaimReduction => "HWClaimRed",
    }
}

/// Full display name for Mermaid node labels
pub fn sumcheck_display_name(id: SumcheckId) -> &'static str {
    match id {
        SumcheckId::SpartanOuter => "SpartanOuter",
        SumcheckId::SpartanProductVirtualization => "SpartanProductVirtualization",
        SumcheckId::SpartanShift => "SpartanShift",
        SumcheckId::InstructionClaimReduction => "InstructionClaimReduction",
        SumcheckId::InstructionInputVirtualization => "InstructionInputVirtualization",
        SumcheckId::InstructionReadRaf => "InstructionReadRaf",
        SumcheckId::InstructionRaVirtualization => "InstructionRaVirtualization",
        SumcheckId::RamReadWriteChecking => "RamReadWriteChecking",
        SumcheckId::RamRafEvaluation => "RamRafEvaluation",
        SumcheckId::RamOutputCheck => "RamOutputCheck",
        SumcheckId::RamValEvaluation => "RamValEvaluation",
        SumcheckId::RamValFinalEvaluation => "RamValFinalEvaluation",
        SumcheckId::RamRaClaimReduction => "RamRaClaimReduction",
        SumcheckId::RamHammingBooleanity => "RamHammingBooleanity",
        SumcheckId::RamRaVirtualization => "RamRaVirtualization",
        SumcheckId::RegistersClaimReduction => "RegistersClaimReduction",
        SumcheckId::RegistersReadWriteChecking => "RegistersReadWriteChecking",
        SumcheckId::RegistersValEvaluation => "RegistersValEvaluation",
        SumcheckId::BytecodeReadRaf => "BytecodeReadRaf",
        SumcheckId::Booleanity => "Booleanity",
        SumcheckId::IncClaimReduction => "IncClaimReduction",
        SumcheckId::HammingWeightClaimReduction => "HammingWeightClaimReduction",
    }
}

/// Parse a SumcheckId from its string representation
/// Normalize function calls by collapsing newlines and extra whitespace
/// This makes it easier to match multiline function calls with regex
fn normalize_function_calls(content: &str) -> String {
    // Replace newlines and multiple spaces with single space
    let re = Regex::new(r"\s+").unwrap();
    re.replace_all(content, " ").to_string()
}

fn parse_sumcheck_id(s: &str) -> Option<SumcheckId> {
    match s {
        "SpartanOuter" => Some(SumcheckId::SpartanOuter),
        "SpartanProductVirtualization" => Some(SumcheckId::SpartanProductVirtualization),
        "SpartanShift" => Some(SumcheckId::SpartanShift),
        "InstructionClaimReduction" => Some(SumcheckId::InstructionClaimReduction),
        "InstructionInputVirtualization" => Some(SumcheckId::InstructionInputVirtualization),
        "InstructionReadRaf" => Some(SumcheckId::InstructionReadRaf),
        "InstructionRaVirtualization" => Some(SumcheckId::InstructionRaVirtualization),
        "RamReadWriteChecking" => Some(SumcheckId::RamReadWriteChecking),
        "RamRafEvaluation" => Some(SumcheckId::RamRafEvaluation),
        "RamOutputCheck" => Some(SumcheckId::RamOutputCheck),
        "RamValEvaluation" => Some(SumcheckId::RamValEvaluation),
        "RamValFinalEvaluation" => Some(SumcheckId::RamValFinalEvaluation),
        "RamRaClaimReduction" => Some(SumcheckId::RamRaClaimReduction),
        "RamHammingBooleanity" => Some(SumcheckId::RamHammingBooleanity),
        "RamRaVirtualization" => Some(SumcheckId::RamRaVirtualization),
        "RegistersClaimReduction" => Some(SumcheckId::RegistersClaimReduction),
        "RegistersReadWriteChecking" => Some(SumcheckId::RegistersReadWriteChecking),
        "RegistersValEvaluation" => Some(SumcheckId::RegistersValEvaluation),
        "BytecodeReadRaf" => Some(SumcheckId::BytecodeReadRaf),
        "Booleanity" => Some(SumcheckId::Booleanity),
        "IncClaimReduction" => Some(SumcheckId::IncClaimReduction),
        "HammingWeightClaimReduction" => Some(SumcheckId::HammingWeightClaimReduction),
        _ => None,
    }
}

/// Mapping from file paths to which sumcheck(s) they implement
fn file_to_sumcheck_mapping() -> HashMap<&'static str, SumcheckId> {
    let mut map = HashMap::new();

    // Spartan
    map.insert("zkvm/spartan/outer.rs", SumcheckId::SpartanOuter);
    map.insert("zkvm/spartan/product.rs", SumcheckId::SpartanProductVirtualization);
    map.insert("zkvm/spartan/shift.rs", SumcheckId::SpartanShift);
    map.insert("zkvm/spartan/instruction_input.rs", SumcheckId::InstructionInputVirtualization);

    // Instruction
    map.insert(
        "zkvm/claim_reductions/instruction_lookups.rs",
        SumcheckId::InstructionClaimReduction,
    );
    map.insert(
        "zkvm/instruction_lookups/read_raf_checking.rs",
        SumcheckId::InstructionReadRaf,
    );
    map.insert(
        "zkvm/instruction_lookups/ra_virtual.rs",
        SumcheckId::InstructionRaVirtualization,
    );

    // RAM
    map.insert("zkvm/ram/read_write_checking.rs", SumcheckId::RamReadWriteChecking);
    map.insert("zkvm/ram/raf_evaluation.rs", SumcheckId::RamRafEvaluation);
    map.insert("zkvm/ram/output_check.rs", SumcheckId::RamOutputCheck);
    map.insert("zkvm/ram/val_evaluation.rs", SumcheckId::RamValEvaluation);
    map.insert("zkvm/ram/val_final.rs", SumcheckId::RamValFinalEvaluation);
    map.insert("zkvm/claim_reductions/ram_ra.rs", SumcheckId::RamRaClaimReduction);
    map.insert("zkvm/ram/hamming_booleanity.rs", SumcheckId::RamHammingBooleanity);
    map.insert("zkvm/ram/ra_virtual.rs", SumcheckId::RamRaVirtualization);

    // Registers
    map.insert("zkvm/claim_reductions/registers.rs", SumcheckId::RegistersClaimReduction);
    map.insert(
        "zkvm/registers/read_write_checking.rs",
        SumcheckId::RegistersReadWriteChecking,
    );
    map.insert("zkvm/registers/val_evaluation.rs", SumcheckId::RegistersValEvaluation);

    // Bytecode
    map.insert("zkvm/bytecode/read_raf_checking.rs", SumcheckId::BytecodeReadRaf);

    // Mixed
    map.insert("subprotocols/booleanity.rs", SumcheckId::Booleanity);
    map.insert("zkvm/claim_reductions/increments.rs", SumcheckId::IncClaimReduction);
    map.insert(
        "zkvm/claim_reductions/hamming_weight.rs",
        SumcheckId::HammingWeightClaimReduction,
    );

    map
}

/// A dependency edge in the DAG
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DependencyEdge {
    pub from: SumcheckId,
    pub to: SumcheckId,
    pub polynomials: BTreeSet<String>,
}

/// The complete DAG of sumcheck dependencies
pub struct SumcheckDag {
    pub edges: Vec<DependencyEdge>,
}

impl Default for SumcheckDag {
    fn default() -> Self {
        Self::new()
    }
}

impl SumcheckDag {
    /// Build the DAG via static analysis of the source code.
    ///
    /// This scans all relevant source files looking for patterns like:
    /// - `get_virtual_polynomial_opening(..., SumcheckId::X)`
    /// - `get_committed_polynomial_opening(..., SumcheckId::X)`
    ///
    /// The consuming sumcheck is determined by which file contains the call.
    pub fn new() -> Self {
        Self::from_source_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("src"))
    }

    /// Build the DAG from a specific source directory
    pub fn from_source_dir(src_dir: PathBuf) -> Self {
        let file_mapping = file_to_sumcheck_mapping();

        // Regex to find get_*_opening calls with SumcheckId
        // Uses (?s) for DOTALL mode so . matches newlines
        // Matches patterns like:
        //   get_virtual_polynomial_opening(VirtualPolynomial::X, SumcheckId::Y)
        //   .get_committed_polynomial_opening(CommittedPolynomial::X, SumcheckId::Y)
        // even when split across multiple lines
        let opening_pattern = Regex::new(
            r"(?s)get_(?:virtual|committed)_polynomial_opening\s*\(\s*(?:VirtualPolynomial|CommittedPolynomial)::(\w+)(?:\([^)]*\))?\s*,\s*SumcheckId::(\w+)"
        ).unwrap();

        // Also match pattern where SumcheckId is on the next line after the poly type
        let opening_pattern_multiline = Regex::new(
            r"(?s)get_(?:virtual|committed)_polynomial_opening\s*\([^,]+,\s*SumcheckId::(\w+)"
        ).unwrap();

        // Collect all edges: (from_sumcheck, to_sumcheck, polynomial_name)
        let mut edge_map: BTreeMap<(SumcheckId, SumcheckId), BTreeSet<String>> = BTreeMap::new();

        for (file_suffix, consuming_sumcheck) in &file_mapping {
            let file_path = src_dir.join(file_suffix);
            if !file_path.exists() {
                eprintln!("Warning: File not found: {}", file_path.display());
                continue;
            }

            let content = match fs::read_to_string(&file_path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Warning: Failed to read {}: {}", file_path.display(), e);
                    continue;
                }
            };

            // First, normalize content by collapsing newlines within function calls
            // This makes regex matching easier
            let normalized = normalize_function_calls(&content);

            // Find all get_*_opening calls with explicit polynomial type
            for cap in opening_pattern.captures_iter(&normalized) {
                let poly_name = cap.get(1).map(|m| m.as_str()).unwrap_or("unknown");
                let source_sumcheck_str = cap.get(2).map(|m| m.as_str()).unwrap_or("");

                if let Some(source_sumcheck) = parse_sumcheck_id(source_sumcheck_str) {
                    // Only add edge if source != consumer and source is in an earlier stage
                    if source_sumcheck != *consuming_sumcheck
                        && sumcheck_stage(source_sumcheck) < sumcheck_stage(*consuming_sumcheck)
                    {
                        edge_map
                            .entry((source_sumcheck, *consuming_sumcheck))
                            .or_default()
                            .insert(poly_name.to_string());
                    }
                }
            }

            // Also find calls where polynomial is a variable (we can't extract the poly name)
            for cap in opening_pattern_multiline.captures_iter(&normalized) {
                let source_sumcheck_str = cap.get(1).map(|m| m.as_str()).unwrap_or("");

                if let Some(source_sumcheck) = parse_sumcheck_id(source_sumcheck_str) {
                    if source_sumcheck != *consuming_sumcheck
                        && sumcheck_stage(source_sumcheck) < sumcheck_stage(*consuming_sumcheck)
                    {
                        // Don't add duplicate if we already have this edge with a known poly
                        edge_map.entry((source_sumcheck, *consuming_sumcheck)).or_default();
                    }
                }
            }
        }

        // Convert to edges
        let edges: Vec<DependencyEdge> = edge_map
            .into_iter()
            .map(|((from, to), polynomials)| DependencyEdge {
                from,
                to,
                polynomials,
            })
            .collect();

        Self { edges }
    }

    /// Get all sumchecks in a given stage
    pub fn sumchecks_in_stage(&self, stage: u8) -> Vec<SumcheckId> {
        use strum::IntoEnumIterator;
        SumcheckId::iter()
            .filter(|id| sumcheck_stage(*id) == stage)
            .collect()
    }

    /// Generate Mermaid diagram source code
    pub fn to_mermaid(&self) -> String {
        let mut output = String::new();

        output.push_str("flowchart TD\n");

        // Group sumchecks by stage
        let mut stages: BTreeMap<u8, Vec<SumcheckId>> = BTreeMap::new();
        for stage in 1..=8 {
            stages.insert(stage, self.sumchecks_in_stage(stage));
        }

        // Add subgraphs for each stage
        for (stage, sumchecks) in &stages {
            if sumchecks.is_empty() && *stage != 8 {
                continue;
            }

            output.push_str(&format!("    subgraph Stage{}[\"Stage {}\"]\n", stage, stage));

            if *stage == 8 {
                output.push_str("        BatchOpening[Batch Opening Proof]\n");
            } else {
                for id in sumchecks {
                    output.push_str(&format!(
                        "        {}[{}]\n",
                        sumcheck_short_name(*id),
                        sumcheck_display_name(*id)
                    ));
                }
            }
            output.push_str("    end\n\n");
        }

        // Group edges by (from_stage, to_stage) for comments
        let mut current_stage_pair = (0u8, 0u8);
        for edge in &self.edges {
            let from_stage = sumcheck_stage(edge.from);
            let to_stage = sumcheck_stage(edge.to);
            let stage_pair = (from_stage, to_stage);

            if stage_pair != current_stage_pair {
                output.push_str(&format!(
                    "    %% ============ Stage {} → Stage {} ============\n",
                    from_stage, to_stage
                ));
                current_stage_pair = stage_pair;
            }

            let label = if edge.polynomials.is_empty() {
                "(deps)".to_string()
            } else {
                edge.polynomials.iter().cloned().collect::<Vec<_>>().join(", ")
            };
            output.push_str(&format!(
                "    {} -->|\"{}\"| {}\n",
                sumcheck_short_name(edge.from),
                label,
                sumcheck_short_name(edge.to)
            ));
        }

        // Add Stage 7 → Stage 8 and Stage 6 → Stage 8 edges (always present)
        output.push_str("\n    %% ============ Stage 7 → Stage 8 ============\n");
        output.push_str("    HWClaimRed -->|\"All RA polys\"| BatchOpening\n");
        output.push_str("\n    %% ============ Stage 6 → Stage 8 ============\n");
        output.push_str("    IncClaimRed -->|\"RamInc, RdInc\"| BatchOpening\n");

        // Add color class definitions
        output.push_str("\n    %% ============ Color Definitions ============\n");

        // Group sumchecks by component for class definitions
        let mut component_sumchecks: BTreeMap<Component, BTreeSet<&'static str>> = BTreeMap::new();
        for stage in 1..=7 {
            for id in self.sumchecks_in_stage(stage) {
                let component = sumcheck_component(id);
                component_sumchecks
                    .entry(component)
                    .or_default()
                    .insert(sumcheck_short_name(id));
            }
        }

        for component in [
            Component::Spartan,
            Component::Instruction,
            Component::Ram,
            Component::Registers,
            Component::Bytecode,
            Component::Mixed,
        ] {
            let class_name = component.name().to_lowercase().replace('/', "");
            output.push_str(&format!(
                "    classDef {} fill:{},stroke:#fff,color:#fff\n",
                class_name,
                component.color()
            ));
        }

        output.push('\n');

        // Apply classes
        for component in [
            Component::Spartan,
            Component::Instruction,
            Component::Ram,
            Component::Registers,
            Component::Bytecode,
            Component::Mixed,
        ] {
            let class_name = component.name().to_lowercase().replace('/', "");
            if let Some(sumchecks) = component_sumchecks.get(&component) {
                let mut names: Vec<_> = sumchecks.iter().copied().collect();
                if component == Component::Mixed {
                    names.push("BatchOpening");
                }
                output.push_str(&format!("    class {} {}\n", names.join(","), class_name));
            }
        }

        output
    }

    /// Generate full markdown documentation
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# Jolt Prover DAG\n\n");
        output.push_str(
            "This document shows the dependency graph between sumchecks in Jolt's proving system.\n\n",
        );
        output.push_str(
            "> **Note:** This file is auto-generated via static analysis of the codebase.\n",
        );
        output.push_str("> Run: `cargo run -p jolt-core --bin generate-dag > book/src/how/architecture/jolt_dag.md`\n");
        output.push_str("> Do not edit manually.\n\n");

        // Color Legend
        output.push_str("## Color Legend\n\n");
        output.push_str("| Color | Component | Description |\n");
        output.push_str("|-------|-----------|-------------|\n");
        output.push_str(&format!(
            "| Gray (`{}`) | **Spartan** | R1CS constraint system sumchecks |\n",
            Component::Spartan.color()
        ));
        output.push_str(&format!(
            "| Blue (`{}`) | **Instruction** | Instruction lookup sumchecks |\n",
            Component::Instruction.color()
        ));
        output.push_str(&format!(
            "| Red (`{}`) | **RAM** | RAM read/write and address sumchecks |\n",
            Component::Ram.color()
        ));
        output.push_str(&format!(
            "| Green (`{}`) | **Registers** | Register read/write sumchecks |\n",
            Component::Registers.color()
        ));
        output.push_str(&format!(
            "| Cyan (`{}`) | **Bytecode** | Bytecode read/address sumchecks |\n",
            Component::Bytecode.color()
        ));
        output.push_str(&format!(
            "| Orange (`{}`) | **Mixed/Opening** | Cross-component sumchecks (Booleanity, Inc, HW, BatchOpening) |\n",
            Component::Mixed.color()
        ));
        output.push('\n');

        // Stage Overview
        output.push_str("## Stage Overview\n\n");
        output.push_str("| Stage | Sumchecks |\n");
        output.push_str("|-------|-----------|");
        for stage in 1..=8 {
            let sumchecks = self.sumchecks_in_stage(stage);
            let names: Vec<_> = if stage == 8 {
                vec!["Batch Opening Proof".to_string()]
            } else {
                sumchecks
                    .iter()
                    .map(|id| sumcheck_display_name(*id).to_string())
                    .collect()
            };
            output.push_str(&format!("\n| {} | {} |", stage, names.join(", ")));
        }
        output.push_str("\n\n");

        // Full DAG
        output.push_str("## Full DAG with Dependencies\n\n");
        output.push_str("```mermaid\n");
        output.push_str(&self.to_mermaid());
        output.push_str("```\n\n");

        // Extraction info
        output.push_str("## How Dependencies are Extracted\n\n");
        output.push_str("Dependencies are extracted via static analysis of the source code by scanning for:\n\n");
        output.push_str("```rust\n");
        output.push_str("// Pattern: get_*_opening(..., SumcheckId::SOURCE)\n");
        output.push_str("// The SOURCE sumcheck produces the opening\n");
        output.push_str("// The file containing the call determines the consuming sumcheck\n");
        output.push_str("accumulator.get_virtual_polynomial_opening(\n");
        output.push_str("    VirtualPolynomial::X,\n");
        output.push_str("    SumcheckId::SOURCE,  // <- This is the producer\n");
        output.push_str(");\n");
        output.push_str("```\n\n");

        // SumcheckId Reference
        output.push_str("## SumcheckId Reference\n\n");
        output.push_str("All sumcheck identifiers (from `opening_proof.rs`):\n\n");
        output.push_str("```rust\n");
        output.push_str("pub enum SumcheckId {\n");
        for stage in 1..=7 {
            for id in self.sumchecks_in_stage(stage) {
                output.push_str(&format!("    {},\n", sumcheck_display_name(id)));
            }
        }
        output.push_str("}\n");
        output.push_str("```\n");

        output
    }
}

// Implement PartialOrd/Ord for Component for BTreeMap
impl PartialOrd for Component {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Component {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_generation() {
        let dag = SumcheckDag::new();

        // Should have found some edges
        assert!(!dag.edges.is_empty(), "DAG should have edges");

        // Print for debugging
        eprintln!("Found {} edges:", dag.edges.len());
        for edge in &dag.edges {
            eprintln!(
                "  {:?} -> {:?}: {:?}",
                edge.from, edge.to, edge.polynomials
            );
        }
    }

    #[test]
    fn test_mermaid_generation() {
        let dag = SumcheckDag::new();
        let mermaid = dag.to_mermaid();
        assert!(mermaid.contains("flowchart TD"));
        assert!(mermaid.contains("SpartanOuter"));
        assert!(mermaid.contains("Stage 1"));
    }

    #[test]
    fn test_markdown_generation() {
        let dag = SumcheckDag::new();
        let markdown = dag.to_markdown();
        assert!(markdown.contains("# Jolt Prover DAG"));
        assert!(markdown.contains("```mermaid"));
        assert!(markdown.contains("static analysis"));
    }
}
