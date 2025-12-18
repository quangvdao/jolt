//! Binary to generate the Jolt Prover DAG documentation.
//!
//! Usage:
//! ```bash
//! cargo run --bin generate-dag > book/src/how/architecture/jolt_dag.md
//! ```
//!
//! Or to verify the DAG is up-to-date (for CI):
//! ```bash
//! cargo run --bin generate-dag -- --check book/src/how/architecture/jolt_dag.md
//! ```

use jolt_core::utils::dag_generator::SumcheckDag;
use std::fs;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let dag = SumcheckDag::new();
    let markdown = dag.to_markdown();

    if args.len() > 2 && args[1] == "--check" {
        // Check mode: verify the file matches what we would generate
        let path = PathBuf::from(&args[2]);
        match fs::read_to_string(&path) {
            Ok(existing) => {
                if existing == markdown {
                    eprintln!("✓ {} is up to date", path.display());
                    std::process::exit(0);
                } else {
                    eprintln!("✗ {} is out of date!", path.display());
                    eprintln!("Run: cargo run --bin generate-dag > {}", path.display());
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("✗ Failed to read {}: {}", path.display(), e);
                eprintln!("Run: cargo run --bin generate-dag > {}", path.display());
                std::process::exit(1);
            }
        }
    } else if args.len() > 1 && args[1] == "--help" {
        eprintln!("Usage:");
        eprintln!("  generate-dag              # Print markdown to stdout");
        eprintln!("  generate-dag --check FILE # Check if FILE is up to date");
        std::process::exit(0);
    } else {
        // Default: print to stdout
        print!("{}", markdown);
    }
}

