//! Benchmark target for tail.

use std::io::{self, Result, Write};

fn main() -> Result<()> {
    writeln!(
        io::stdout().lock(),
        "tail: benchmark is not yet implemented"
    )
}
