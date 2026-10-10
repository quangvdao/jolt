//! Benchmark target for machinery.

use std::io::{self, Result, Write};

fn main() -> Result<()> {
    writeln!(
        io::stdout().lock(),
        "machinery: benchmark is not yet implemented"
    )
}
