//! Benchmark target for probe.

use std::io::{self, Write};

fn main() -> io::Result<()> {
    writeln!(
        io::stdout().lock(),
        "probe: benchmark is not yet implemented"
    )
}
