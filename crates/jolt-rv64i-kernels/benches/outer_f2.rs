//! Benchmark target for outer f2.

use std::io::{self, Write};

fn main() -> io::Result<()> {
    writeln!(
        io::stdout().lock(),
        "outer_f2: benchmark is not yet implemented"
    )
}
